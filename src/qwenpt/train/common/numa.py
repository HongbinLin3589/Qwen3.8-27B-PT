"""Bind a torchrun worker's CPU cores and future host allocations near its GPU."""

import ctypes
import errno
import json
import os
from pathlib import Path


def _parse_cpu_list(value: str) -> set[int]:
    result = set()
    for part in value.strip().split(","):
        if part:
            bounds = [int(number) for number in part.split("-")]
            result.update(range(bounds[0], bounds[-1] + 1))
    return result


def _visible_gpu_pci_ids(count: int) -> list[str]:
    # CUDA ordinals honor CUDA_VISIBLE_DEVICES, unlike nvidia-smi's physical indices.
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

    def checked(function, *args):
        result = function(*args)
        if result:
            raise RuntimeError(f"CUDA driver query failed with code {result}")

    checked(driver.cuInit, 0)
    visible_count = ctypes.c_int()
    checked(driver.cuDeviceGetCount, ctypes.byref(visible_count))
    if count > visible_count.value:
        raise RuntimeError(f"Need {count} visible GPUs, found {visible_count.value}")
    pci_ids = []
    for ordinal in range(count):
        device = ctypes.c_int()
        checked(driver.cuDeviceGet, ctypes.byref(device), ordinal)
        bus_id = ctypes.create_string_buffer(32)
        checked(driver.cuDeviceGetPCIBusId, bus_id, len(bus_id), device)
        domain, bus, slot = bus_id.value.decode().lower().split(":")
        pci_ids.append(f"{int(domain, 16):04x}:{bus}:{slot}")
    return pci_ids


def _allowed_memory_nodes() -> set[int]:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Mems_allowed_list:"):
            return _parse_cpu_list(line.split(":", 1)[1])
    raise RuntimeError("Cannot read this process's allowed memory nodes")


def _local_cpus(node: int, allowed: set[int], sysfs: Path) -> list[int]:
    candidates = _parse_cpu_list((sysfs / f"devices/system/node/node{node}/cpulist").read_text()) & allowed
    cores = {}
    for cpu in sorted(candidates):
        topology = sysfs / f"devices/system/cpu/cpu{cpu}/topology"
        key = tuple((topology / name).read_text().strip() for name in ("physical_package_id", "core_id"))
        cores.setdefault(key, cpu)
    return sorted(cores.values())


def _bind_memory(node: int, allowed_nodes: set[int]) -> dict:
    library = ctypes.CDLL("libnuma.so.1", use_errno=True)
    library.numa_num_possible_nodes.argtypes = []
    library.numa_num_possible_nodes.restype = ctypes.c_int
    library.set_mempolicy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
    library.set_mempolicy.restype = ctypes.c_long
    library.get_mempolicy.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong,
        ctypes.c_void_p, ctypes.c_ulong,
    ]
    library.get_mempolicy.restype = ctypes.c_long
    mask_bits = max(library.numa_num_possible_nodes(), max(allowed_nodes) + 1)
    # Linux get_nodes decrements maxnode; match libnuma 2.0.16's bmp->size + 1.
    # The kernel-sized mask also satisfies get_mempolicy on older kernels.
    maxnode = mask_bits + 1
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    mask_type = ctypes.c_ulong * ((mask_bits + bits - 1) // bits)
    desired = mask_type()
    desired[node // bits] |= 1 << (node % bits)
    if library.set_mempolicy(2, desired, maxnode) != 0:  # Linux MPOL_BIND.
        error = ctypes.get_errno() or errno.EINVAL
        raise OSError(error, f"set_mempolicy(MPOL_BIND, node={node}): {os.strerror(error)}")
    actual_mode, actual_mask = ctypes.c_int(), mask_type()
    if library.get_mempolicy(ctypes.byref(actual_mode), actual_mask, maxnode, None, 0) != 0:
        error = ctypes.get_errno() or errno.EINVAL
        raise OSError(error, f"get_mempolicy: {os.strerror(error)}")
    actual_nodes = [index for index in range(mask_bits) if actual_mask[index // bits] & (1 << (index % bits))]
    if actual_mode.value != 2 or actual_nodes != [node]:
        raise RuntimeError(f"Memory policy verification failed: mode={actual_mode.value}, nodes={actual_nodes}")
    return {"mode": "MPOL_BIND", "mode_id": actual_mode.value, "nodes": actual_nodes}


def bind_to_local_gpu(threads_per_rank: int = 4, receipt_dir: str | Path | None = None) -> dict:
    """Bind before model creation; return and optionally persist the verified rank receipt.

    ``threads_per_rank`` selects distinct physical cores per local GPU rank and
    sets OMP/MKL thread counts; ``receipt_dir`` receives ``rank_<rank>.json``.
    Insufficient allowed local cores/memory nodes or failed policy calls raise.
    """
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if threads_per_rank < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid local rank/world size or threads_per_rank")
    sysfs = Path("/sys")
    pci_ids = _visible_gpu_pci_ids(world_size)
    gpu_nodes = [int((sysfs / f"bus/pci/devices/{pci_id}/numa_node").read_text()) for pci_id in pci_ids]
    node = gpu_nodes[rank]
    allowed_nodes = _allowed_memory_nodes()
    if node < 0 or node not in allowed_nodes:
        raise RuntimeError(f"GPU NUMA node {node} is unavailable in allowed memory nodes {sorted(allowed_nodes)}")
    allowed_cpus = os.sched_getaffinity(0)
    cores = _local_cpus(node, allowed_cpus, sysfs)
    peers = [index for index, peer_node in enumerate(gpu_nodes) if peer_node == node]
    required = threads_per_rank * len(peers)
    if len(cores) < required:
        raise RuntimeError(f"NUMA node {node} needs {required} allowed physical cores for ranks {peers}; found {cores}")
    start = peers.index(rank) * threads_per_rank
    selected = cores[start:start + threads_per_rank]
    # Bind already-created driver/helper threads too; new threads inherit this mask.
    bound_thread_ids = []
    for thread_id in sorted(int(path.name) for path in Path("/proc/self/task").iterdir()):
        try:
            os.sched_setaffinity(thread_id, selected)
        except ProcessLookupError:
            continue
        if os.sched_getaffinity(thread_id) != set(selected):
            raise RuntimeError(f"CPU affinity verification failed for thread {thread_id}")
        bound_thread_ids.append(thread_id)
    policy = _bind_memory(node, allowed_nodes)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_rank)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_rank)
    receipt = {
        "pid": os.getpid(), "local_rank": rank, "local_world_size": world_size,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_pci_bus_id": pci_ids[rank], "gpu_numa_node": node,
        "allowed_cpus_before": sorted(allowed_cpus), "allowed_memory_nodes": sorted(allowed_nodes),
        "cpu_affinity": selected, "threads_per_rank": threads_per_rank,
        "physical_cores": len(selected), "bound_thread_ids": bound_thread_ids, "memory_policy": policy,
    }
    if receipt_dir is not None:
        directory = Path(receipt_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"rank_{rank}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
