"""Verify NUMA selection and policy receipts without GPU queries or live binding."""

import ctypes
import errno
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qwenpt.train.common import numa


def test_parse_cpu_list_and_physical_core_selection(tmp_path):
    assert numa._parse_cpu_list("0-2,5,9-10\n") == {0, 1, 2, 5, 9, 10}
    node = tmp_path / "devices/system/node/node1"
    node.mkdir(parents=True)
    (node / "cpulist").write_text("4-7,12-15")
    for cpu in range(4, 8):
        for sibling in (cpu, cpu + 8):
            topology = tmp_path / f"devices/system/cpu/cpu{sibling}/topology"
            topology.mkdir(parents=True)
            (topology / "physical_package_id").write_text("1")
            (topology / "core_id").write_text(str(cpu))
    # SMT siblings do not qualify as additional physical cores.
    assert numa._local_cpus(1, set(range(16)), tmp_path) == [4, 5, 6, 7]
    assert numa._local_cpus(1, {0, 4, 13}, tmp_path) == [4, 13]


def test_driver_queries_use_cuda_ordinals_and_normalize_pci_domain(monkeypatch):
    driver = SimpleNamespace(cuInit=Mock(return_value=0))

    def count(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 2
        return 0

    def device(pointer, ordinal):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = ordinal + 70
        return 0

    def pci(buffer, length, device):
        buffer.value = [b"00000000:a1:00.0", b"00000000:31:00.0"][device.value - 70]
        return 0

    driver.cuDeviceGetCount = Mock(side_effect=count)
    driver.cuDeviceGet = Mock(side_effect=device)
    driver.cuDeviceGetPCIBusId = Mock(side_effect=pci)
    monkeypatch.setattr(numa.ctypes, "CDLL", Mock(return_value=driver))
    assert numa._visible_gpu_pci_ids(2) == ["0000:a1:00.0", "0000:31:00.0"]
    with pytest.raises(RuntimeError, match="Need 3 visible GPUs"):
        numa._visible_gpu_pci_ids(3)
    driver.cuInit.return_value = 100
    with pytest.raises(RuntimeError, match="code 100"):
        numa._visible_gpu_pci_ids(2)


@pytest.fixture
def rank_environment(monkeypatch, tmp_path):
    for key, value in {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "2", "CUDA_VISIBLE_DEVICES": "3,1"}.items():
        monkeypatch.setenv(key, value)
    sysfs = tmp_path / "sys"
    for pci in ("0000:a1:00.0", "0000:31:00.0"):
        directory = sysfs / f"bus/pci/devices/{pci}"
        directory.mkdir(parents=True)
        (directory / "numa_node").write_text("1")
    task_directory = tmp_path / "tasks"
    task_directory.mkdir()
    for thread_id in (100, 101):
        (task_directory / str(thread_id)).mkdir()
    monkeypatch.setattr(numa, "Path", lambda value: {"/sys": sysfs, "/proc/self/task": task_directory}.get(str(value), Path(value)))
    monkeypatch.setattr(numa, "_visible_gpu_pci_ids", lambda count: ["0000:a1:00.0", "0000:31:00.0"][:count])
    monkeypatch.setattr(numa, "_allowed_memory_nodes", lambda: {0, 1})
    local_cores = list(range(48, 56))
    monkeypatch.setattr(numa, "_local_cpus", lambda node, allowed, root: local_cores.copy())
    bindings = {}
    monkeypatch.setattr(numa.os, "sched_getaffinity", lambda tid: bindings.get(tid, set(range(96))))
    monkeypatch.setattr(numa.os, "sched_setaffinity", lambda tid, cpus: bindings.__setitem__(tid, set(cpus)))
    memory = Mock(return_value={"mode": "MPOL_BIND", "mode_id": 2, "nodes": [1]})
    monkeypatch.setattr(numa, "_bind_memory", memory)
    return SimpleNamespace(cores=local_cores, bindings=bindings, memory=memory, sysfs=sysfs)


def test_two_ranks_choose_disjoint_local_cores_and_write_verified_receipts(rank_environment, monkeypatch, tmp_path):
    first = numa.bind_to_local_gpu(receipt_dir=tmp_path / "receipts")
    rank_environment.bindings.clear()
    monkeypatch.setenv("LOCAL_RANK", "1")
    second = numa.bind_to_local_gpu(receipt_dir=tmp_path / "receipts")
    assert first["cpu_affinity"] == [48, 49, 50, 51]
    assert second["cpu_affinity"] == [52, 53, 54, 55]
    assert not set(first["cpu_affinity"]) & set(second["cpu_affinity"])
    assert second["gpu_pci_bus_id"] == "0000:31:00.0"
    assert second["gpu_numa_node"] == 1
    assert second["cuda_visible_devices"] == "3,1"
    assert rank_environment.bindings == {100: {52, 53, 54, 55}, 101: {52, 53, 54, 55}}
    rank_environment.memory.assert_called_with(1, {0, 1})
    assert json.loads((tmp_path / "receipts/rank_1.json").read_text()) == second
    assert numa.os.environ["OMP_NUM_THREADS"] == numa.os.environ["MKL_NUM_THREADS"] == "4"


def test_insufficient_local_cores_fails_before_any_binding(rank_environment):
    rank_environment.cores[:] = [48, 49, 50, 51]
    with pytest.raises(RuntimeError, match="needs 8 allowed physical cores"):
        numa.bind_to_local_gpu()
    assert rank_environment.bindings == {}
    rank_environment.memory.assert_not_called()


def test_affinity_readback_mismatch_fails_before_memory_binding(rank_environment, monkeypatch):
    monkeypatch.setattr(numa.os, "sched_setaffinity", lambda tid, cpus: None)
    with pytest.raises(RuntimeError, match="CPU affinity verification failed"):
        numa.bind_to_local_gpu()
    rank_environment.memory.assert_not_called()


@pytest.mark.parametrize("node", [-1, 2])
def test_unknown_or_disallowed_gpu_node_fails_before_binding(rank_environment, node):
    (rank_environment.sysfs / "bus/pci/devices/0000:a1:00.0/numa_node").write_text(str(node))
    with pytest.raises(RuntimeError, match="unavailable in allowed memory nodes"):
        numa.bind_to_local_gpu()
    assert rank_environment.bindings == {}


def test_allowed_memory_nodes_reads_process_cgroup_effective_mask(monkeypatch, tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nMems_allowed_list:\t0-1,4\n")
    monkeypatch.setattr(numa, "Path", lambda value: status)
    assert numa._allowed_memory_nodes() == {0, 1, 4}
    status.write_text("Name:\tpython\n")
    with pytest.raises(RuntimeError, match="Cannot read"):
        numa._allowed_memory_nodes()


def _memory_library(monkeypatch, *, mode=2, nodes=None, set_error=0):
    applied_nodes = []

    def set_policy(policy, mask, maxnode):
        if set_error:
            ctypes.set_errno(set_error)
            return -1
        assert policy == 2
        bits = ctypes.sizeof(ctypes.c_ulong) * 8
        # Reproduce get_nodes' --maxnode and removal of the last excluded bit.
        applied_nodes[:] = [node for node in range(maxnode - 1) if mask[node // bits] & (1 << (node % bits))]
        if not applied_nodes:
            ctypes.set_errno(errno.EINVAL)
            return -1
        return 0

    def get_policy(policy, mask, maxnode, address, flags):
        ctypes.cast(policy, ctypes.POINTER(ctypes.c_int))[0] = mode
        bits = ctypes.sizeof(ctypes.c_ulong) * 8
        for node in applied_nodes if nodes is None else nodes:
            mask[node // bits] |= 1 << (node % bits)
        return 0

    library = SimpleNamespace(
        numa_num_possible_nodes=Mock(return_value=128),
        set_mempolicy=Mock(side_effect=set_policy), get_mempolicy=Mock(side_effect=get_policy),
    )
    monkeypatch.setattr(numa.ctypes, "CDLL", Mock(return_value=library))
    return library


@pytest.mark.parametrize("node", [0, 1, 63, 65, 127])
def test_memory_policy_requires_exact_readback_and_handles_large_node_ids(monkeypatch, node):
    library = _memory_library(monkeypatch)
    assert numa._bind_memory(node, {0, node}) == {"mode": "MPOL_BIND", "mode_id": 2, "nodes": [node]}
    mask = library.set_mempolicy.call_args.args[1]
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    assert mask[node // bits] == 1 << (node % bits)
    assert len(mask) * bits >= 128
    assert library.set_mempolicy.call_args.args[2] == 129
    assert library.get_mempolicy.call_args.args[2] == 129


def test_memory_policy_permission_failure_is_explicit(monkeypatch):
    library = _memory_library(monkeypatch, set_error=errno.EPERM)
    with pytest.raises(PermissionError, match="set_mempolicy"):
        numa._bind_memory(1, {0, 1})
    library.get_mempolicy.assert_not_called()


@pytest.mark.parametrize("mode,nodes", [(0, (1,)), (2, (0, 1))])
def test_memory_policy_inexact_readback_fails(monkeypatch, mode, nodes):
    _memory_library(monkeypatch, mode=mode, nodes=nodes)
    with pytest.raises(RuntimeError, match="verification failed"):
        numa._bind_memory(1, {0, 1})
