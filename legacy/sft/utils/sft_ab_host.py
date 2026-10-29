"""Plan equal eight-core CPU budgets for the two-GPU SFT A/B allocation."""

import argparse
import json
import os
from pathlib import Path

from qwenpt.train.utils.numa import _allowed_memory_nodes, _local_cpus, _visible_gpu_pci_ids


def _node_meminfo(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        name, value = line.split(":", 1)
        key = name.split(maxsplit=2)[-1]
        parts = value.split()
        values[key] = {"value": int(parts[0]), "unit": parts[1] if len(parts) > 1 else "count"}
    return values


def plan_host_affinity() -> dict:
    """Return verified CPU masks and topology without changing process affinity.

    Both masks contain eight physical cores; two colocated GPUs use four remote
    baseline cores versus eight local optimized cores. Missing topology or CPU
    capacity raises instead of expanding either arm's effective CPU budget.
    """
    reserved = os.environ.get("SLURM_CPUS_PER_TASK")
    if reserved is not None and int(reserved) != 16:
        raise RuntimeError(f"SFT A/B expects 16 reserved CPUs, got SLURM_CPUS_PER_TASK={reserved}")
    sysfs = Path("/sys")
    allocation_cpus = os.sched_getaffinity(0)
    allowed_memory_nodes = _allowed_memory_nodes()
    pci_ids = _visible_gpu_pci_ids(2)
    gpu_nodes = [int((sysfs / f"bus/pci/devices/{pci_id}/numa_node").read_text()) for pci_id in pci_ids]
    if any(node < 0 or node not in allowed_memory_nodes for node in gpu_nodes):
        raise RuntimeError(f"GPU NUMA nodes {gpu_nodes} unavailable in allowed memory nodes {sorted(allowed_memory_nodes)}")
    node_paths = sorted((sysfs / "devices/system/node").glob("node[0-9]*"), key=lambda path: int(path.name[4:]))
    node_cores = {int(path.name[4:]): _local_cpus(int(path.name[4:]), allocation_cpus, sysfs) for path in node_paths}
    if sum(len(cores) for cores in node_cores.values()) < 16:
        raise RuntimeError(f"SFT A/B needs 16 reserved physical cores; allowed node cores are {node_cores}")
    if gpu_nodes[0] == gpu_nodes[1]:
        local = node_cores[gpu_nodes[0]]
        remote = sorted(cpu for node, cores in node_cores.items() if node != gpu_nodes[0] for cpu in cores)
        if len(local) < 8 or len(remote) < 4:
            raise RuntimeError(f"Colocated GPUs require 8 local and 4 remote physical cores; found local={local}, remote={remote}")
        optimized_cpus = local[:8]
        baseline_cpus = sorted(local[:4] + remote[:4])
        explanation = "Both GPUs share a NUMA node: baseline uses four local plus four remote cores; optimized uses eight local cores."
    else:
        if any(len(node_cores[node]) < 4 for node in gpu_nodes):
            raise RuntimeError(f"Separate GPU NUMA nodes require four physical cores each; allowed node cores are {node_cores}")
        optimized_cpus = sorted(cpu for node in gpu_nodes for cpu in node_cores[node][:4])
        baseline_cpus = optimized_cpus.copy()
        explanation = "GPUs are on separate NUMA nodes: both arms use the same four local cores per GPU, eight total; optimized additionally isolates rank affinity and binds rank memory."
    if len(set(optimized_cpus)) != 8 or len(set(baseline_cpus)) != 8:
        raise RuntimeError("A/B masks must each contain exactly eight distinct physical cores")
    return {
        "allocation_cpus": sorted(allocation_cpus),
        "reserved_cpus_per_task": int(reserved) if reserved is not None else None,
        "allocation_physical_cores": sum(len(cores) for cores in node_cores.values()),
        "allowed_memory_nodes": sorted(allowed_memory_nodes),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": [{"local_rank": rank, "pci_bus_id": pci_id, "numa_node": gpu_nodes[rank]} for rank, pci_id in enumerate(pci_ids)],
        "node_allowed_physical_cpus": {str(node): cores for node, cores in node_cores.items()},
        "node_meminfo": {path.name: _node_meminfo(path / "meminfo") for path in node_paths},
        "local8": optimized_cpus,
        "baseline8": baseline_cpus,
        "baseline_cpu_list": ",".join(map(str, baseline_cpus)),
        "optimized_cpu_list": ",".join(map(str, optimized_cpus)),
        "effective_physical_cores_per_arm": 8,
        "threads_per_rank": 4,
        "explanation": explanation,
        "comparison_limits": [
            "Both arms reserve 16 CPUs but use eight physical cores; resource cost remains a 16-CPU allocation.",
            "The optimized arm jointly changes rank CPU isolation and memory placement; this test does not identify their individual speedups.",
            "A reserved 16-CPU allocation restricted to eight active cores does not reproduce contention in an allocation reserving only eight CPUs.",
            "Per-node free memory is a planning snapshot; strict local memory binding still requires enough host memory throughout training.",
        ],
    }


def main() -> None:
    """Write the host allocation plan to the required ``--output`` JSON path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = plan_host_affinity()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "baseline_cpu_list": plan["baseline_cpu_list"], "optimized_cpu_list": plan["optimized_cpu_list"]}))


if __name__ == "__main__":
    main()
