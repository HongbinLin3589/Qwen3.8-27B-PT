#!/usr/bin/env bash
#SBATCH --job-name=qwenpt-sft-ab
#SBATCH --partition=acd_u
#SBATCH --output=slurm_log/sft-performance-ab/output_%j.txt
#SBATCH --error=slurm_log/sft-performance-ab/err_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=04:00:00

set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source ./.venv/bin/activate
output="$PWD/checkpoints/sft-performance-ab/${SLURM_JOB_ID:?}"
mkdir -p "$output/source/src" "$output/source/legacy"
# Freeze all three arms together, including the custom dataset adapter.
cp -a src/qwenpt "$output/source/src/"
cp legacy/__init__.py "$output/source/legacy/"
cp -a legacy/sft "$output/source/legacy/"
python - "$output" <<'PY'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

directory = Path(sys.argv[1])
files = sorted((directory / "source").rglob("*.py")) + sorted((directory / "source").rglob("*.sh"))
manifest = {
    "sha256": {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    "versions": {name: importlib.metadata.version(name) for name in ("torch", "verl", "transformers", "fla-core", "tilelang", "triton")},
}
(directory / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
export PYTHONPATH="$output/source:$output/source/src${PYTHONPATH:+:$PYTHONPATH}"
# Do not let the submit directory shadow the frozen legacy package.
export PYTHONSAFEPATH=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
python -m legacy.sft.utils.sft_ab_host --output "$output/host_plan.json"
optimized_cpus="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["optimized_cpu_list"])' "$output/host_plan.json")"
for local_rank in 0 1; do
    LOCAL_RANK="$local_rank" LOCAL_WORLD_SIZE=2 taskset -c "$optimized_cpus" \
        python -c 'import sys; from qwenpt.train.common.numa import bind_to_local_gpu; bind_to_local_gpu(receipt_dir=sys.argv[1])' \
        "$output/binding_probe"
done
python -m legacy.sft.utils.benchmark_ce --output "$output/ce_gpu_validation.json" \
    > "$output/ce_gpu_validation.out" 2> "$output/ce_gpu_validation.err"

sampler_pid=""
cleanup_sampler() {
    if [[ -n "$sampler_pid" ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
        sampler_pid=""
    fi
}
trap cleanup_sampler EXIT
for arm in baseline numa numa_ce; do
    arm_dir="$output/$arm"
    mkdir -p "$arm_dir"
    cpu_key=optimized_cpu_list
    binding=true
    ce=native
    if [[ "$arm" == baseline ]]; then
        cpu_key=baseline_cpu_list
        binding=false
    elif [[ "$arm" == numa_ce ]]; then
        ce=chunked
    fi
    cpus="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$output/host_plan.json" "$cpu_key")"
    pci_ids="$(python -c 'import json,sys; print(",".join(g["pci_bus_id"] for g in json.load(open(sys.argv[1]))["gpus"]))' "$output/host_plan.json")"
    nvidia-smi --id="$pci_ids" \
        --query-gpu=timestamp,pci.bus_id,utilization.gpu,utilization.memory,power.draw,memory.used \
        --format=csv -l 2 > "$arm_dir/gpu_samples.csv" 2> "$arm_dir/gpu_samples.err" &
    sampler_pid=$!
    export QWENPT_SFT_OUTPUT_DIR="$arm_dir"
    taskset -c "$cpus" timeout --signal=TERM --kill-after=60s 40m \
        bash "$output/source/legacy/sft/scripts/sft.sh" \
        "+trainer.benchmark_steps=11" "+trainer.numa_binding=$binding" \
        "+trainer.ce_backend=$ce" "+trainer.ce_chunk_size=256" \
        "trainer.logger=[console,file]" "trainer.experiment_name=sft-ab-$arm" \
        > "$arm_dir/output.log" 2> "$arm_dir/error.log"
    cleanup_sampler
done
python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); (p/"all_arms_completed.json").write_text(json.dumps({"status":"completed","arms":["baseline","numa","numa_ce"],"steps_per_arm":11,"warmup_steps_discarded":3},indent=2)+"\n")' "$output"
