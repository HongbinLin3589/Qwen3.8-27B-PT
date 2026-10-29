#!/usr/bin/env bash
#SBATCH --job-name=qwenpt-sft-repro
#SBATCH --partition=acd_u
#SBATCH --output=slurm_log/sft-performance-ab/repro_output_%j.txt
#SBATCH --error=slurm_log/sft-performance-ab/repro_err_%j.txt
#SBATCH --nodes=1
#SBATCH --nodelist=ACD1-7
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=02:00:00

set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source ./.venv/bin/activate
reference="${QWENPT_SFT_REFERENCE_DIR:-$PWD/checkpoints/sft-performance-ab/744322}"
if [[ ! -f "$reference/source/legacy/sft/entrypoint.py" || ! -f "$reference/source/legacy/sft/scripts/sft.sh" ]]; then
    echo "Reference predates the legacy namespace; use its original frozen scripts, or set QWENPT_SFT_REFERENCE_DIR to a legacy snapshot." >&2
    exit 2
fi
output="$reference/repro/${SLURM_JOB_ID:?}"
mkdir -p "$output/source"
cp -a "$reference/source/." "$output/source/"
# Only post-fit diagnostics differ; model, dataset, CE and trainer stay frozen.
for module in sft_benchmark sft_runtime sft_ab_report sft_repro_report; do
    cp "legacy/sft/utils/$module.py" "$output/source/legacy/sft/utils/"
done
cp legacy/sft/scripts/sft_repro.sh "$output/source/legacy/sft/scripts/"
python - "$output" "$reference" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

directory, reference = map(Path, sys.argv[1:])
files = sorted((directory / "source").rglob("*.py")) + sorted((directory / "source").rglob("*.sh"))
manifest = {
    "reference_run": str(reference),
    "diagnostic_overlay": ["sft_benchmark.py", "sft_runtime.py", "sft_ab_report.py", "sft_repro_report.py"],
    "sha256": {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
}
(directory / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
export PYTHONPATH="$output/source:$output/source/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline QWENPT_SFT_RUNTIME_DIAGNOSTICS=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
python -m legacy.sft.utils.sft_ab_host --output "$output/host_plan.json"
cpus="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["baseline_cpu_list"])' "$output/host_plan.json")"
for arm in baseline_repeat1 baseline_repeat2; do
    arm_dir="$output/$arm"
    mkdir -p "$arm_dir"
    export QWENPT_SFT_OUTPUT_DIR="$arm_dir"
    taskset -c "$cpus" timeout --signal=TERM --kill-after=60s 40m \
        bash "$output/source/legacy/sft/scripts/sft.sh" \
        "+trainer.benchmark_steps=6" "+trainer.numa_binding=false" \
        "+trainer.ce_backend=native" "+trainer.ce_chunk_size=256" \
        "trainer.logger=[console,file]" "trainer.experiment_name=sft-$arm" \
        > "$arm_dir/output.log" 2> "$arm_dir/error.log"
done
python -m legacy.sft.utils.sft_repro_report --run-dir "$output" --steps 6
