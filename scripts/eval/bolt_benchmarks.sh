#!/usr/bin/env bash
#SBATCH --job-name=bolt-eval
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_log/bolt-benchmarks/%x_%j.out
#SBATCH --error=slurm_log/bolt-benchmarks/%x_%j.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$project_root"
run_dir="${1:?Pass the frozen evaluation run directory}"
variant="${2:?Pass original or step_N}"
benchmark="${3:?Pass gpqa_diamond or aime2024/2025/2026}"
mode="${4:-full}"

mapfile -t settings < <("$project_root/.venv/bin/python" - "$run_dir" "$variant" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
config = json.loads((root / 'run_manifest.json').read_text())
item = config['variants'][sys.argv[2]]
print(item['model_path'])
print(item['fingerprint'])
print(root)
PY
)
[[ ${#settings[@]} == 3 ]]
model_path="${settings[0]}"
fingerprint="${settings[1]}"
run_dir="${settings[2]}"
if [[ "$variant" != original ]]; then
    [[ -f "$model_path/manifest.complete" ]]
fi

# Reuse the already tested CUDA runtime without changing the training environment.
eval_python=/data/user/jqiu236/Projects/PricePTQ/.venv/bin/python
export PYTHONPATH="$run_dir/source_snapshot/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export CUDA_HOME=/usr/local/cuda-12.8
export CC=/data/apps/gcc/12.5/bin/gcc CXX=/data/apps/gcc/12.5/bin/g++
export LD_LIBRARY_PATH="/data/apps/gcc/12.5/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$(dirname -- "$eval_python"):$CUDA_HOME/bin:/data/apps/gcc/12.5/bin:$PATH"
export VLLM_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/qwenpt-eval-${SLURM_JOB_ID:-$$}"
mkdir -p "$VLLM_CACHE_ROOT"

extra=()
case "$mode" in
    full) output="$run_dir/results/$variant/$benchmark" ;;
    smoke)
        output="$run_dir/smoke/$variant/$benchmark"
        extra=(--limit 1 --samples-override 2 --max-new-tokens 8192)
        ;;
    *) echo "Unknown evaluation mode: $mode" >&2; exit 2 ;;
esac

"$eval_python" -m qwenpt.eval.run_benchmark \
    --bank "$run_dir/bank/question_bank.jsonl" --model "$model_path" \
    --variant "$variant" --variant-fingerprint "$fingerprint" \
    --benchmark "$benchmark" --output "$output" \
    --max-new-tokens 32768 --max-model-len 36864 --max-num-seqs 8 \
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --seed 20261021 "${extra[@]}"
