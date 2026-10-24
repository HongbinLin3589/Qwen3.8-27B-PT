#!/usr/bin/env bash
#SBATCH --job-name=bolt-rollout
#SBATCH --partition=acd_u
#SBATCH --output=slurm_log/bolt-rollout/output_%j.txt
#SBATCH --error=slurm_log/bolt-rollout/err_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]}")}"
project_root="$(cd -- "$project_root" && pwd)"
while [[ ! -f "$project_root/pyproject.toml" ]]; do
    if [[ "$project_root" == / ]]; then
        echo "Run this script from inside the Qwen3.8-27B-PT repository." >&2
        exit 1
    fi
    project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"

# This existing vLLM CUDA 12.9 runtime supports Qwen3.5 without changing SFT's .venv.
rollout_python="${BOLT_ROLLOUT_PYTHON:-/data/user/jqiu236/Projects/PricePTQ/.venv/bin/python}"
model_path="$project_root/models/Qwen3.8-27B"
if [[ ! -f "$model_path/config.json" ]]; then
    hub_cache="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
    model_path="$hub_cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
fi
source_data="$project_root/data/curated/math-rl"
run_dir="$project_root/data/processed/bolt/math-rl-n8-8k-v1"
shard_dir="$run_dir/shards"
num_shards=8
mode="${1:-submit}"
shift || true
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export CUDA_HOME=/usr/local/cuda-12.8
export CC=/data/apps/gcc/12.5/bin/gcc CXX=/data/apps/gcc/12.5/bin/g++
export LD_LIBRARY_PATH="/data/apps/gcc/12.5/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$(dirname -- "$rollout_python"):$CUDA_HOME/bin:/data/apps/gcc/12.5/bin:$PATH"

case "$mode" in
    submit)
        mkdir -p "$project_root/slurm_log/bolt-rollout" "$shard_dir"
        script="$project_root/scripts/train/bolt_rollout.sh"
        for ((shard_id=0; shard_id<num_shards; shard_id++)); do
            job_id="$(env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
                sbatch --parsable --job-name="bolt-gen-$shard_id" "$@" "$script" generate "$shard_id")"
            job_id="${job_id%%;*}"
            echo "shard=$shard_id job_id=$job_id"
        done
        ;;
    generate)
        shard_id="${1:?Pass a shard index}"
        shift
        # Concurrent nodes can reuse PIDs and collide in shared compile-cache temporary files.
        export VLLM_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/qwenpt-vllm-${SLURM_JOB_ID:-$$}"
        mkdir -p "$VLLM_CACHE_ROOT"
        printf 'VLLM_CACHE_ROOT=%s\n' "$VLLM_CACHE_ROOT"
        "$rollout_python" -m qwenpt.train.utils.bolt_rollout generate \
            --data "$source_data" --model "$model_path" --output "$shard_dir" \
            --finalize-output "$run_dir" \
            --shard-id "$shard_id" --num-shards "$num_shards" \
            --num-candidates 8 --max-new-tokens 8192 --max-model-len 12288 \
            --max-num-seqs 16 --max-num-batched-tokens 8192 --batch-prompts 4 \
            --gpu-memory-utilization 0.92 --temperature 1.0 --top-p 0.95 --top-k 20 \
            --seed 20261020 "$@"
        ;;
    finalize)
        "$rollout_python" -m qwenpt.train.utils.bolt_rollout finalize \
            --data "$source_data" --shards "$shard_dir"/shard_*.jsonl --output "$run_dir" \
            --num-candidates 8 --seed 20261020 --val-ratio 0.1 --reuse-existing "$@"
        "$project_root/.venv/bin/python" -m qwenpt.train.utils.bolt_verify \
            --output "$run_dir" --tokenizer "$model_path" --max-length 12288
        ;;
    finalize-family)
        "$rollout_python" -m qwenpt.train.utils.bolt_recovery \
            --data "$source_data" --output "$run_dir" --reuse-existing "$@"
        ;;
    *)
        echo "Usage: bash scripts/train/bolt_rollout.sh {submit|generate SHARD_ID|finalize|finalize-family --shards FILE...}" >&2
        exit 1
        ;;
esac
