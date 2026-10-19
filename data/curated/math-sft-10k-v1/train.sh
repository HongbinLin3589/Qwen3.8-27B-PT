#!/usr/bin/env bash
#SBATCH --job-name=qwenpt-math10k
#SBATCH --partition=acd_u
#SBATCH --output=math10k_%j.out
#SBATCH --error=math10k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]}")}"
project_root="$(cd -- "$project_root" && pwd)"
while [[ ! -f "$project_root/pyproject.toml" ]]; do
    [[ "$project_root" != / ]] || { echo 'Run or submit from inside the repository.' >&2; exit 1; }
    project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"
source .venv/bin/activate
export QWENPT_MODEL_PATH="${QWENPT_MODEL_PATH:-$project_root/models/Qwen3.8-27B}"
export QWENPT_OUTPUT_DIR="${QWENPT_OUTPUT_DIR:-$project_root/checkpoints/sft-math-10k-v1}"
prepared_dir="$project_root/data/curated/math-sft-10k-v1/prepared"
[[ -f "$QWENPT_MODEL_PATH/config.json" ]] || { echo 'Set QWENPT_MODEL_PATH to the local Qwen3.8-27B directory.' >&2; exit 1; }
python data/curated/math-sft-10k-v1/prepare.py --output-dir "$prepared_dir" --tokenizer "$QWENPT_MODEL_PATH"
export QWENPT_TRAIN_FILE="$prepared_dir/train.parquet"
export QWENPT_VAL_FILE="$prepared_dir/val.parquet"
bash scripts/train/sft.sh \
    data.max_length=8192 \
    data.train_batch_size=4 \
    trainer.total_training_steps=null \
    trainer.total_epochs=1 \
    trainer.experiment_name=sft-math-10k-v1 \
    trainer.save_freq=250 \
    trainer.test_freq=250 \
    trainer.max_ckpt_to_keep=3 \
    "$@"
