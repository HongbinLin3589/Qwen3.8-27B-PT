#!/usr/bin/env bash
#SBATCH --job-name=qwenpt-sft-math
#SBATCH --partition=acd_u
#SBATCH --output=sft_%j.out
#SBATCH --error=sft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=72:00:00

set -euo pipefail

# Slurm executes a spooled copy, so locate the repository from the submit directory.
project_root="${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]}")}"
project_root="$(cd -- "$project_root" && pwd)"
while [[ ! -f "$project_root/pyproject.toml" ]]; do
    if [[ "$project_root" == / ]]; then
        echo "Submit this script from inside the Qwen3.8-27B-PT repository." >&2
        exit 1
    fi
    project_root="$(dirname -- "$project_root")"
done
cd -- "$project_root"
source ./.venv/bin/activate

# Prefer the project model directory, with the verified local snapshot as fallback.
model_path="$project_root/models/Qwen3.8-27B"
if [[ ! -f "$model_path/config.json" ]]; then
    hub_cache="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
    model_path="$hub_cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
fi
source_data="$project_root/data/curated/math-sft-10k-v1"
prepared_dir="$project_root/data/processed/sft/math-sft-10k-v1-seed20261020"
train_file="$prepared_dir/train.parquet"
val_file="$prepared_dir/val.parquet"
output_dir="${QWENPT_SFT_OUTPUT_DIR:-$project_root/checkpoints/sft-math-10k-full-2epoch/${SLURM_JOB_ID:-local}}"
num_gpus=2

if [[ ! -f "$model_path/config.json" ]]; then
    echo "Set model_path in scripts/train/sft.sh to a local model directory." >&2
    exit 1
fi
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
# MKL takes precedence for Torch CPU operations; keep the per-rank budgets equal.
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Slurm exports ROCm masks even on H100 nodes; verl rejects simultaneous CUDA masks.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
# FLA routes the affected Hopper backward kernel through TileLang with CUDA 12 nvcc.
export CUDA_HOME=/usr/local/cuda-12.8
export CC=/data/apps/gcc/12.5/bin/gcc
export CXX=/data/apps/gcc/12.5/bin/g++
export PATH="$CUDA_HOME/bin:/data/apps/gcc/12.5/bin:$PATH"
export FLA_TILELANG=1
unset FLA_CI_ENV FLA_DISABLE_BACKEND_DISPATCH

python -m qwenpt.train.utils.prepare_data --data "$source_data" \
    --output "$prepared_dir" --seed 20261020 --val-ratio 0.1 --reuse-existing
mkdir -p -- "$output_dir"
export VERL_FILE_LOGGER_PATH="$output_dir/metrics.jsonl"
export WANDB_DIR="${WANDB_DIR:-$output_dir}"
export WANDB_MODE="${WANDB_MODE:-online}"

overrides=(
    "model.path=$model_path"
    "model.use_remove_padding=false"
    "model.enable_gradient_checkpointing=true"
    "model.lora_rank=0"
    "model.use_fused_kernels=false"
    "+model.override_config.attn_implementation=sdpa"
    "engine.strategy=fsdp2"
    "engine.fsdp_size=$num_gpus"
    # Keep FP32 parameters and Adam states on CPU; FSDP2 computes in BF16.
    "engine.model_dtype=float32"
    "engine.offload_policy=true"
    "engine.use_torch_compile=false"
    "data.train_files=$train_file"
    "data.val_files=$val_file"
    "data.custom_cls.path=pkg://qwenpt.train.verl.dataset"
    "data.custom_cls.name=QwenSFTDataset"
    "data.enable_thinking_default=true"
    "+data.loss_scope=all_assistant"
    "+data.apply_chat_template_kwargs.preserve_thinking=true"
    "data.pad_mode=no_padding"
    "data.max_length=8192"
    "data.truncation=error"
    "data.train_batch_size=8"
    "data.micro_batch_size_per_gpu=1"
    "data.use_dynamic_bsz=false"
    "data.num_workers=0"
    "optim.lr=1e-5"
    "optim.lr_scheduler_type=cosine"
    "optim.lr_warmup_steps_ratio=0.1"
    "trainer.project_name=Qwen3.8-27B-PT"
    "trainer.experiment_name=sft-math-10k-full-2epoch"
    "trainer.default_local_dir=$output_dir"
    "trainer.total_epochs=2"
    "trainer.total_training_steps=null"
    "trainer.seed=20261020"
    "+trainer.cpu_offload_pin_memory=false"
    "trainer.save_freq=after_each_epoch"
    "trainer.test_freq=-1"
    "trainer.max_ckpt_to_keep=2"
    "+checkpoint.save_lora_only=false"
    "trainer.resume_mode=disable"
    "trainer.logger=[console,file,wandb]"
    "trainer.n_gpus_per_node=$num_gpus"
    "hydra.run.dir=$output_dir/hydra"
    "hydra.job.chdir=false"
)

python -m qwenpt.train.utils.wandb_proxy -- \
    torchrun --standalone --nnodes=1 --nproc-per-node="$num_gpus" \
    --module qwenpt.train.sft "${overrides[@]}" "$@"
