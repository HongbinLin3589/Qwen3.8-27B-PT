#!/usr/bin/env bash
#SBATCH --job-name=qwenpt-bolt
#SBATCH --partition=acd_u
#SBATCH --output=slurm_log/bolt/output_%j.txt
#SBATCH --error=slurm_log/bolt/err_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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

model_path="$project_root/models/Qwen3.8-27B"
if [[ ! -f "$model_path/config.json" ]]; then
    hub_cache="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
    model_path="$hub_cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
fi
# Data must already be split by complete prompt groups, with fixed reward labels.
train_file="$project_root/data/processed/bolt/math-rl-n8-8k-v1/train.parquet"
val_file="$project_root/data/processed/bolt/math-rl-n8-8k-v1/val.parquet"
if [[ ! -f "$val_file" ]]; then
    val_file=null
fi
output_dir="$project_root/checkpoints/bolt/${SLURM_JOB_ID:-local}"
num_gpus=2

if [[ ! -f "$model_path/config.json" || ! -f "$train_file" ]]; then
    echo "Set model_path and prepared BOLT train_file in scripts/train/bolt.sh (see docs/train/bolt.md)." >&2
    exit 1
fi
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
# Match the verified CUDA 12 / Hopper FLA backward setup used by SFT.
export CUDA_HOME=/usr/local/cuda-12.8
export CC=/data/apps/gcc/12.5/bin/gcc
export CXX=/data/apps/gcc/12.5/bin/g++
export PATH="$CUDA_HOME/bin:/data/apps/gcc/12.5/bin:$PATH"
export FLA_TILELANG=1
unset FLA_CI_ENV FLA_DISABLE_BACKEND_DISPATCH
mkdir -p -- "$output_dir"
export VERL_FILE_LOGGER_PATH="$output_dir/metrics.jsonl"
export WANDB_DIR="${WANDB_DIR:-$output_dir}"
export WANDB_MODE="${WANDB_MODE:-online}"

overrides=(
    "model.path=$model_path"
    "model.use_remove_padding=false"
    "model.enable_gradient_checkpointing=true"
    "model.lora_rank=8"
    "model.lora_alpha=16"
    "model.use_fused_kernels=false"
    "model.fused_kernel_options.impl_backend=torch"
    "+model.override_config.attn_implementation=sdpa"
    "engine.strategy=fsdp2"
    "engine.fsdp_size=$num_gpus"
    "engine.model_dtype=bfloat16"
    "engine.use_torch_compile=false"
    'engine.seed=${trainer.seed}'
    "data.train_files=$train_file"
    "data.val_files=$val_file"
    "data.custom_cls.path=pkg://qwenpt.train.verl.bolt_dataset"
    "data.custom_cls.name=QwenBOLTDataset"
    "+data.bolt_beta=0.1"
    "data.enable_thinking_default=true"
    "+data.loss_scope=all_assistant"
    "+data.apply_chat_template_kwargs.preserve_thinking=true"
    "data.pad_mode=no_padding"
    "data.max_length=12288"
    "data.truncation=error"
    "data.train_batch_size=32"
    "data.micro_batch_size_per_gpu=1"
    "data.use_dynamic_bsz=false"
    "data.num_workers=0"
    "optim.lr=5e-6"
    "optim.lr_scheduler_type=cosine"
    "optim.lr_warmup_steps_ratio=0.1"
    "trainer.project_name=Qwen3.8-27B-PT"
    "trainer.experiment_name=bolt"
    "trainer.default_local_dir=$output_dir"
    "trainer.total_epochs=1"
    "trainer.seed=20261020"
    "trainer.save_freq=100"
    "trainer.test_freq=-1"
    "trainer.max_ckpt_to_keep=6"
    "+checkpoint.save_lora_only=true"
    "trainer.resume_mode=disable"
    "trainer.logger=[console,file,wandb]"
    "trainer.n_gpus_per_node=$num_gpus"
    "hydra.run.dir=$output_dir/hydra"
    "hydra.job.chdir=false"
)

python -m qwenpt.train.utils.wandb_proxy -- \
    torchrun --standalone --nnodes=1 --nproc-per-node="$num_gpus" \
    --module qwenpt.train.bolt "${overrides[@]}" "$@"
