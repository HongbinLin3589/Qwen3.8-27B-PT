#!/usr/bin/env bash
#SBATCH --job-name=qwenpt-sft
#SBATCH --partition=acd_u
#SBATCH --output=sft_%j.out
#SBATCH --error=sft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00

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

# Optional environment overrides let curated datasets reuse this entry point.
model_path="${QWENPT_MODEL_PATH:-$project_root/models/Qwen3.8-27B}"
train_file="${QWENPT_TRAIN_FILE:-$project_root/data/processed/sft/train.parquet}"
val_file="${QWENPT_VAL_FILE:-$project_root/data/processed/sft/val.parquet}"
output_dir="${QWENPT_OUTPUT_DIR:-$project_root/checkpoints/sft-smoke}"

if [[ ! -f "$model_path/config.json" ]]; then
    echo "Set model_path in scripts/train/sft.sh to a local model directory." >&2
    exit 1
fi
for data_file in "$train_file" "$val_file"; do
    if [[ ! -f "$data_file" ]]; then
        echo "SFT Parquet file not found: $data_file" >&2
        exit 1
    fi
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

overrides=(
    "model.path=$model_path"
    "model.use_remove_padding=false"
    "model.enable_gradient_checkpointing=true"
    "model.lora_rank=8"
    "model.lora_alpha=16"
    "model.use_fused_kernels=false"
    "+model.override_config.attn_implementation=sdpa"
    "engine.strategy=fsdp2"
    "engine.model_dtype=bfloat16"
    "engine.use_torch_compile=false"
    "data.train_files=$train_file"
    "data.val_files=$val_file"
    "data.custom_cls.path=pkg://qwenpt.train.verl.dataset"
    "data.custom_cls.name=QwenSFTDataset"
    "data.enable_thinking_default=true"
    "+data.loss_scope=all_assistant"
    "+data.apply_chat_template_kwargs.preserve_thinking=true"
    "data.pad_mode=no_padding"
    "data.max_length=2048"
    "data.truncation=error"
    "data.train_batch_size=2"
    "data.micro_batch_size_per_gpu=1"
    "data.use_dynamic_bsz=false"
    "data.num_workers=0"
    "optim.lr=1e-5"
    "trainer.project_name=qwenpt"
    "trainer.experiment_name=sft-smoke"
    "trainer.default_local_dir=$output_dir"
    "trainer.total_epochs=1"
    "trainer.total_training_steps=1"
    "trainer.save_freq=1"
    "trainer.test_freq=1"
    "trainer.max_ckpt_to_keep=1"
    "trainer.resume_mode=disable"
    "trainer.logger=[console]"
    "trainer.n_gpus_per_node=1"
    "hydra.run.dir=$output_dir/hydra"
    "hydra.job.chdir=false"
)

torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    --module qwenpt.train.sft "${overrides[@]}" "$@"
