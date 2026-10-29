#!/usr/bin/env bash

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

model_path="${QWENPT_SFT_MODEL_PATH:-$project_root/models/Qwen3.8-27B}"
if [[ -z "${QWENPT_SFT_MODEL_PATH:-}" && ! -f "$model_path/config.json" ]]; then
    model_path="$(CUDA_VISIBLE_DEVICES='' python -c \
        "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.8-27B', revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0', local_files_only=True))")"
fi
prepared_dir="$project_root/data/processed/sft/qwen27b-v2"
train_file="${QWENPT_SFT_TRAIN_FILE:-$prepared_dir/train/pool.parquet}"
val_file="${QWENPT_SFT_VAL_FILE:-$prepared_dir/validation/pool.parquet}"
output_dir="${QWENPT_SFT_OUTPUT_DIR:-$project_root/checkpoints/sft-v2-100k-1epoch}"
num_gpus=6
seed=20261028

if [[ ! -f "$model_path/config.json" ]]; then
    echo "Set QWENPT_SFT_MODEL_PATH to the local model directory." >&2
    exit 1
fi
if [[ ! -f "$train_file" || ! -f "$val_file" ]]; then
    echo "Set QWENPT_SFT_TRAIN_FILE and QWENPT_SFT_VAL_FILE to the prepared v2 split files." >&2
    exit 1
fi
# qwenpt GPUs 0–1 are reserved for another service.
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
# MKL takes precedence for Torch CPU operations; keep the per-rank budgets equal.
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# verl rejects simultaneous CUDA and ROCm masks.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
# Use qwenpt's prepared compiler environment rather than paths from the old Slurm host.
export FLA_TILELANG=1
unset FLA_CI_ENV FLA_DISABLE_BACKEND_DISPATCH

mkdir -p -- "$output_dir"
export VERL_FILE_LOGGER_PATH="$output_dir/metrics.jsonl"
export WANDB_DIR="${WANDB_DIR:-$output_dir}"
export WANDB_MODE="${WANDB_MODE:-online}"

overrides=(
    "model.path=$model_path"
    "model.use_remove_padding=true"
    "model.enable_gradient_checkpointing=true"
    "model.enable_activation_offload=false"
    "model.lora_rank=0"
    "model.use_fused_kernels=false"
    "model.use_liger=false"
    "+model.override_config.attn_implementation=flash_attention_2"
    "engine.strategy=fsdp2"
    "engine.fsdp_size=$num_gpus"
    # Keep the established FP32 parameter/Adam precision with BF16 computation.
    "engine.model_dtype=float32"
    "engine.dtype=bfloat16"
    "engine.offload_policy=false"
    "engine.param_offload=false"
    "engine.optimizer_offload=false"
    "engine.ulysses_sequence_parallel_size=1"
    "engine.use_torch_compile=false"
    "engine.use_no_sync_for_gradient_accumulation=false"
    "engine.seed=$seed"
    "+engine.mixed_precision={param_dtype:bf16,reduce_dtype:fp32,buffer_dtype:fp32}"
    "data.train_files=$train_file"
    "data.val_files=$val_file"
    "data.train_max_samples=-1"
    "data.val_max_samples=-1"
    "data.custom_cls.path=pkg://qwenpt.train.verl.dataset"
    "data.custom_cls.name=QwenSFTDataset"
    "data.enable_thinking_default=true"
    "+data.loss_scope=all_assistant"
    "+data.apply_chat_template_kwargs.preserve_thinking=true"
    "data.pad_mode=no_padding"
    "data.max_length=262144"
    "data.truncation=error"
    "data.train_batch_size=96"
    "data.micro_batch_size_per_gpu=1"
    "data.use_dynamic_bsz=true"
    # Preserve the full longest row; this budget still needs capacity validation for native/no-offload.
    "data.max_token_len_per_gpu=${QWENPT_SFT_MAX_TOKENS:-231346}"
    "data.num_workers=0"
    "+data.length_bucketing=true"
    "+data.length_source=tokenized"
    "+data.length_bucket_drop_last=false"
    "+data.length_bucket_pad_tail=true"
    "+data.full_validation=true"
    "optim.lr=1e-5"
    "optim.lr_scheduler_type=cosine"
    "optim.lr_warmup_steps_ratio=0.1"
    "optim.override_optimizer_config={foreach:false,fused:false}"
    "trainer.project_name=Qwen3.8-27B-PT"
    "trainer.experiment_name=sft-v2-100k-1epoch"
    "trainer.default_local_dir=$output_dir"
    "trainer.total_epochs=1"
    "trainer.total_training_steps=null"
    "trainer.seed=$seed"
    "+trainer.progress_tracking=true"
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

exec torchrun --standalone --nnodes=1 --nproc-per-node="$num_gpus" \
    --module qwenpt.train.sft "${overrides[@]}" "$@"
