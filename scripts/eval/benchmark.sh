#!/usr/bin/env bash
# Generic API/mock evaluation. Does not submit Slurm jobs or start a GPU server.
# Example: bash scripts/eval/benchmark.sh plan --config configs/eval/demo.yaml \
#   --split subset --output runs/eval/demo-subset
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
eval_python="${BENCHMARK_PYTHON:-python}"
exec "$eval_python" -m qwenpt.eval "$@"
