# Shared setup for every script under scripts/. Source it, do not execute it:
#   source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
# Submit jobs from the repository root: sbatch copies the job script to a spool
# directory, so the job cannot locate the repository from its own path.
# Proxy variables are not handled here; per AGENTS.md they are dropped at
# submission time with `env -u ... sbatch`.

QWENPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QWENPT_ROOT

# The editable install of qwenpt lives in this venv, so `python -m qwenpt...`
# works on every node that sees the shared filesystem.
source "$QWENPT_ROOT/.venv/bin/activate"
