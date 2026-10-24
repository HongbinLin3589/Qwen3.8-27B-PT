# Qwen3.8-27B Post Training

For full documentation, refer to [Feishu](https://bcnaodifzwip.feishu.cn/wiki/BVn0wYFgjiH1hnkLAUmccwgsnTg).

Do not commit or push personal AI agent files or configurations, such as `AGENTS.md`, `.agent/`, or similar files and directories, to this repository.

## Data preprocessing

The [data pipeline](docs/data/README.md) lives in `src/qwenpt/data` and is installed with the main `qwenpt` package. It downloads and normalizes datasets, filters tasks, and exports SFT/RL Parquet. Prepared SFT JSON, JSONL and Parquet can also be imported directly. See the [format contract](docs/data/data_format.md) and [release notes](docs/data/CHANGELOG.md).

Install from the repository root into the existing Python 3.12 training environment:

```bash
source .venv/bin/activate
python -m pip install -e '.[code]'
python -m qwenpt.data list
python -m unittest discover -s tests/data -v
```

If the environment was created with uv without pip, use `uv pip install --python .venv/bin/python -e '.[code]'` for installation. The `datapipe` console command remains available. Optional `hf`, `embed` and `code` extras enable dataset downloads, embedding models and code-reference dependencies.

Run commands from the repository root. Configurations, examples and tests live in `configs/data`, `examples/data` and `tests/data`. For example, export prepared SFT rows with local screenshots embedded in Parquet:

```bash
python -m qwenpt.data sft --input /path/to/samples.jsonl --out /path/to/export \
  --config configs/data/sft.yaml --set export.sft.media_root=/path/to/dataset
```

The default student/teacher/reviewer backends are mocks for local verification. Live model integration and RFT/APEX trajectory adapters remain separate work. The repository's verl SFT training entry point is `scripts/train/sft.sh`.

Offline BOLT (formerly DGO) training uses `scripts/train/bolt.sh` and the same verl engine.
It consumes complete SFT-format trajectories with `prompt_id` and scalar `reward` columns, normalizes Boltzmann weights over complete prompt groups, and averages weighted response-level losses.
Run `bash scripts/train/bolt_rollout.sh submit` to sample eight candidates per question from `data/curated/math-rl` using eight independent single-GPU `acd_u` jobs.
The loader appends APEX's instruction `Please reason step by step, and put your final answer within \boxed{}.` to the last user message, preserves it in exported trajectories, and hashes these actual conditioning messages for `prompt_id` while leaving the source pool unchanged.
After all shard completion receipts are ready, the last worker exports and validates the data on CPU within its existing allocation.
Generation uses up to 8,192 new tokens within a 12,288-token context, retains failed and truncated candidates, and grades final answers with APEX's binary math verifier while preserving reasoning separately.
Complete prompt groups are split with seed `20261020` into 90% training and 10% validation; a full run produces 21,600 training rows and 2,400 validation rows under `data/processed/bolt/math-rl-n8-8k-v1/`.
Whole-group JSONL checkpoints and configuration fingerprints support resuming generation, and `BOLT_ROLLOUT_PYTHON` can override the existing inference runtime without changing the training environment.
The final `native_input_check.json` records full token-level validation through the native BOLT dataset in this project's training `.venv` with Transformers 5.9.0, including complete groups, lengths, assistant masks, and reward weights.
See [BOLT training](docs/train/bolt.md) for the data contract, formula, and launch instructions.
