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
