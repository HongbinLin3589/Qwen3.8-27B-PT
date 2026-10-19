# Qwen3.8-27B Post Training

For full documentation, refer to [Feishu](https://bcnaodifzwip.feishu.cn/wiki/BVn0wYFgjiH1hnkLAUmccwgsnTg).

Do not commit or push personal AI agent files or configurations, such as `AGENTS.md`, `.agent/`, or similar files and directories, to this repository.

## Data preprocessing

The standalone [data-pipeline](data-pipeline/README.md) package downloads and normalizes datasets, filters tasks, and exports SFT/RL Parquet. Prepared SFT JSON, JSONL and Parquet can also be imported directly. See the [format contract](data-pipeline/docs/data_format.md) and [release notes](data-pipeline/CHANGELOG.md).

Install it into the repository's existing Python 3.12+ environment:

```bash
source .venv/bin/activate
python -m pip install -e './data-pipeline[dev]'
python -m datapipe list
python -m unittest discover -s data-pipeline/tests -v
```

If the environment was created with uv without pip, use `uv pip install --python .venv/bin/python -e './data-pipeline[dev]'` for the installation step.

Run from the `data-pipeline/` directory when using its relative configuration and example paths. For example, export already prepared SFT rows with local screenshots embedded in Parquet:

```bash
cd data-pipeline
python -m datapipe sft --input /path/to/samples.jsonl --out /path/to/export \
  --config configs/sft.yaml --set export.sft.media_root=/path/to/dataset
```

The default student/teacher/reviewer backends are mocks for local verification. Live model integration and RFT/APEX trajectory adapters remain separate work. The repository's verl SFT training entry point is `scripts/train/sft.sh`.
