# 旧 BOLT benchmark 归档

此目录保留原模型与四个 BOLT checkpoint 的 GPQA/AIME GPU 评测流程，供历史核查，后续可整体删除 `legacy/`。
当前评测开发入口仍是 `scripts/eval/benchmark.sh` 和 `python -m qwenpt.eval`，当前源码和测试不依赖本目录。
归档不随 `qwenpt` 包安装，以下命令均从仓库根目录运行。

## 文件与职责

| 位置 | 职责 |
|---|---|
| `run_benchmark.py` | 单卡 vLLM 生成、旧判分协议与题组级断点恢复 |
| `scoring.py` | 历史 thinking 分离、GPQA 选项与 AIME boxed 数学判分 |
| `summarize.py` | 五个模型 × 四个任务的覆盖核验、准确率、avg@32 和百分点差值汇总 |
| `monitor.py` | 原 20 项 Slurm 作业状态、日志与结果完整性监控 |
| `export_adapter.py` | FSDP2 LoRA checkpoint 导出与 HF 权重合并 |
| `scripts/bolt_benchmarks.sh` | 旧 Slurm GPU 作业入口 |
| `tests/` | 上述归档流程的专用测试，仅显式运行 |
| `docs/bolt_checkpoint_benchmarks_20261021.md` | 原实验协议与历史命令记录，旧路径不代表当前入口 |

旧的 `qwenpt.eval.run_benchmark`、`summarize`、`monitor` 和 `export_adapter` 模块已退出活动包，不提供兼容转发入口。
归档命名空间改为 `legacy.benchmark.*`，例如：

```bash
source ./.venv/bin/activate
PYTHONPATH=src python -m legacy.benchmark.run_benchmark --help
PYTHONPATH=src python -m legacy.benchmark.summarize --help
PYTHONPATH=src python -m legacy.benchmark.monitor --help
PYTHONPATH=src python -m legacy.benchmark.export_adapter --help
```

原 GPU 作业脚本仍消费历史 `run_manifest.json` 与冻结模型目录，未改为当前框架的 YAML 配置。
如确需重新冻结归档作业，`source_snapshot/` 中须同时包含仓库的 `legacy/` 和所需 `src/qwenpt/`，脚本通过这两个路径加载归档 runner 与共用数学验证器。
本次未修改历史实验输出、已有冻结源码、模型或 Slurm 作业。
迁移会改变源码校验值，不能用新归档代码冒充原冻结版本续跑旧结果目录。

## 共用工具保留在当前代码

- `src/qwenpt/eval/prepare_benchmarks.py` 及其测试仍供当前接口准备冻结 GPQA/AIME 题库。
- `src/qwenpt/eval/scoring.py` 仅保留当前评分器使用的 `last_boxed()`；归档保留原评分实现。
- `src/qwenpt/train/utils/bolt_math.py` 同时服务训练与当前数学评分，不属于待删除归档。

## 验证

项目 pytest 默认只收集 `tests/`，当前 benchmark CI 只执行 `tests/eval/`。
归档的 CPU 测试可在具备 PyYAML、pytest、数学判分依赖的 Linux 环境显式运行：

```bash
PYTHONPATH=src python -m pytest legacy/benchmark/tests --ignore=legacy/benchmark/tests/test_export_adapter.py -q
```

导出测试另需原有 Torch 与 safetensors 环境，完整归档测试命令为 `PYTHONPATH=src python -m pytest legacy/benchmark/tests -q`。
GPU 模型生成仍需原有 Slurm/vLLM 环境，CPU 测试不代表真实模型评测已运行。
