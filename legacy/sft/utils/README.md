# SFT 实验工具

此目录保存从正式训练包移出的性能、数值和单次实验诊断工具。
正式训练入口不导入这些工具；需要复现实验时从仓库根目录调用 `python -m legacy.sft.utils.<module>`。
矩阵工具 `sft_sweep` 只启动 `legacy.sft.entrypoint`，不会把实验开关交给正式 `qwenpt.train.sft`。

- `sft_benchmark`、`sft_sweep`、`sft_runtime`：短测记录、独立进程矩阵及只读运行时诊断。
- `benchmark_sft_heads`、`benchmark_sft_fsdp_heads`、`verify_sft_offload`：输出头／嵌套 FSDP 数值和 GPU 验收。
- `benchmark_ce`、`sft_ab_host`、`sft_ab_report`、`sft_repro_report`：旧 Slurm A/B 和重复性实验链。
- `verify_sft_run`：历史两卡 LoRA 运行的日志、checkpoint 与恢复游标验收，不适用于当前六卡全参数 SFT。
- `monitor_sft`：固定两 rank、2250 步数学实验的历史监控，不是当前全量 SFT 的通用监控器。

数据完整性准备保留在 `qwenpt.train.sft.prepare_data`，默认 `--max-samples -1` 验收全部原始 train 行，只有显式指定正数时才做有界抽样。
该改名保持原有 manifest 字段和 Parquet 列，包括历史来源标签 `prepare_sft_benchmark_data.py`，标签对应的哈希仍取实际生产者文件。
NUMA 与文件工具位于 `qwenpt.train.common`，数据检查和正式进度审计位于 `qwenpt.train.sft`；旧工具路径保留薄兼容入口。
旧 Slurm W&B 代理位于本目录的 `wandb_proxy.py`，从仓库根目录通过 `python -m legacy.sft.utils.wandb_proxy -- <command...>` 显式运行，测试位于 `legacy/sft/tests/test_wandb_proxy.py`。
正式 SFT/BOLT 均直接使用 `exec torchrun`，正式包不保留旧 `qwenpt.train.utils.wandb_proxy` 路径的兼容 stub。

历史 `outputs/` 中冻结的 driver、源码清单和哈希证据保持原样。
其中使用旧 `qwenpt.train.utils` 实验路径的入口必须结合当时的完整源码快照运行，不能直接套到清理后的生产包。
这里不提供生产路径兼容 stub，也不重写历史证据来伪装新路径兼容。
