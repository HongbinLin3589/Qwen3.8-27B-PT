# SFT 实验归档

本目录保存从正式训练包移出的 SFT 性能、数值和内存实验，便于回查与复现。
它不安装进 `qwenpt`，正式入口及默认测试不依赖此目录。
当前正式配置见 `docs/train/sft_v2_native.md`。

| 路径 | 职责 |
| --- | --- |
| `entrypoint.py` | 原实验训练入口，包含受控短测、输出头切换、CPU 卸载及额外重算挂钩 |
| `verl/` | 分块 CE、融合输出头、CPU 加载修复、子层检查点与 MLP 分块 |
| `utils/` | 性能扫描、梯度对照、卸载验收、旧 Slurm 监控、W&B SSH 代理和报告 |
| `scripts/` | 旧两卡数学 SFT、A/B 和重复性实验启动器 |
| `tests/` | 归档模块的 CPU 回归与手动 GPU 验收工具测试 |

从仓库根目录显式运行归档命令。

```bash
source ./.venv/bin/activate
CUDA_VISIBLE_DEVICES='' python -m legacy.sft.entrypoint --cfg job
CUDA_VISIBLE_DEVICES='' python -m legacy.sft.utils.sft_sweep --help
CUDA_VISIBLE_DEVICES='' python -m pytest legacy/sft/tests
```

旧脚本保留原两卡数学实验参数，不用于当前 qwenpt 六卡正式训练。
旧 Slurm 启动脚本通过 `python -m legacy.sft.utils.wandb_proxy -- <command...>` 管理 W&B 专用 SSH 隧道与训练子进程。
该包装器及其测试均位于本归档目录，正式包不保留旧 `qwenpt.train.utils.wandb_proxy` 路径的兼容 stub。
`monitor_sft` 保留固定两 rank、2250 步等旧协议，不能作为当前全量训练监控器。
正式数据准备与进度审计继续使用 `qwenpt.train.utils.prepare_sft_data` 和 `qwenpt.train.utils.sft_progress`。
更多工具说明见 [utils/README.md](utils/README.md)。

`outputs/` 中历史源码、哈希、PASS/FAIL 和冻结 driver 不在本次迁移中改写。
依赖旧包路径的历史 driver 需要配套的完整源码快照，不提供生产包兼容 stub，也不把路径调整后的文件当作原始哈希证据。
