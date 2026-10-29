# BOLT checkpoint 与原模型的统一基准评测

日期：2026-10-21。

## 范围与状态

比较原始 `Qwen/Qwen3.8-27B` 固定快照与 BOLT 作业 `744218` 的第 100、200、300、400 步 checkpoint。
本次 GPQA Diamond 使用每题一次生成，AIME 2024、2025、2026 各使用每题 32 次生成的平均准确率。
五个模型与四个任务分别使用独立单 H100 作业并行运行，共 20 个正式评测作业。
短测、完整评测及最终汇总必须分别记录，不将短测或部分题目的结果报告为完整基准准确率。

运行目录为 `outputs/eval/bolt-744218-ckpt100-400-20261021/`。
`run_manifest.json` 保存模型指纹、协议、冻结源码校验和与计划范围，实际作业编号另存 `submission.json`。
最终指标写入 `summary.json`、`summary.csv` 与 `summary.md`，未完成任务只显示覆盖进度。

两个真实单卡短测 `745770`、`745771` 均以 `COMPLETED / 0:0` 结束，分别验证原模型 GPQA 与合并后第 400 步模型 AIME2026 的生成、thinking 分离、判分、落盘和 engine 关闭。
完整评测的 20 项任务已提交；集群用户 GPU 配额限制部分任务排队。
最初两个原模型任务在 `ACD1-59` CUDA 初始化失败且没有生成结果，已保持配置重提，并对相关排队任务排除该节点。
所有尝试和当前有效作业号保存在 `submission.json`，失败初始化不作为任何样本的错误答案计入。
ACD 队列不接受无 GPU 的独立汇总作业，因此监控使用独立的轻量本地 CPU 进程，不额外申请 GPU。
用户明确授权后，监控入口为 `qwenpt.eval.monitor`，每 300 秒只读查询 20 项作业与对应产物，仅在异常或全部完成时通过本机 `codex queue` 唤起同一聊天。
正常运行、正常配额排队及单个任务正常完成均保持静默，同一异常去重，通知提交失败时保留事件并重试。
源代码冻结在独立的 `monitor_source/`，不修改已运行生成任务的 `source_snapshot/`、模型或采样配置。
`monitor/state.json` 保存监控 PID、检查时间和事件投递状态，`monitor/snapshot.json` 保存全部最新作业状态、覆盖、日志证据与异常。
旧的无通知 `watch_eval.py` 观察进程在新监控验收后停止，以避免两个进程重复汇总。
`final_validation.json` 在全部作业终止后保存题目覆盖与调度器退出状态的联合验收结果。
原始 Slurm 非零退出、日志致命错误、损坏或缺失结果、初始化超过 30 分钟无进展、生成超过 90 分钟无完整题组以及连续三次状态查询失败均属于监控异常。
生成停滞事件按已完成题数区分，恢复进度后再次停滞可再次提醒；旧的失败重试记录不会触发当前任务异常。

最终汇总对每个模型报告四项准确率，并报告 $\Delta_{\mathrm{pp}}=100(\mathrm{accuracy}_{\mathrm{checkpoint}}-\mathrm{accuracy}_{\mathrm{original}})$。
该差值是百分点，原模型或对应 checkpoint 的任务未完整完成时不显示差值，准确率下降时保留负值。
汇总 JSON、CSV 与 Markdown 均使用原子替换写入，读取运行中的 JSONL 时只容忍最后一个尚未写完的记录。

## 冻结模型

基座快照为 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`，与当前 BOLT 的训练初始化一致。
`input_checkpoints/` 保存四个 checkpoint 的两 rank 模型分片、LoRA 元信息和 FSDP 配置，并逐文件核对 SHA256，防止训练后续保留策略清理早期权重。
`adapters/step_N/` 保存完整 PEFT adapter 与导出记录，包含视觉与文本模块，未静默丢弃目标参数。
`models/step_N/` 保存独立合并模型，按原基座分片逐项计算 $W'=\operatorname{BF16}(\operatorname{FP32}(W)+(\alpha/r)BA)$。
所有 1,199 个基座 tensor key 与 606 对 LoRA A/B 均检查覆盖、shape、有限值和输出回读，18 个权重分片均记录 SHA256。
`manifest.complete` 在所有分片和元数据写完后随目录原子发布，评测拒绝使用没有该凭据的合并模型。
基座、原生 checkpoint 和训练环境均保留原状，推理使用合并后的标准模型以避免运行时 LoRA 支持差异。

## 题库与指标

| 任务 | 来源 | 固定 revision | 题数 | 每题采样 |
| --- | --- | --- | ---: | ---: |
| GPQA Diamond | `Idavidrein/gpqa` | `633f5ee89ab8ad4522a9f850766b73f62147ffdd` | 198 | 1 |
| AIME 2024 | `Maxwell-Jia/AIME_2024` | `8d88b2876a82a080e2f172cc9b25d0d9d2cb4792` | 30 | 32 |
| AIME 2025 | `math-ai/aime25` | `563bb8404243c5f09de6ec262f2db674fe5bce9b` | 30 | 32 |
| AIME 2026 | `MathArena/aime_2026` | `d61f486805306e6e6b5913504c07763a99e1b112` | 30 | 32 |

原有 `math-ai/aime26` 缓存与 MathArena 版本的 30 个答案一致，但第 10、13 题有影响题意的转录差异，因此正式评测采用固定版本的完整 MathArena 题面。
该 Parquet 的 SHA256 为 `ae72853349a46c5f08666feb42db2d09fb4ac4fd4768c726b29702df21c7768e`。
四个任务合计 288 题、每个模型 3,078 次生成，五个模型共 15,390 次生成。

GPQA 使用 [simple-evals 的多选模板](https://github.com/openai/simple-evals/blob/main/common.py)，选项排列由种子 0 在题库准备时一次性冻结，五个模型共用同一排列与正确字母映射。
AIME 使用 [DeepSeek DeepInfer 的 boxed 提示词](https://github.com/deepseek-ai/DeepInfer/blob/main/eval/aime/bench.py)。
冻结题库的 SHA256 为 `eb446d40dff5b8eb67ec2842ebe105a416f5e76afdee365a4609ae04acb4ff19`。

对每年 AIME，报告 $\mathrm{avg@32}=\frac{1}{30\times32}\sum_{q=1}^{30}\sum_{s=1}^{32}\mathbf{1}\{\hat a_{q,s}=a_q\}$。
该指标不是多数票准确率，也不是至少一次答对的 pass@32。
判分只读取最后一个 `</think>` 后的内容，缺失思考结束标记时没有可判分的最终回答。
GPQA 提取最后一个符合 `Answer: A` 或 `Answer: $A$` 格式的字母，AIME 提取最后完整 boxed 表达式并用现有数学验证器检查等价性。
原始文本、预测、正确性、生成 token 数、结束原因、思考未结束和格式失败标记全部保留。

## 生成与运行环境

所有模型均启用 thinking、`preserve_thinking=true` 与显式 `reasoning_effort=xhigh`。
统一设置 temperature 1.0、top-p 0.95、top-k 20、输出最多 32,768 token、完整上下文最多 36,864 token、seed `20261021`。
GPQA 每批最多 8 题，AIME 每批 1 题并请求 32 个候选；每个题目的 seed 为固定基准 seed 加该任务内题目索引。
这些预算是本项目的统一比较协议，不将分数称为完整复现模型卡官方评测。

复用已经验证的 `/data/user/jqiu236/Projects/PricePTQ/.venv/bin/python`，不安装或修改训练 `.venv`。
该推理环境包含 vLLM `0.25.1+cu129`、Torch `2.11.0+cu129`、Transformers `5.14.1` 与数学验证器。
每作业 TP1、BF16、`language_model_only=true`、`max_num_seqs=8`、batch token 上限 8,192、显存比例 0.92。
avg@32 的 32 个候选通过调度排队完成，不要求同时驻留显存。
每个 Slurm 作业使用独立编译缓存，并在退出时显式关闭 vLLM engine。

## 入口与验收

```bash
source ./.venv/bin/activate
python -m qwenpt.eval.prepare_benchmarks \
  --output-dir outputs/eval/bolt-744218-ckpt100-400-20261021/bank \
  --aime2026-source outputs/eval/bolt-744218-ckpt100-400-20261021/sources/aime2026-d61f486.parquet

PYTHONPATH="$PWD/src" /data/user/jqiu236/Projects/PricePTQ/.venv/bin/python -m pytest tests/eval -q
python -m qwenpt.eval.summarize \
  --run-dir outputs/eval/bolt-744218-ckpt100-400-20261021 --require-complete
```

提交前创建 `slurm_log/bolt-benchmarks/`，调用 `sbatch` 时清除六个通用代理变量。
正式作业调用冻结的 `source_snapshot/scripts/eval/bolt_benchmarks.sh RUN_DIR VARIANT BENCHMARK full`，短测使用独立的 `smoke` 输出目录。
每份结果清单绑定题库、模型、源码、模板、数学判分器版本和采样协议，配置不一致时拒绝续跑。
每题必须保存唯一且完整的 sample index 集合，写完后 flush 与 fsync；重复题目、缺失候选和混合协议均不能通过最终覆盖验收。
最终验收需要四个任务的题数与采样数全部齐全，并核对 20 个 Slurm 作业的终态及原始产物。
