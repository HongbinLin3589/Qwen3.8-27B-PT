# Qwen3.8-27B 数学推理 SFT：10,000 条

这批数据是从固定版本的 [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/tree/e4e141ec9dea9f8326f4d347be56105859b2bd68) `default` 子集筛选、去重、去污染并转换的数学推理示范。共 **9,500 条训练样本 + 500 条内部验证样本**，每题只保留一条完整解答。原始解答由 DeepSeek-R1 生成，题目上游为 [NuminaMath 1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)。没有使用本项目的 mock 模型生成训练回答。

每行包含结构化 `messages` 和 `enable_thinking: true`，以及来源、原始行号、题目 UUID、所选 generation、标准答案、实际 token 数、验证记录等元数据。只有 `messages` 和思考开关进入聊天模板；标准答案等审计元数据不会拼进用户输入。完整的 `<think>…</think>` 与最终回答通过仓库现有模板转换并保留监督。

数据分为 **10 个 Parquet 文件，每个 1,000 条**（`data-00000-of-00010.parquet` 至 `data-00009-of-00010.parquet`）。每个文件中 `split` 字段明确标记 950 条 `train` 和 50 条 `validation`。`prepare.py` 按该字段拆分，恢复原有 9,500 / 500 的划分、顺序与样本内容；训练时请使用下面的入口，避免把验证样本一起训练。分片方式变化已通过完整表格逐行相等检查。

## 开始训练

在已安装本仓库 CUDA 12.9 / verl 依赖、已配置 GPU 的训练机上，从仓库根目录运行：

```bash
git pull --ff-only origin main
bash data/curated/math-sft-10k-v1/train.sh
```

默认模型目录为 `models/Qwen3.8-27B`，其中需要有完整模型权重和 tokenizer。自定义模型目录：

```bash
QWENPT_MODEL_PATH=/absolute/path/to/Qwen3.8-27B \
  bash data/curated/math-sft-10k-v1/train.sh
```

在原仓库使用的 Slurm 集群上提交：

```bash
sbatch data/curated/math-sft-10k-v1/train.sh
```

脚本沿用原仓库 `acd_u` 分区与单 GPU 设置，申请时长 12 小时；可用 `sbatch --partition=... --time=...` 按实际集群覆盖。GPU 显存需求取决于训练环境、模型实现及序列长度，本地没有运行 CUDA 前后向训练。

启动脚本先核对每个分片的 SHA-256 和行数，在 `prepared/` 合并出 `train.parquet`、`val.parquet`，再用**训练机模型目录中的 tokenizer** 逐条加载全部 10,000 条数据。任何超长或没有 assistant 监督的样本都会报错并停止。已存在的不同数据不会被静默覆盖。通过后复用 `scripts/train/sft.sh`，覆盖为：

- 完整序列最大 8,192 token，`truncation=error`，不截断答案。
- LoRA rank 8、alpha 16、学习率 1e-5，沿用原入口。
- 全局 batch 4、单 GPU micro batch 1、训练 1 epoch；移除原来只跑 1 step 的限制。
- 每 250 step 保存及验证，最多保留 3 个 checkpoint。
- 输出到 `checkpoints/sft-math-10k-v1`，可通过 `QWENPT_OUTPUT_DIR` 覆盖。

可以在脚本末尾继续传入 verl 参数，例如 `trainer.total_training_steps=5` 做 GPU 冒烟测试。未经训练和评测，这些设置不代表已证明能够提升基座能力。

只制作训练输入、不启动训练：

```bash
source .venv/bin/activate
python data/curated/math-sft-10k-v1/prepare.py \
  --tokenizer models/Qwen3.8-27B
```

## 筛选与检查

`validation.json` 保存实际数量、丢弃原因、各类别分布、长度分布及每个产物的 SHA-256。`source_manifest.json` 保存原始数据、tokenizer 和评测集的固定版本 URL 及校验和。

1. 只接受上游 `correctness_math_verify=true` 且 `is_reasoning_complete=true` 的完整轨迹。排除证明题、缺失答案、明显依赖缺失图片的题目，以及含 ChatML 边界注入的文本。上游仅 LLM judge 判断通过的轨迹不进入本批。
2. 按规范化题目精确去重，再用仓库现有 MinHash 去重：5-gram、128 permutations、16 bands、阈值 0.8。不同最终答案不豁免近重复过滤。选择和随机划分种子固定为 20261019。
3. 对 MATH-500、AIME 2024/2025、AMC 2023、Minerva、OlympiadBench 对应的下载快照做精确匹配及 13-word-gram 去污染。具体样本数、版本和词组重合阈值见两个 JSON 文件。未运行 embedding 语义去污染；无法宣称覆盖尚未提供的完整 3.1 评测清单。
4. 使用 `math-verify==0.9.0` 重新解析并核对**最终回答**与标准答案，拒绝解析失败、答案不等价或检查异常的样本。最终答案正确不等于每一步推理均已证明正确。
5. 用官方 `Qwen/Qwen3.8-27B` tokenizer 的固定快照及仓库 `tokenize_sft_sample` 计算完整序列，长度只保留 1,024–8,192 token。长度包含模板添加的 xhigh system 提示及所有边界 token。
6. 按数学类别轮流选样，类别内优先上游 R1 多次生成中正确率较低的题。这个正确率是**教师的难度代理**，没有测量目标 Qwen 的 pass rate。若同题有多个合格解答，只取字符数较少的完整解答；如其长度或答案复核失败，整题舍弃。
7. 所有选中样本在划分前已一起去重；随后随机划分 9,500 / 500。检查 Parquet 写入读回保持原样，格式验证无错误，逐条执行实际 `QwenSFTDataset`，检查长度、assistant loss mask 与 position IDs。

这是一批**数学推理 SFT** 数据，不覆盖 SWE/终端环境、工具调用轨迹、多模态或能力保留回放。正式增加训练量前，应先在你的完整评测清单上做基座与 LoRA 对照，并通过真实目标模型采样测量题目难度。

## 复现

使用仓库原有环境，另外安装构建依赖：

```bash
python -m pip install -e './data-pipeline[dev]' math-verify==0.9.0
python data/curated/math-sft-10k-v1/fetch_sources.py --raw-dir /scratch/math10k-raw
python data/curated/math-sft-10k-v1/build.py \
  --raw-dir /scratch/math10k-raw --output /scratch/math10k-rebuilt --workers 4
```

构建输出必须是新目录。复现需要下载约 650 MB 的候选和 tokenizer/评测资源，不需要模型权重。构建、复核与格式检查的具体版本记录在报告中；没有 GPU 推理采样，也没有伪造目标模型的难度记录。

## 来源与许可

OpenR1-Math-220k 与 NuminaMath 1.5 的官方数据卡均标注 Apache-2.0；附有许可证原文 `LICENSE.txt` 和未经改写的上游数据卡 `SOURCE_CARD.md`。本目录的 Parquet 是经过筛选、添加来源元数据及改写消息包装的派生版本；题目与示范回答保留上游内容，新增的用户末尾格式指令由本次转换加入。

来源贡献者及引用请见 `SOURCE_CARD.md` 和上游项目。评测题目仅在本地用于去污染，不作为训练样本发布。
