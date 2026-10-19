# 数学推理 SFT 数据：10,000 条

10 个 Parquet 文件：`data-00000-of-00010.parquet` 至 `data-00009-of-00010.parquet`，每个文件恰好 1,000 条。

每个文件中，`split` 字段标记 950 条 `train` 和 50 条 `validation`；总计 9,500 条训练样本和 500 条内部验证样本。重分片保留原始划分、样本顺序及全部字段。使用时需要按 `split` 分开读取，避免验证样本进入训练。仓库原有 SFT 加载器不会自动按此字段过滤，因此不能把全部分片原样同时作为训练和验证输入。

每行包含结构化 `messages`（user / assistant）与布尔字段 `enable_thinking`，另有来源、标准答案、样本 ID、实际 token 数及验证记录。只有消息和思考开关进入聊天模板，标准答案等审计字段不应拼入用户输入。

数据来源为固定版本的 [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/tree/e4e141ec9dea9f8326f4d347be56105859b2bd68)，解答由 DeepSeek-R1 生成，题目上游为 NuminaMath 1.5。数据卡标注 Apache-2.0；原始许可及来源说明见 `LICENSE.txt`、`SOURCE_CARD.md`。

筛选使用上游完整性与规则验证标记，并使用 math-verify 重新核对最终答案；进行了精确与 MinHash 去重，以及对所列评测快照的词面去污染。固定版本的官方 Qwen/Qwen3.8-27B tokenizer 与仓库实际 QwenSFTDataset 已全量加载通过。完整序列最大 8,192 token，未截断。逐文件行数、SHA-256、筛选统计和限制见 `validation.json`；固定来源版本与下载校验和见 `source_manifest.json`。

这是数学推理 SFT 数据。最终答案正确不保证每一步推理正确；未运行 GPU 前后向训练、目标模型难度采样或 embedding 语义去污染，不能据此承诺训练收益。每个文件均混合两种 split，使用前必须正确划分。
