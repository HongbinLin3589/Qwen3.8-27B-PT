# 训练数据格式（verl）

本文说明 `datapipe export` 写出的文件格式，以及每一项可以怎么调。所有选项都在配置的 `export` 段，默认值见 `src/qwenpt/data/defaults.yaml`。文中关于 verl 读取方式的说明，都对照过 verl 源码（volcengine/verl，提交 `18cfb748`，2026-10-16）：`verl/utils/dataset/multiturn_sft_dataset.py`、`rl_dataset.py`、`vision_utils.py`。

检查任意一个文件（导出的 Parquet，或手写的 `.json` / `.jsonl`）：

```bash
datapipe validate export/r1/sft/trajectories.parquet
datapipe validate my_samples.jsonl --kind sft
```

已有 SFT 数据可直接导入，无需创建 Task 或运行模型：

```bash
datapipe sft --input samples.json samples.jsonl samples.parquet --out export/prepared \
  --config configs/data/sft.yaml --set export.sft.media_root=/absolute/path/to/dataset
```

JSON 可以是单个样本或数组。输入列名使用上文标准名称；输出列名可用 `export.sft.keys` 修改。`configs/data/sft.yaml` 对不合格输入直接报错；设置 `export.sft.on_invalid=drop` 可将不合格行及原因写到 `sft/invalid.jsonl`。同名输入文件会合并样本而不会相互覆盖。JSON/JSONL 中无法表示二进制图片，已有图片字节应通过 Parquet 导入。

## SFT 数据

verl 默认读取 Parquet，每行一条样本。`messages` 按时间顺序保存完整对话：

```json
{
  "messages": [
    {"role": "system", "content": "你是数学助手。"},
    {"role": "user", "content": "有 3 个苹果，再买 2 个，一共有多少个？"},
    {"role": "assistant", "content": "3 + 2 = 5。\n#### 5"}
  ]
}
```

- `system` 可选，`user` 是输入，`assistant` 是示范回答。多轮对话就继续往后追加消息。
- verl 默认对所有 assistant 回合计算损失，包含历史 assistant。最后一条消息必须是有正文或工具调用的 assistant；若只训练最后一步，需要在训练侧另外配置损失掩码。

可选字段：

| 字段 | 类型 / 示例 | 含义 |
| --- | --- | --- |
| `tools` | `list[dict]`，OpenAI function schema | 工具定义。调用写在 `assistant.tool_calls` 里，返回值写成 `role=tool` 的消息 |
| `images` | `[{"image": "/data/a.jpg"}]` 或 `[{"bytes": <图片字节>}]` | 按顺序对应 content 中的 `<image>`，数量必须一致 |
| `videos` | `[{"video": "/data/a.mp4"}]` | 按顺序对应 content 中的 `<video>`，数量必须一致 |
| `enable_thinking` | `true` / `false` | 传给对话模板的思考模式开关，是否生效取决于模板 |

工具调用的写法见 `examples/data/verl_format/computer_use_sft.json`（FinanceGym computer-use 样本，含 3 轮历史操作和 4 张截图）。要点如下：

- `tool_calls[].id` 必须和后面 tool 消息的 `tool_call_id` 一一对应。
- `function.arguments` 是 JSON **字符串**。导出时如果是字典，会自动转成字符串。
- 最后一条 assistant 发出的调用，不需要有对应的 tool 返回。

### 导出前的检查

不通过的行写进 `sft/invalid.jsonl` 并附原因；设 `on_invalid: error` 则直接报错。检查内容：

- 角色只能是 system / user / assistant / tool。system 只能放在第一条。content 必须是字符串。
- 最后一条消息是 assistant。
- 每个 `tool_call_id` 前面都有对应的调用，且只返回一次；进入下一个 user/assistant 回合前，上一批调用必须返回（最后一条 assistant 的调用除外）。调用 id 不能重复，arguments 必须是能解析成对象的 JSON 字符串。
- 被调用的工具必须在 `tools` 里声明，名称不重复。工具参数 schema 必须有效，参数按 JSON Schema 检查 required、类型、enum 等约束；只解析本地 `$ref`，不会联网获取外部 schema。
- `enable_thinking` 如提供，必须是真正的布尔值，字符串 `"false"`、数字 0/1 和 null 不接受。`computer_use` 的 coordinate 必须恰有两个 0–999 的有限数值，布尔值不接受。
- `<image>` / `<video>` 的个数与 `images` / `videos` 的条数一致。每条媒体只能有 `image`（或 `video`）和 `bytes` 两者之一，因为 verl 的 `process_image` 不允许两者同时出现。
- 媒体路径必须为非空字符串，内嵌字节必须为非空二进制数据。导出和 `validate` 默认检查本地路径存在；`validate --media-root DIR` 指定相对路径的基准，`--no-check-media` 仅做结构校验。URL 可达性、媒体是否可解码需在训练环境进一步验证。

### 可调范围（`export.sft`）

| 选项 | 取值 | 作用 |
| --- | --- | --- |
| `sources` | `trajectories`、`reference` | SFT 样本从哪来：第 9 步的教师轨迹，或数据集自带的解答。数学题如果只有答案、没有解答，不会导出 |
| `extra_files` | `.json` / `.jsonl` / `.parquet` 路径列表 | 外部做好的 SFT 样本（如 FinanceGym 转出来的），和其他样本走同样的格式化和检查 |
| `keys` | 例如 `{messages: conversations, images: imgs}` | 改列名。必须与 verl 的 `data.messages_key`、`image_key`、`video_key`、`tools_key`、`enable_thinking_key` 一致 |
| `enable_thinking` | `keep` / `omit` / `true` / `false` | `keep` 保留每条样本自己的值；`omit` 不写这一列；`true` 或 `false` 给所有行写同一个值 |
| `media` | `path` / `absolute_path` / `bytes` | `path` 原样写路径；`absolute_path` 写成绝对路径；`bytes` 把图片字节嵌进 Parquet，写成 `{"bytes": ...}`，适合训练机上没有原图的情况 |
| `media_root` | 目录 | 相对路径从这个目录开始找 |
| `check_media_exists` | `true`（默认）/ `false` | 导出时检查本地图片/视频文件是否存在。找不到的样本记为不合格；URL 不做网络探测 |
| `empty_media_columns` | `true` / `false` | 给多模态样本补缺省的 `images: []` / `videos: []`；输入已经显式提供的空媒体列表会保留。纯文本文件不携带媒体列 |
| `split_by_modality` | `true`（默认）/ `false` | 纯文本样本写到 `<集合>.parquet`，带图或视频的写到 `<集合>.multimodal.parquet`。原因见下一节 |
| `system_prompts`（在 `export` 下） | `{domain: 文本}` | 按领域加 system 消息。不配置就不加 |

## RL 数据

你给的格式说明里 RL 部分是空的，这里沿用 verl 的标准 RL 格式（`RLHFDataset`）：

```json
{"data_source": "polaris-53k",
 "prompt": [{"role": "user", "content": "... Let's think step by step and output the final answer within \\boxed{}."}],
 "ability": "math",
 "reward_model": {"style": "rule", "ground_truth": "9375"},
 "extra_info": {"index": 0, "split": "train", "task_id": "polaris-53k-0", "source": "polaris-53k",
                "verifier": "answer_match", "verifier_version": "answer_match@ad7f417f6635"}}
```

- `ground_truth`：数学题是答案字符串。代码题是一个 JSON 字符串，内容是测试用例和判题方式（`python_asserts` / `python_function` / `stdio`）。
- 打分：在 verl 里设置 `custom_reward_function.path=<项目>/src/qwenpt/data/export/verl_reward.py`、`custom_reward_function.name=compute_score`，它会用同一套验证器判分。
- 题目带图片时（Task 的 `meta.images`），行里会多一列 `images`，提示里的 `<image>` 个数要与之对应。

可调范围（`export.rl`）：

| 选项 | 作用 |
| --- | --- |
| `split` | 文件名 `rl/<split>.parquet`，同时写进 `extra_info.split` |
| `verifier_types` | 只导出这些判题方式的题，例如 `[answer_match]`。默认全部导出 |
| `keys` | 改列名，例如 `{prompt: raw_prompt}`。必须与 verl 的 `data.prompt_key` 等一致 |
| `prompt_suffix` | 按判题方式在题面后追加一段话，可以用 `{function}` 指代测试调用的函数名 |
| `reward_style` | 写进 `reward_model.style`，默认 `rule` |
| `ability_map` | 领域到 `ability` 的映射，默认直接用领域名 |
| `extra_info_fields` | `extra_info` 里保留哪些字段 |
| `data_source_map`（在 `export` 下） | 数据源到 `data_source` 的映射。verl 按 `data_source` 选奖励函数，也按它分组统计 |

## verl 读取 Parquet 时要注意的事

写盘时，列名取所有行的并集；首行缺少的 `tools`、`enable_thinking` 等字段不会丢失。所有行均用 PyArrow 读回并比较，发生实质内容变化则报错，临时文件不会替换目标文件。导出报告另外记录嵌套 struct 自动补出的 null 键。此检查覆盖存储格式；实际训练还需用所选 tokenizer/processor 做集成验证。以下几点对照过上述 verl 源码：

1. **只要文件里有 `images` 或 `videos` 列，verl 就要求加载多模态 processor**（`multiturn_sft_dataset.py` 的 `_build_messages`）。纯文本模型读带这两列的文件会直接报错，所以默认把纯文本样本和多模态样本分开写成两个文件。
2. **同一个文件里，某行缺媒体列会被读成 null**，verl 随后会在 `len(None)` 处报错。所以多模态文件里只要有一行带 `videos`，其他行就补成 `[]`。`datapipe validate` 也会检查这一点。
3. **Parquet 会补出值为 null 的键**。同一列里的字典会被统一成相同的键，缺的键填 null，而 verl 读回来时不会去掉它们。实测你给的 computer-use 样本读回来后，没有工具调用的消息会多出 `tool_calls: null`、`tool_call_id: null`，`tools` 本身不变。
   - 【推论】这对模板是否有影响，取决于模板怎么判断：写成 `if message.tool_calls` 不受影响；写成 `'tool_calls' in message` 会把 null 当成有调用。我没有逐一核对各家模板。
   - 如果同一个集合里有多个不同的工具 schema，现在会自动生成 `.tools-<hash>.parquet` 分文件，避免工具参数 schema 彼此补出 null 属性。报告列出每个文件；训练时须读取全部分文件。
4. **`json_columns` 存成 JSON 字符串的列，verl 不会解析**。SFT 的五个标准列及其重命名禁止使用此选项；仅用于自定义加载器解码的其他列。
5. **图片字节和路径只能二选一**：`{"bytes": ...}` 和 `{"image": ...}` 不能同时出现在同一条里（`vision_utils.process_image` 里有 assert）。
