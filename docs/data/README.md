# datapipe：SFT + 离线 RL 数据流水线

这是第四章 4.2 数据处理流程（第 2–9 步）的实现，目标模型是 Qwen3.8-27B。核心代码位于 `src/qwenpt/data`，与训练代码共用根目录的 `pyproject.toml`。下文所有命令从仓库根目录运行。它把公开数据集或自建任务变成 verl 能直接读取的训练文件，整个流程分三条命令：

```
download（下载并统一格式） → run（去重、去污染、校验、过滤） → export（写成 verl Parquet）
```

每一步都是可替换的模块：加数据集只需改 YAML；加验证器、处理步骤或模型后端，只需写一个函数并注册。

- 导出的训练数据格式及各项可调范围：[data_format.md](data_format.md)
- 每一步的依据：[依据核对与最小方案.md](依据核对与最小方案.md)
- 实测结果：[公开数据演示结果.md](公开数据演示结果.md)（GSM8K / MBPP），[4.1数据实跑结果.md](4.1数据实跑结果.md)（POLARIS + DeepCoder）
- 版本变化：[CHANGELOG.md](CHANGELOG.md)

## 安装

```bash
python -m pip install -e .           # 统一安装 qwenpt，包含 datapipe 命令；使用仓库的 Python 3.12 训练环境
python -m pip install -e ".[hf,code]"          # 可选：hf = 从 Hugging Face 下载更快；code = LeetCode 参考解要用的 sortedcontainers
```

安装后也可以把 `datapipe` 换成 `python -m qwenpt.data`。Python 导入统一使用 `qwenpt.data`，例如 `from qwenpt.data.schema import Task`。纯源码调试时可设置 `PYTHONPATH=src`。

## 已有 SFT 样本直接导入

已有 `messages` 对话样本可以通过 `sft` 命令输入，支持单条/数组 JSON、逐行 JSONL，以及已有 Parquet。输入使用 `messages`、`tools`、`images`、`videos`、`enable_thinking` 标准列名；`run` 仍用于需要判题和过滤的内部 Task 数据。

```bash
datapipe sft --input samples.jsonl --out export/prepared --config configs/data/sft.yaml \
  --set export.sft.media_root=/absolute/path/to/dataset
datapipe validate export/prepared/sft/samples.multimodal.parquet --kind sft
```

`configs/data/sft.yaml` 遇到无效样本就报错，默认将图片嵌入 Parquet，保持 `enable_thinking` 的每条原值，并给多模态样本补空媒体列表。多文件可在 `--input` 后连续列出。`media_root` 指向 `images/` 的上一级目录；视频以路径保存，迁移训练机时需一并提供视频文件。

批次字段从所有行共同推导，首行没有的可选字段也会保留。所有行写盘后都会读回比较，发现内容丢失则报错。不同工具 schema 自动分成 `.tools-<hash>.parquet`；纯文本与多模态默认分开。训练文件列表应包含输出目录下的全部 `.parquet` 文件。

`validate` 默认检查本地媒体是否存在；只检查结构时使用 `--no-check-media`。相对媒体路径默认相对于当前工作目录，亦可用 `--media-root` 指定。URL 的远端可达性和图片解码、真实模型模板的行为需要另行验证。

## 快速开始

```bash
# 1. 下载：GSM8K、MBPP 作训练候选，GSM8K test、HumanEval、MBPP 测试集作评测集
datapipe download --sources configs/data/sources/hf.yaml --out data/pipeline/
#    连不上 HF 时，可以改用 GitHub 上的同一批文件：
datapipe download --sources configs/data/sources/github.yaml --out data/pipeline/

# 2. 处理：去重、对 data/pipeline/eval/ 里所有评测集去污染、执行校验……
datapipe run --data data/pipeline/ --config configs/data/github_demo.yaml --out runs/demo

# 3. 导出 verl Parquet（格式选项见 data_format.md）
datapipe export --run runs/demo --out export/demo --config configs/data/github_demo.yaml
```

`configs/data/github_demo.yaml` 关掉了需要模型采样的步骤，没有 GPU 也能跑完。

规模更大的例子是 4.1 表里的两项：POLARIS-53K 数学题和 DeepCoder 仓库里的代码题，评测集为 AIME24/25、AMC23、MATH-500、Minerva、OlympiadBench、HumanEval+ 和 LeetCode/Codeforces/CodeContests 测试题。全部数据通过 git 从 GitHub 获取：

```bash
datapipe download --sources configs/data/sources/math_code.yaml --out data/pipeline/math_code
datapipe run    --data data/pipeline/math_code --config configs/data/math_code.yaml --out runs/math_code
datapipe export --run runs/math_code --config configs/data/math_code.yaml --out export/math_code
```

实测结果见 `4.1数据实跑结果.md`。接上模型以后，改用 `configs/data/minimal.yaml` 或默认配置。

其他命令：
- `datapipe validate FILE`：检查一个 `.parquet` / `.jsonl` / `.json` 是否符合 SFT 或 RL 格式（手写的样本也可以）。
- `datapipe list`：列出所有已注册的组件。
- `datapipe show-config --config X`：查看合并后的完整配置。
- `python -m unittest discover -s tests/data`：运行测试。

## 目录

```
pyproject.toml            qwenpt 统一包信息、依赖、datapipe 命令
docs/data/
  CHANGELOG.md            版本变化
  data_format.md          训练数据格式与可调范围
configs/data/
  minimal.yaml            最小可行方案：只保留有依据的步骤
  github_demo.yaml        没有模型时的演示配置
  sources/hf.yaml         从 Hugging Face 下载的数据源清单
  sources/github.yaml     同一批数据从 GitHub 下载
  sources/math_code.yaml  4.1 表中的 POLARIS-53K 与 DeepCoder 代码题（git 下载）
  math_code.yaml          上面这批数据的处理配置
src/qwenpt/data/
  defaults.yaml           所有参数的默认值（原稿里的数字都在这里）；其他配置在它上面覆盖
  sources/                第 1 步，下载与统一格式
    loaders.py              数据从哪来：hf / url / file / git
    mappers.py              一行原始数据怎么变成任务：fields / mbpp / humaneval / rllm_code
  stages/                 第 2–8 步，每个文件一步，都用 @register_stage 注册
  verifiers.py            判对错：answer_match / python_asserts / python_function / stdio / command
  backends/               需要基础设施的部分：沙箱、嵌入、模型、审查 agent
  export/__init__.py      导出器注册（@register_exporter）、导出流程、validate
  export/verl.py          verl SFT / RL 行的构造、格式校验、图片路径/字节、按 verl 方式读回检查
  export/verl_reward.py   给 verl 用的奖励函数，复用上面的验证器
  schema.py               Task、Trajectory 等数据结构
  pipeline.py, cli.py     串起各步骤；命令行入口
examples/data/            小样例、verl 格式样例（含 computer-use 样本）
tests/data/               数据流水线与 SFT 格式回归测试
```

## 数据格式

### 下载后（`data/pipeline/`）

| 文件 | 内容 |
| --- | --- |
| `tasks.jsonl` | 所有训练候选，每行一个 Task（见下） |
| `train/<名字>.jsonl` | 按数据源分开的训练候选 |
| `eval/<名字>.jsonl` | 评测集，每行包括 `{id, prompt, repo}`，只用于去污染 |
| `manifest.json` | 每个数据源下载了多少行、写入多少行、出错信息 |

Task 的主要字段：

```json
{"id": "gsm8k-0", "source": "gsm8k", "domain": "math", "prompt": "题面",
 "created_at": "2026-05-01", "repo": null,
 "verifier": {"type": "answer_match", "answer": "72"},
 "reference": {"text": "完整解答 ... #### 72"}}
```

`verifier.type` 有五种：
- `answer_match`：比对最终答案，取 `\boxed{}`、`####` 之后的内容或最后一个数。
- `python_asserts`：运行 MBPP / HumanEval 那种 assert 测试。
- `python_function`：调用函数，比对输入和期望输出。
- `stdio`：把整个程序当作标准输入输出题来运行，逐个测试点比对输出（忽略空白，浮点误差 1e-6）。
- `command`：在容器里运行测试命令，看退出码。SWE 和终端任务用这种，此时还要填 `env`（镜像、仓库目录、测试命令）。

### 处理后（`runs/<名字>/`）

| 文件 | 内容 |
| --- | --- |
| `tasks.jsonl` | 通过所有步骤的任务；`history` 记录每一步的结果 |
| `dropped.jsonl` | 被删掉的任务，写明在哪一步、因为什么 |
| `eval_holdout.jsonl` | 截止日期之后的题、属于评测仓库的题 |
| `rollouts.jsonl` | 难度过滤时学生模型的全部采样（第 9 步字段） |
| `trajectories.jsonl` | 通过验证的教师轨迹（第 9 步字段） |
| `compose_pool.jsonl` | 全对且没被留作锚点的题 |
| `audit/<来源>.jsonl` | 人工抽检样本 |
| `report.md` | 每一步剩多少、删了多少、为什么 |

### 导出后（`export/<名字>/`，verl Parquet，每行一条样本）

| 文件 | 内容 |
| --- | --- |
| `rl/train.parquet` | RL 题：`prompt`、`data_source`、`ability`、`reward_model.ground_truth`、`extra_info` |
| `sft/<集合>.parquet` | 纯文本 SFT 样本：`messages`，以及可选的 `tools`、`enable_thinking` |
| `sft/<集合>.multimodal.parquet` | 带 `images` / `videos` 的 SFT 样本 |
| `sft/invalid.jsonl` | 没通过格式检查的样本和原因 |
| `export_report.json` | 各文件行数；按 verl 的方式读回后，和写入时不同的地方 |

集合名是 `trajectories`（教师轨迹）、`reference`（数据集自带的解答），或 `extra_files` 里的文件名。

SFT 样本的格式就是你给的那份说明（`messages` + 可选的 `tools` / `images` / `videos` / `enable_thinking`）。RL 沿用 verl 的标准格式。列名、思考开关、图片写路径还是嵌入字节、纯文本和多模态是否分文件、RL 的追加提示和 `extra_info` 字段，都可以在配置里调。完整说明和 verl 读取时要注意的事见 [data_format.md](data_format.md)。

## 怎么修改和扩展

### 加一个数据集：只改 YAML

在 `configs/data/sources/*.yaml` 里加一项。大多数数据集用 `fields` 映射器就够了：

```yaml
- name: math-hard
  role: train                      # train 或 eval
  license: MIT
  loader: {type: hf, repo: HuggingFaceH4/MATH-500, split: test, limit: 1000}
  where: [{field: level, in: [4, 5]}]        # 可选：按字段过滤
  mapper:
    type: fields
    id: unique_id                  # 字段名；也可以写模板 "{index}"
    prompt: problem                # 嵌套字段用点号，例如 prompt.0.content
    answer: answer
    reference_text: solution       # 完整解答，导出 SFT 时用
    domain: math                   # domain 和 verifier 是字面值
    verifier: answer_match
```

- 数据在 git 仓库里时（GitHub 等），用 `loader: {type: git, repo: owner/name, ref: 分支, path: 路径或通配符}`，只会拉取匹配的文件。
- verl 格式的源数据（`prompt` 是消息列表，答案在 `reward_model.ground_truth`）直接用 `fields`：`prompt: prompt.0.content`、`answer: reward_model.ground_truth`。
- 题面里有各题共用的套话时，用 `prompt_strip`（正则）去掉；只想在比对时忽略、不改题面，用 dedup / decontam_split 的 `ignore_patterns`。否则一句共用的 "Let's think step by step ... \\boxed{}" 就会让所有题都和评测集有 13-gram 重合。
- 用 HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- 读取私有数据集：`export HF_TOKEN=...`
- HF 下载会依次尝试三种方式：`datasets` 库、`huggingface_hub` 下载原始文件、datasets-server HTTP 接口，哪种可用就用哪种。

### 加一种数据格式：写一个 mapper

```python
# my_mappers.py，在运行前 import
from qwenpt.data.sources import register_mapper

@register_mapper("my_format")
def map_my_format(row, index, *, source, **opts):
    return {"id": f"{source}-{index}", "source": source, "domain": "math",
            "prompt": row["q"], "verifier": {"type": "answer_match", "answer": row["a"]}}
```

`@register_loader("name")` 的用法相同，用于新的数据来源，例如内部对象存储。

### 加一种判分方式：写一个验证器

```python
from qwenpt.data.verifiers import register_verifier
from qwenpt.data.schema import VerifyResult

@register_verifier("json_equal")
def json_equal(task, solution, sandbox=None, container=None):
    ok = ...  # 比对 solution.text 和 task.verifier.answer
    return VerifyResult(ok, float(ok), "json_equal@v1")
```

### 加一种导出格式：写一个 exporter

```python
from qwenpt.data.export import register_exporter

@register_exporter("openrlhf")
def export_openrlhf(run_dir, out_dir, cfg):   # cfg 是配置里的 export 段
    ...
    return {"train.jsonl": n}                  # 文件 -> 行数，命令行会打印出来
```

然后在配置里设 `export.format: openrlhf`。

### 加、删或调整处理步骤

- **关掉一步**：设 `<步骤名>.enabled: false`。
- **调整顺序**：改 `order` 列表。
- **新增一步**：

  ```python
  from qwenpt.data.stages import Stage, register_stage

  @register_stage("length_filter")
  class LengthFilter(Stage):
      def run(self, tasks):
          keep = []
          for t in tasks:
              if len(t.prompt) > self.cfg.get("max_chars", 4000):
                  self.ctx.drop(t, self.name, "too_long")
              else:
                  keep.append(t)
          return keep
  ```

  然后在配置的 `order` 里加入 `length_filter`（未被 import 的模块可以写 `my_pkg.stages:LengthFilter`），参数放在同名的配置段里。

### 接入真实基础设施：换后端

`configs/data/*.yaml` 的 `backends` 段决定每个后端用哪个实现。`type` 可以写注册名，也可以写 `包.模块:类名`，也可以在代码里用 `@register_backend("name")` 注册自己的类。

| 后端 | 现在可用的实现 | 状态 |
| --- | --- | --- |
| `sandbox` | `local_dir`（无隔离，仅开发用）、`docker` | 两者都实测过 |
| `embedder` | `hashing`（本地，较弱）、`sentence_transformers`、`openai_embeddings` | 后两个没实测过 |
| `student` / `teacher` | `mock_policy`、`openai_policy`（vLLM / SGLang，单轮） | `openai_policy` 没连真实服务测过；SWE 和终端题需要接入你们的多轮 harness |
| `student_scorer` / `teacher_scorer` | `mock_scorer`、`vllm_scorer` | `vllm_scorer` 没实测过 |
| `review_agent` | `mock_review` | 只有 mock |
| `shortcut_agent` | `grep_shortcut` | 简单的启发式规则，不是 LLM |

## 每一步做什么

| 步骤 | 配置段 | 做什么 |
| --- | --- | --- |
| 1 下载与统一格式 | `configs/data/sources/*.yaml` | 从 HF / URL / 本地读取，转成 Task |
| 2 去重 | `dedup` | 精确去重；MinHash 和嵌入近重复默认开，minimal 配置里关掉 |
| 2 去污染与切分 | `decontam_split` | 对所有评测集做 13-gram 加嵌入去污染；按截止日期、按仓库切出评测集 |
| 3 执行一致性 | `env_check` | SWE / 终端任务开 2+4 个全新容器；函数题和数学题检查参考答案能通过、空答案不能通过 |
| 4 验证器审查 | `verifier_review` | 模型多次解答，审查 agent 找假阳性和假阴性；导出人工抽检样本 |
| 5 泄漏检查 | `leak_check` | 清理编译产物和缓存，然后找残留的答案 |
| 6 难度过滤（插件，默认关） | `difficulty` | 学生模型采样后分档 |
| 8 教师轨迹 | `trajectory` | 只保留独立验证通过、长度合理的轨迹 |
| 9 存储 | `schema.Trajectory` | 来源、采样设置、奖励、验证器版本、两方 log-prob、工具调用等 |
| 导出 | `export` | verl SFT / RL Parquet |

## 已知局限

- 在开发环境里连不上 Hugging Face，所以 `configs/data/sources/hf.yaml` 中的数据集名和字段名是按记忆填写的，没有实际下载验证过；GitHub 版已实测。第一次运行后请检查 `data/pipeline/manifest.json`。
- 本地哈希嵌入在 MBPP 演示中漏掉了 31% 的改写题。正式使用请换成真实的嵌入模型。
- 纯文本相似度会把只差一个词的题判为重复（例如 odd 和 even、2×n 和 3×n）。答案保护只能覆盖函数题、stdio 题（同一输入的期望输出不同就不合并）和数学题（标准答案不同就不合并）。
- `stdio` 判题器做逐词比对，不支持需要特判（多解）的题目，这类题会被误判为错。
- Parquet 会给没有工具调用的消息补上 `tool_calls: null`，verl 读取时不会去掉。是否影响训练取决于对话模板的写法，详见 data_format.md。
