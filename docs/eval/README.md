# Benchmark 评测接口

本接口沿用仓库的 `src/qwenpt/eval`、`configs/eval`、`scripts/eval`、`tests/eval` 结构，支持固定训练子集 `subset` 和训后完整题库 `full`。生成与评分分开执行，可以接本地 vLLM/SGLang 的 OpenAI-compatible 服务，也可以填写远程 API 地址。被测模型与 LLM judge 使用独立的模型名、地址和密钥。

**当前可直接复用的数据是仓库已有的 GPQA Diamond、AIME 2024/2025/2026 冻结题库。** 新接口还提供统一 JSONL 输入，以及选择题、字符串匹配、数学判等、LLM judge 四类评分器。它不会自动下载所有 benchmark，也不包含浏览器、终端、手机、桌面或代码沙箱；这些任务必须接对应官方 harness 和评分器后才能报告成绩。

## 1. 两种 split 的含义

| 项目 | `--split subset` | `--split full` |
|---|---|---|
| 用途 | 训练过程中检查明显退化、比较 checkpoint | 训练完成后验证指定版本的全部题目 |
| 选题 | 固定 seed 分层抽样，或显式题目 ID 清单 | 配置声明的每个 benchmark 的全部题目 |
| 题量 | `subset.counts`，或 `subset.ids_file` | 必须与 `dataset.expected_counts` 完全匹配 |
| 记录 | 保存实际选中 ID、split、题库校验值 | 保存相同信息，并检查完整覆盖 |
| 运行目录 | 例如 `runs/eval/step-100-subset` | 例如 `runs/eval/final-full` |

`full` 是**完整的已声明题库**，不等于接口能够自行证明这是官方全集。准备数据的人必须核对官方版本、split、题目清单和题量；把缺题文件及错误题量一起填写进去，接口无法识别这种来源错误。程序拒绝计数不符、重复 ID 或选题缺失，CLI 不提供 `--limit` 来缩短全集。

题目覆盖与采样次数独立：`full` 可以每题一次；`subset` 也可以每题多次。采样次数优先级为 `benchmarks.<name>.samples` > 顶层 `samples` > 默认 `1`。题库行中的旧 `samples` 字段保留兼容，不控制新接口执行。

频繁查看的子集属于开发数据。将它从最终留出统计中单列，同时保留官方全集成绩；不要把多次据此调整过训练数据的小集称为完全未见的验证集。

## 2. 安装与完全离线的接口检查

API 客户端使用 Python 标准库；基本配置只需要 PyYAML。为了使用这个接口，不必在工作站安装整套 CUDA、verl 或训练依赖。以下命令均从仓库根目录运行，推荐 Python 3.12。

Linux / Bash：

```bash
python3.12 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install 'PyYAML>=6,<7' 'pytest>=9,<10'
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Windows / PowerShell：

```powershell
py -3.12 -m venv .venv-eval
.\.venv-eval\Scripts\Activate.ps1
python -m pip install 'PyYAML>=6,<7' 'pytest>=9,<10'
$env:PYTHONPATH = "$PWD\src"
```

`plan` 只读取本地配置和题库，冻结选题，不需要模型服务或 API key：

```bash
python -m qwenpt.eval plan --config configs/eval/demo.yaml --split subset --output runs/eval/demo-subset
python -m qwenpt.eval run --config configs/eval/demo.yaml --split subset --output runs/eval/demo-subset --backend mock
python -m qwenpt.eval score --config configs/eval/demo.yaml --split subset --output runs/eval/demo-subset
```

这个 demo 有四条**人工合成的接口样例**，子集选两条。`mock` 只返回每条题目的 `mock_response`，不会调用模型；它只能验证数据流和评分逻辑，不能用来证明模型能力。结果会标明 mock。要检查全集路径，换一个输出目录运行：

```bash
python -m qwenpt.eval run --config configs/eval/demo.yaml --split full --output runs/eval/demo-full --backend mock
python -m qwenpt.eval score --config configs/eval/demo.yaml --split full --output runs/eval/demo-full
```

Linux 上也可以用包装脚本：

```bash
BENCHMARK_PYTHON=.venv-eval/bin/python bash scripts/eval/benchmark.sh plan --config configs/eval/demo.yaml --split subset --output runs/eval/demo-subset
```

脚本只设置仓库路径并转发参数，不会提交 Slurm 作业或启动推理服务。

训练代码也可调用同一 Python 接口（保存并发布 checkpoint 后，务必同步 `model.revision` 与实际服务权重）：

```python
from qwenpt.eval.suite import load_config, plan, run, score

config = load_config("configs/eval/bolt_math_gpqa.yaml")
output = "runs/eval/step-100-subset"
plan(config, "subset", output)  # 仅冻结题目，离线操作
generation = run(config, "subset", output)  # 访问已配置模型服务
if not generation["complete"]:
    raise RuntimeError("评测生成尚未完整，检查错误日志后重试")
metrics = score(config, "subset", output)
if not metrics["complete"]:
    raise RuntimeError("评分尚未完整，不应发布正式成绩")
```

## 3. 填写被测模型和裁判 API

先复制空白模板，在本地编辑 `.env`：

```bash
cp configs/eval/.env.example .env
```

PowerShell 使用 `Copy-Item configs/eval/.env.example .env`。字段分别是：

| 环境变量 | 含义 |
|---|---|
| `BENCHMARK_BASE_URL` | 被测服务的 API 根地址，通常以 `/v1` 结尾；不要填 `/chat/completions` |
| `BENCHMARK_MODEL` | 服务公开的被测模型名称 |
| `BENCHMARK_API_KEY` | 被测服务密钥；仅保存在本地环境或 `.env` |
| `JUDGE_BASE_URL` | 裁判服务 API 根地址，可与被测服务不同 |
| `JUDGE_MODEL` | 裁判模型 ID，必须是该服务实际支持的 ID |
| `JUDGE_API_KEY` | 裁判服务密钥；规则判分不需要 |

例如，本地 vLLM 的 `BENCHMARK_BASE_URL` 可以是 `http://127.0.0.1:8000/v1`。如果本地服务明确无需认证，在配置中将对应的 `api_key_required` 改为 `false`，密钥留空即可。不会自动把空密钥替换为某家付费服务，也没有默认公网 endpoint。

`--env-file .env` 显式加载该文件；也可以由调度器注入进程环境变量。已有进程环境变量优先，文件不会覆盖它们。文件采用简单的 `KEY=value` 格式，可去除成对引号，不执行 shell 命令或展开其他变量。YAML 只保存密钥的环境变量名，不填写真实密钥。`.env`、真实 token、带认证信息的 URL 不应提交；输出中记录服务和模型信息，但不记录 API key。

在正式配置的 `model.revision` 填写**不可变的 checkpoint/export 哈希或模型快照版本**。服务名可能一直叫同一个别名，revision 必须随背后权重变化；仅修改模型服务器上的权重而不修改 revision，会破坏结果可追溯性。

使用 LLM judge 时还必须填写 `judge.revision`。优先使用实际模型快照版本；服务没有固定快照时，至少固定评测周期（如 `judge-eval-20261024-v1`），记录响应中的 `response_model`，并在同一周期内评原版与训练版。周期标签不能保证供应商不更新服务，仍需在报告中说明可复现性的边界。`FILL_`、`REPLACE`、`TODO` 等模板占位符会被预检拒绝，不能作为实际版本使用。

云 API 或开源裁判均通过同一接口配置。可将 `JUDGE_MODEL` 填为实际可访问的新模型（例如服务支持时填写 `gpt-5.6-sol`），也可填自部署开放权重模型的 served name；代码不假定模型名称一定可用。不要把 Codex 中显示的模型名自动等同于某个 API 账户可调用的模型名。

## 4. 复用目前的 GPQA / AIME 题库

已有的题库文件可以直接作为 `bank`。如果尚未准备，仓库原有导入器会读取已下载的 Hugging Face 缓存以及固定版本的 AIME 2026 Parquet：

```bash
python -m qwenpt.eval.prepare_benchmarks --output-dir data/processed/eval/bolt-bank --cache-root /path/to/huggingface-cache --aime2026-source /path/to/aime2026.parquet
```

该导入器另需 `pyarrow`；复用已有训练/数据环境即可，也可在独立环境安装 `pyarrow==23.0.1`。具体源 revision、缓存路径和 AIME 2026 SHA256 固定在 [prepare_benchmarks.py](../../src/qwenpt/eval/prepare_benchmarks.py)。它不会自动下载缺失文件，错误版本或缺失缓存需要先补齐。产出 `question_bank.jsonl`、`manifest.json`、`bank.sha256`，请保留来源 manifest。

编辑 [bolt_math_gpqa.yaml](../../configs/eval/bolt_math_gpqa.yaml) 的 `bank` 和 `model.revision`，填写 `.env`，然后：

```bash
# 训练小集：GPQA 24 + AIME 各年 10 = 54 道题，每题一次。
python -m qwenpt.eval plan --config configs/eval/bolt_math_gpqa.yaml --split subset --output runs/eval/step-100-subset
python -m qwenpt.eval run --config configs/eval/bolt_math_gpqa.yaml --split subset --output runs/eval/step-100-subset --env-file .env
python -m qwenpt.eval score --config configs/eval/bolt_math_gpqa.yaml --split subset --output runs/eval/step-100-subset

# 训后全集：GPQA 198 + AIME 各年 30 = 288 道题，每题一次。
python -m qwenpt.eval plan --config configs/eval/bolt_math_gpqa.yaml --split full --output runs/eval/final-full
python -m qwenpt.eval run --config configs/eval/bolt_math_gpqa.yaml --split full --output runs/eval/final-full --env-file .env
python -m qwenpt.eval score --config configs/eval/bolt_math_gpqa.yaml --split full --output runs/eval/final-full
```

`run` 默认后端为 `api`，只生成；即使配置含 LLM judge，也不会在这一步调用裁判。`score` 复用已生成答案并评分。规则评分不需要模型服务在线；数学评分需要项目已有的 `math_verify` 和 `latex2sympy2_extended`，优先复用已验证的评测运行环境。独立环境可安装 `math-verify` 和 `latex2sympy2_extended` 后先验证、固定版本；缺少依赖时不会偷偷换成字符串比较。

示例配置每题仅一次，与旧 BOLT 的 AIME `avg@32` 不同。如需每年 AIME 每题 32 次，在相应 benchmark 下显式填写：

```yaml
benchmarks:
  aime2024:
    samples: 32
    scorer:
      type: math_equivalence
  # aime2025、aime2026 同样配置 samples: 32。
```

同时保留原配置的 GPQA 和其他 AIME 定义。32 次输出的**平均正确率**不是 `pass@32`；新接口与旧结果对比前，还要对齐提示词、采样参数、模型版本及 scorer。

`generation` 原样作为服务支持的生成参数使用。例如 vLLM 通常使用 `max_tokens`；某些云模型需要 `max_completion_tokens`，或不允许自定义 `temperature`。请按服务实际契约修改，避免同时设置两种 token 上限。服务端需要准备足够的上下文和输出预算；本接口不替用户部署模型，也不会自动截短问题来绕过上下文限制。

## 5. YAML 配置与题库格式

可从三个配置开始：

- [demo.yaml](../../configs/eval/demo.yaml)：离线 mock + 规则判分。
- [demo_judge.yaml](../../configs/eval/demo_judge.yaml)：相同合成数据，演示独立 LLM judge。
- [bolt_math_gpqa.yaml](../../configs/eval/bolt_math_gpqa.yaml)：当前真实冻结题库的 API 评测配置。

主要字段：

| 字段 | 要求 |
|---|---|
| `version` | 当前为 `1` |
| `suite` | 评测方案名 |
| `bank` | UTF-8 JSONL；相对路径以 YAML 所在目录为基准 |
| `dataset.name / revision / split` | 数据来源、不可变版本、所用 split；避免使用会移动的 `main/latest` |
| `dataset.expected_counts` | 每个已配置 benchmark 的完整题数，必须与题库相符 |
| `model.revision` | 不可变 checkpoint/export 版本 |
| `model.*_env` | endpoint、模型名、密钥的环境变量名 |
| `generation` | Chat Completions 生成参数；不允许覆盖消息、认证字段或工具调用 |
| `samples` | 每题生成次数，默认 `1`；可被单个 benchmark 覆盖 |
| `subset.seed / counts / stratify_by` | 确定性的子集抽样参数；默认按题目 `stratum` 字段分层 |
| `subset.ids_file` | 可选，改为指定确切 ID；与 `counts` 方案二选一 |
| `benchmarks.<name>.scorer` | 评分方式与评分参数 |
| `judge` | 仅 LLM judge 使用的 endpoint、模型名、密钥环境变量及必填 `revision` |

题库每行是一个 JSON 对象，`benchmark + question_id` 在题库中唯一。最小选择题样例：

```json
{"benchmark":"example_choice","question_id":"q001","prompt":"Which number is even? A) 1 B) 2 C) 3 D) 5. End with Answer: LETTER.","answer":"B","stratum":"math/easy","metadata":{"source_revision":"immutable-revision"}}
```

- `benchmark` 必须与配置中的名字对应；`question_id` 是非空字符串。
- 使用 `prompt` 字符串，或使用 Chat Completions 形式的 `messages`。`messages` 可以传图像内容块，但具体 endpoint 必须支持该模态，媒体路径/地址也需要调用方准备。
- `answer` 为参考答案。规则与 correctness judge 需要它；开放式 rubric 任务的要求见下一节。
- `stratum` 用于分层，例如学科、难度、语言；不具备人工标签时不要编造难度。当前 BOLT 题库没有该字段，同一 benchmark 会作为一层抽样。
- `metadata` 保留来源与数据版本；可选 `rubric` 为该题评分标准。
- `samples` 兼容旧题库，但新接口以配置为准；`mock_response` 仅供合成离线样例使用，不会成为 API 生成的答案。

示例多模态输入结构：

```json
{"benchmark":"example_vision","question_id":"v001","messages":[{"role":"user","content":[{"type":"text","text":"Identify the object."},{"type":"image_url","image_url":{"url":"https://your-data-host.example/example.png"}}]}],"answer":"reference answer","stratum":"object"}
```

这是传输格式示例，不是已实现的视觉 benchmark 导入器。图像 URL 必须在运行前替换为可访问、版本固定的实际资源。

抽样按 seed 和题目身份稳定排序，再在各层分配数量；同一份题库、配置和 seed 会选中相同题目。更强的冻结方式是设置：

当前分配采用各层轮流取样，偏向均衡覆盖，而不是按原始题库比例抽样；因此开发子集均分不能直接当作全集分数估计。题目次序变化不会改变所选 ID，但题库文件校验值仍会变化并拒绝复用旧输出目录。

```yaml
subset:
  ids_file: ../../examples/eval/subset_ids.example.json
```

ID 文件格式为 `[{"benchmark":"demo_choice","question_id":"choice-001"}, ...]`，参考 [示例文件](../../examples/eval/subset_ids.example.json)。此时删去 `counts`，将选择的 ID 文件纳入版本管理，且不要在 checkpoint 之间重新抽题。

## 6. 评分器与 LLM judge

| `scorer.type` | 适用输入 | 是否需要裁判服务 |
|---|---|---|
| `multiple_choice` | 最终选项，例如 `Answer: B` | 否 |
| `exact_match` | 可直接匹配的短答案 | 否 |
| `math_equivalence` | 规范最终数学答案，建议 `\boxed{...}` | 否；需要数学验证依赖 |
| `llm_judge` | 参考答案判等或显式 rubric 评分 | 是 |

API 返回的最终 `content` 与 `reasoning_content` 分开保存；评分以最终回答为准。普通 API 最终回答不需要额外伪造 `</think>`。不要把长推理中的某个选项或数字当作最终答案。

`exact_match` 默认忽略大小写并规范化空白，可用 `case_sensitive: true`、`normalize_whitespace: false` 调整。`multiple_choice` 接受明确的末行 `Answer: B`、单独选项或 boxed 选项；可通过 `labels: [A, B, C, D]` 固定合法标签。不同官方 benchmark 的答案提取规则可能不同，应先完成协议对齐。

正确性 judge 示例：

```yaml
benchmarks:
  your_benchmark:
    scorer:
      type: llm_judge
      mode: correctness
      rubric: "依据标准答案判断最终回答是否正确，允许等价表达，拒绝事实矛盾。"
      structured_output: json_object
      judge_parameters:
        max_completion_tokens: 1024
judge:
  revision: judge-eval-20261024-v1  # 改为本次实际模型版本/固定评测周期
  base_url_env: JUDGE_BASE_URL
  model_env: JUDGE_MODEL
  api_key_env: JUDGE_API_KEY
  api_key_required: true
```

`mode: correctness` 是默认模式，使用参考答案并要求二值评分。`mode: rubric` 可按显式量表输出 `[0,1]` 的分数，适用于写作等任务，不能把其均分称为准确率。配置中的 `rubric` 或题目自带 `rubric` 必须非空。`structured_output` 可选 `none`（默认，以提示要求 JSON）、`json_object` 或 `json_schema`；只有服务支持时才选择后两者。评分失败、裁判格式错误和 API 错误会记录为错误，不悄然当作正确或正常完成。

测试 judge 前，先填写 `.env` 的裁判配置，并将 `demo_judge.yaml` 中的 `judge.revision` 占位符改为实际版本。最小流程如下；`run --backend mock` 仍是离线，但 **`score` 会调用你填写的裁判服务**：

```bash
python -m qwenpt.eval run --config configs/eval/demo_judge.yaml --split subset --output runs/eval/judge-demo --backend mock
python -m qwenpt.eval score --config configs/eval/demo_judge.yaml --split subset --output runs/eval/judge-demo --env-file .env --score-tag judge-v1
```

更换裁判模型、rubric 或判分参数后，可复用同一批 predictions，使用新的 `--score-tag judge-v2`。不要修改被测模型、题库、split 或生成参数来重评旧答案，这些属于另一场生成实验。原模型和训练后模型必须用同一个 judge、rubric 和预算评分。

通用 LLM judge 是本项目的评分协议，**不自动等价于某 benchmark 的官方评分实现**。IFBench 必须用其约束验证器，代码需要执行测试，GUI/办公任务需要验证环境与文件；不能为了接入方便，将它们一律改成“让 LLM 看答案打分”。开源裁判上线前应以人工标签校准，并与选定 API 裁判交叉核查。

## 7. 输出、断点恢复与完整性

每个模型 revision / split / 生成配置使用独立 `--output`：

```text
runs/eval/step-100-subset/
  selection.json                 # 选题清单和数据覆盖
  manifest.json                  # 生成设置与指纹
  predictions.jsonl              # 已持久化的模型输出
  generation_summary.json        # 生成完成度
  errors.jsonl                   # 生成错误（如有）
  scores/
    default/                     # 或 --score-tag 指定的名称
      manifest.json              # scorer / judge 协议指纹
      scores.jsonl               # 逐样本评分
      summary.json               # 各 benchmark 得分与覆盖
      errors.jsonl               # 评分错误（如有）
```

重新运行相同命令会检查指纹并跳过已完成样本，失败项可再次尝试；不需要额外 `--resume`。题库、选题、模型 revision、生成参数等变化时，旧输出目录会被拒绝，改用新目录。只改变评分协议时使用新 score tag，保留旧评分结果以便比较。

`plan` 成功只表示本地数据和选择方案通过检查；不表示 endpoint 可访问、模型具有对应能力或评测已完成。`run` 完成也不代表评分完成。报告要检查 summary 的 `complete`、`full_coverage`、`is_mock`、`expected_samples`、`completed_samples`、`missing_samples`，并查看错误日志。尚未完成时 `final_mean_score` 为 `null`；`mean_on_scored` 仅是已评分样本的暂时统计。错误日志保留历次失败，因此重试后日志中仍可能存在已解决的错误，以当前覆盖计数判断是否完成。

达到输出 token 上限的答案会标为 `length` 并计入 `length_limited_samples`，属于被测模型在固定预算内的实际输出，应保留而非挑掉。网络或裁判错误造成的缺失则使任务保持未完成。CLI 在未完成或失败时返回非零退出码 `2`；缺失项不能从分母悄悄剔除，更不能标记为“全集完成”。

同一个输出目录不应由多个进程并发写入。分开目录运行不同 checkpoint；该接口不负责 Slurm 调度、服务并发限流或分布式评测。中断后先确认原进程已结束，再用原命令恢复。

## 8. 与既有 BOLT 评测和项目清单的关系

旧 BOLT GPU runner、汇总、监控、模型导出工具与专用测试已迁移至 [legacy/benchmark](../../legacy/benchmark/README.md)，退出当前评测开发目录。
该归档记录五个模型版本 × 四个 benchmark 的历史 HPC 流程，GPQA 每题一次、AIME 每题 32 次；旧 `smoke` 不能直接当成冻结的训练子集。
当前接口不依赖 `legacy/`，后续删除归档不影响本接口。
题库准备工具 `qwenpt.eval.prepare_benchmarks` 与最终答案提取工具 `qwenpt.eval.scoring.last_boxed` 仍保留在当前模块，数学判等继续复用训练侧 `bolt_math`。
本次拆分修改了评分文件的源码指纹，但未改变当前评分算法；已有 predictions 可继续使用，重评时若旧 score tag 的指纹不匹配，请使用新的 `--score-tag` 保留原评分记录。

新接口的 API 输出和 score 目录独立，不写入旧 `results/<variant>/<benchmark>/results.jsonl`，也不交给旧 20 项作业监控汇总。原有模型导出和 checkpoint 加载由已有训练流程负责；新 API 接口消费已经启动的模型服务。

前期讨论的 **380 条全能力训练小集仍是目标方案，不是本次已经全部接入的清单**：

| 规划范围 | 子集配额 | 本仓库当前状态 |
|---|---:|---|
| 知识、科学、指令、数学、代码、中文、写作、拒答 | 256 | 有 GPQA / AIME 题库；其他任务仍需版本固定、导入器及官方评分对齐 |
| 长上下文、视觉、图表、文档、视频 | 84 | 接口可传 messages；媒体准备、官方 scorer 与长输入协议需逐项接入 |
| 终端、仓库代码、多工具、浏览器、手机、桌面、Excel | 40 | 需要各自环境和执行验证器，本次没有实现环境 harness |
| HLE 等低频正式验收 | 不纳入 380 | 需要题库导入、任务协议与 judge 校准 |

当前 `bolt_math_gpqa.yaml` 的 54 题是**现有代码能够接入的阶段性训练子集**，不等于已完成 380 题方案。正式加入新 benchmark 的步骤是：固定数据 revision/官方 split → 转成带来源信息的题库或接外部 harness → 实现对应官方 scorer → 冻结子集 IDs 与完整题量 → 用原版模型跑基线 → 再比较 checkpoint。当前 CLI 没有外部 harness 命令执行适配器；不能把 shell 命令填进 scorer 配置并期待自动运行。

## 9. 验证与排障

```bash
python -m pytest tests/eval/test_suite.py tests/eval/test_api_backend.py tests/eval/test_api_scoring.py -q
```

当前评测测试使用 `python -m pytest tests/eval -q`，包含 API 接口、评分与题库准备测试，不收集归档测试。
默认 `python -m pytest` 仅从 `tests/` 收集；归档测试需按 [legacy 说明](../../legacy/benchmark/README.md)显式运行。
通用 API/规则接口支持 Windows 与 Linux，但数学 scorer 复用已有的数学验证器，其超时/子进程逻辑需在原有验证过的 Linux runtime 使用。
本次 Windows 环境未通过该数学依赖的实测，不宣称数学评分已在 Windows 验证。
离线 demo 和本地 HTTP 测试验证接口工程，不包含真实付费 API 可用性、真实数据成绩或 GUI 环境验证。

| 情况 | 处理 |
|---|---|
| `No module named qwenpt` | 从仓库根目录运行并设置 `PYTHONPATH=src`，或在已有环境安装项目 |
| endpoint / model / key 缺失 | 填写 `.env` 并显式传 `--env-file .env`；本地无认证服务需配置 `api_key_required: false` |
| 服务拒绝参数 | 检查 `max_tokens` / `max_completion_tokens`、temperature 和服务支持的生成字段 |
| 数学验证依赖缺失 | 在已验证 runtime 中评分，或准备独立依赖并固定版本 |
| 题库数量或 ID 校验失败 | 补齐正确版本的数据；不能把 expected_counts 改小以掩盖丢题 |
| 恢复时指纹不一致 | 确认是否修改了权重、数据或生成配置；新实验使用新输出目录 |
| judge 改动后旧 score tag 冲突 | 使用新的 `--score-tag` 重评同一批 predictions |
| 少量题异常导致未完成 | 查看对应 errors 记录，修复服务/数据/评分配置后按兼容规则重试；不要当作正式全集分数 |
