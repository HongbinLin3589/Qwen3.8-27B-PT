# 通用模型 Benchmark

从 [project.yaml](../../configs/eval/project.yaml) 运行项目评测：覆盖指令遵循、科学、代码、数学、终端、仓库编码，以及浏览器、表格、桌面办公和手机场景。入口为 `python -m qwenpt.eval project`，必须明确选择 `--profile` 和 `--split`。模型服务、checkpoint 版本和 API key 由本地 `.env` 提供。

范围依据是 **2026-10-29 核对的飞书[基准清单与评分方式](https://bcnaodifzwip.feishu.cn/wiki/PeYVw1Nboi6krBk0neYc6qGonJc)，revision 17** 及[统一评测协议](https://bcnaodifzwip.feishu.cn/wiki/Tj72wi2rdiG5pWk4oGfcvexonlf)。150 题和场景各 3–5 题是文档中的建议规模，题目与运行协议仍需在原版基线阶段冻结。本仓库将 HMMT / Shortlist 合计 20 题分成各 10 题，场景各取 5 题；这些实现选择不追溯写成会议结论。

## 1. 跑哪些能力

| Profile | Task ID | 能力 | subset | full 独立任务 | 实际评分链路 |
|---|---|---|---:|---:|---|
| core | `ifbench` | 指令与输出约束 | 50 | 300 | 官方 strict / loose 规则；主报 loose prompt accuracy |
| core | `gpqa_diamond` | 科学推理 | 30 | 198 | 固定选项排列与标准答案比较 |
| core | `livecodebench` | 代码生成 | 30 | 175 | 官方执行测试，在隔离 Docker 中评分 |
| core | `hmmt_feb_2026` | 数学推理 | 10 | 33 | 最终答案解析与数学判等 |
| core | `apex_shortlist` | 困难数学 | 10 | 47 | 最终答案解析与数学判等 |
| core | `terminal_bench_2_1` | 终端交互 | 10 | 89 | Harbor + Terminus-2 + 任务 verifier |
| core | `deepswe_1_1` | 仓库级编码 | 10 | 113 | Pier + mini-swe-agent + 独立行为 verifier |
| scenarios | `webarena_verified` | 浏览器 | 5 | 812 | 项目 agent 的响应与 HAR，交给官方 evaluator |
| scenarios | `spreadsheetbench_verified` | Excel / 表格 | 5 | 400 | 项目 agent 的工作簿，经重算与官方文件比较 |
| scenarios | `osworld_verified` | 文档、PPT、桌面办公 | 5 | 117 | 原生 PromptAgent 与应用状态 evaluator |
| scenarios | `androidworld` | 手机 | 5 | 116 | 原生 M3A、模拟器与状态检查 |

- `--profile core`：训练期核心 **150** 题；全集 **955** 个独立任务。
- `--profile scenarios`：独立场景试跑 **20** 题；全集 **1,445** 个独立任务，不强制每次 checkpoint 都跑。
- `--profile all`：两部分合计 **170** 题；全集 **2,400** 个独立任务。

`full` 的范围是表中固定版本和领域。LiveCodeBench 选 **v6 增量 test6.jsonl 的 175 题**，实际日期为 2025-01-04 至 2025-04-06；它不是累计 `release_v6` 的 1,055 题，也不额外按月份删题。OSWorld 当前全集限定 Calc / Impress / Writer 三类办公任务，共 117 题，不表示 OSWorld 所有应用。AndroidWorld 按冻结 seed 为 116 个任务类型各生成一个参数实例。

题数和重复次数分开：核心子集每题一次；Terminal-Bench 全集每题 **5 trials**，其余默认每题一次。DeepSWE 正式重复方案仍需基线后冻结。不同能力的分数分别报告。初期 core 与 scenarios 不启用额外 LLM/VLM judge；agent 使用模型完成任务不等于模型充当裁判。

## 2. 准备运行环境和 API

建议在 Linux 评测主机使用 Python 3.12。通用 API、选题和配置检查支持 Windows；代码执行、终端、仓库及 GUI 任务还需要各自环境。以下命令从仓库根目录运行。

```bash
python3.12 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install 'PyYAML>=6,<7' 'pytest>=9,<10' 'huggingface_hub>=0.34,<2' \
  'pyarrow==23.0.1' 'math-verify==0.9.0' 'latex2sympy2-extended==1.11.0'
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
cp configs/eval/.env.example .env
```

PowerShell 对应：`py -3.12 -m venv .venv-eval`、`.\.venv-eval\Scripts\Activate.ps1`、`$env:PYTHONPATH="$PWD\src"`、`Copy-Item configs/eval/.env.example .env`。完整环境任务仍优先在已配置的 Linux 主机运行。

在 `.env` 填写：

| 字段 | 内容 |
|---|---|
| `BENCHMARK_BASE_URL` | OpenAI-compatible API 根地址，例如 `http://127.0.0.1:8000/v1` |
| `BENCHMARK_MODEL` | 服务实际公开的模型名称，不是任意本地 checkpoint 路径 |
| `BENCHMARK_API_KEY` | 被测服务 key；仅保存在本地 |
| `BENCHMARK_REVISION` | 不可变 checkpoint / export 哈希或模型快照；权重变化时更新 |
| `HF_TOKEN` | 可选；下载受限 GPQA 数据时需账户已取得数据访问资格 |
| `LCB_SANDBOX_IMAGE` | 下节构建的 LiveCodeBench Docker 镜像完整 `sha256:` ID |
| `JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY` | 仅后续显式 llm_judge 任务需要；初期核心及场景留空即可 |

命令使用 `--env-file .env` 才加载该文件；已有进程环境变量优先，文件不执行 shell、不做变量替换。真实 key 不写 YAML、不提交 Git。项目入口将 `BENCHMARK_REVISION` 统一应用于所有任务；生成的单任务 YAML 若直接运行，也必须填入实际 `model.revision`。`FILL_*`、`main`、`latest` 等占位或移动版本会被拒绝。

本地明确无认证的服务可设置 `api_key_required: false`。Harbor/Pier 的内部 SDK 仍要求 key 字符串，因此此时使用非密钥占位值 `EMPTY`；不会读取或替换成另一服务的密钥。模型必须预先启动，支持所需工具、图像及上下文预算；本接口不启动训练或模型服务器。

## 3. 下载并冻结数据、规则和环境

查看登记清单，然后准备五项单轮题库：

```bash
python -m qwenpt.eval project list --config configs/eval/project.yaml
python -m qwenpt.eval prepare --config configs/eval/sources.yaml \
  --output data/processed/eval/general --download --env-file .env
```

[sources.yaml](../../configs/eval/sources.yaml) 固定来源、commit、文件、官方 split 和预期题数。`prepare --download` 下载 IFBench、GPQA、LiveCodeBench、HMMT、Shortlist，并准备 IFBench / LiveCodeBench 的官方代码 checkout；不下载整套网站、桌面或模拟器环境。输出每项 `*.jsonl`、`*.source.json`、单任务 `*.yaml` 和 `preparation_summary.json`。各项 `ready: true` 才表示数据准备成功。数据受限、文件缺失或版本不符会明确列出。

离线主机可在来源配置设置 `local_files`，顺序与 `files` 一致，并省略 `--download`；不能用少题文件配较小 `expected_count` 冒充全集。`--tasks ifbench livecodebench` 可只准备指定来源，其他能力仍保留为未准备。

### IFBench 官方规则

```bash
python -m pip install ./.eval-tools/IFBench 'nltk==3.9.2'
python -m nltk.downloader punkt punkt_tab stopwords averaged_perceptron_tagger_eng
```

评分直接调用固定 checkout 的 `evaluation_lib.py`，主指标使用 loose prompt accuracy。汇总中的 `ifbench.strict` / `ifbench.loose` 保留 prompt accuracy、instruction_correct、instruction_total 和按约束总数计算的指令准确率；不会将各题约束均分简单平均。规则 checkout 必须处于精确 commit 且 tracked 文件无修改。NLTK 资源需预先准备；缺失时预检报错，不在评测中自动下载或改用 LLM。

### LiveCodeBench 官方执行沙箱

```bash
docker build -f scripts/eval/Dockerfile.livecodebench -t qwenpt-lcb .
docker image inspect --format '{{.Id}}' qwenpt-lcb
```

将完整镜像 ID 填入 `.env` 的 `LCB_SANDBOX_IMAGE`。镜像身份和官方 grader commit 进入结果记录。生成阶段只发送题面和公开 starter code，测试用例保留在本地。评分使用无网络、只读挂载、非 root、有限 CPU/内存的 Docker 容器；key 不注入评分容器。Docker 不可用时任务保持未完成。`timeout` 为每用例的整数秒数，默认 6 秒。

### 终端与仓库编码

```bash
git clone https://github.com/harbor-framework/terminal-bench-2-1.git .eval-tools/terminal-bench-2-1
git -C .eval-tools/terminal-bench-2-1 checkout --detach 7131e4375048a0e408a8fb404b5f499d726b695b
git clone https://github.com/datacurve-ai/deep-swe.git .eval-tools/deep-swe
git -C .eval-tools/deep-swe checkout --detach 0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea
uv tool install harbor==0.25.0
uv tool install datacurve-pier==0.3.1
```

也可将固定版本安装到独立虚拟环境，在 `project.yaml` 的 `settings.executable` 指定 `harbor` / `pier` 可执行文件。需要运行中的 Docker。DeepSWE 使用 Pier 的独立 verifier 环境，`agent_version: 2.4.6` 固定 mini-swe-agent。项目生成原生 job 配置并读取逐 trial `result.json`；任务保留多步工具执行，不压成一次问答。

### 浏览器、表格、桌面与手机

将下列官方仓库克隆到对应路径，checkout [project.yaml](../../configs/eval/project.yaml) 登记的完整 `revision`，并按该版本官方说明安装环境。每项可用 `settings.executable` 指定独立环境的 **Python** 路径。

| 场景 | 官方源码和本地路径 | 运行前准备 |
|---|---|---|
| WebArena | [ServiceNow/webarena-verified](https://github.com/ServiceNow/webarena-verified)，`.eval-tools/webarena-verified` | Python 3.11+，安装 checkout；网站与 native config；项目浏览器 agent 的响应和 HAR |
| SpreadsheetBench | [RUCKBReasoning/SpreadsheetBench](https://github.com/RUCKBReasoning/SpreadsheetBench)，`.eval-tools/SpreadsheetBench` | 安装官方 requirements；Verified 数据、输入及答案工作簿；项目 agent 输出；LibreOffice 7.5+ 或 Windows Excel / pywin32 |
| OSWorld | [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)，`.eval-tools/OSWorld` | 官方 requirements、VM 镜像、Docker / KVM、截图与控制接口；被测模型视觉服务 |
| AndroidWorld | [google-research/android_world](https://github.com/google-research/android_world)，`.eval-tools/android_world` | 官方 requirements、初始化后的模拟器与应用、Android SDK / adb；被测模型视觉服务 |

WebArena 可用 `python -m pip install -e .eval-tools/webarena-verified` 安装。SpreadsheetBench 的 [Verified 数据入口](https://huggingface.co/datasets/KAKA22/SpreadsheetBench/blob/main/spreadsheetbench_verified_400.tar.gz) 需按项目配置解压到 `data/raw/eval/SpreadsheetBench-Verified/`，保留 `dataset.json` 和完整文件树。OSWorld 和 AndroidWorld 的 VM / 模拟器初始化按对应官方 README 执行；适配器不会替用户初始化手机设备。

WebArena / SpreadsheetBench 接入官方评分，浏览器操作轨迹和工作簿由项目 agent 生成。OSWorld / AndroidWorld 运行原生 PromptAgent / M3A，仅将模型请求接到统一服务，保留原生 action parser 与状态评分。缺环境或缺产物不会被跳过后宣布全套完成。

WebArena / SpreadsheetBench 的 `native_results_path`、`native_run_manifest_file` 指向本轮原始产物及 provenance。先 `project plan --profile scenarios` 冻结 `scenario_plan.json` 和 `selected_tasks.json`，项目 agent 使用其中的 task IDs、模型 revision 与预算。产物布局：

```text
runs/native/webarena/<task_id>/agent_response.json
runs/native/webarena/<task_id>/network.har
runs/native/spreadsheet/<task_id>/<number>_<task_id>_output.xlsx
```

同一道 SpreadsheetBench 题的全部输入工作簿都需要对应输出。provenance JSON 的 `engine`、`revision`、`model_revision`、`split`、`attempts`、`task_ids` 必须与 plan 一致并反映真实来源，不能给其他模型产物改标签。多次尝试时在产物根目录增加 `attempt-000` 等子目录。评测保存原始产物哈希，并在副本上重算工作簿。

## 4. 训练期子集与训后全集

环境和数据准备完成后，检查整套核心，再冻结选题、生成和评分：

```bash
python -m qwenpt.eval project doctor --config configs/eval/project.yaml --profile core \
  --split subset --output runs/eval/base-core-subset --env-file .env
python -m qwenpt.eval project plan --config configs/eval/project.yaml --profile core \
  --split subset --output runs/eval/base-core-subset --env-file .env
python -m qwenpt.eval project run --config configs/eval/project.yaml --profile core \
  --split subset --output runs/eval/base-core-subset --env-file .env
python -m qwenpt.eval project score --config configs/eval/project.yaml --profile core \
  --split subset --output runs/eval/base-core-subset --env-file .env
```

每个 checkpoint 使用独立输出目录和真实 `BENCHMARK_REVISION`，保持数据、agent、harness、上下文、生成预算和 scorer 不变。`plan` 不调用模型；`doctor` 检查全部所选任务的数据、API 配置、参数、密钥、规则依赖和运行环境，再允许生成。该检查不请求模型，不代表服务已启动或支持所有模型专用参数。`run` 先预检所选任务；单轮题库只生成，原生 harness / 场景可能同时启动环境、执行工具和判分；`score` 汇总固定输出。

场景试跑单独使用 `--profile scenarios --split subset` 及新输出目录。只调试一项时可追加 `--tasks ifbench` 或对应 ID，报告会记录 `omitted_profile_tasks`，不能视为完整 profile 已完成。

训后使用 `--profile all --split full`，依次执行：

```bash
python -m qwenpt.eval project doctor --config configs/eval/project.yaml --profile all \
  --split full --output runs/eval/final-all-full --env-file .env
python -m qwenpt.eval project plan --config configs/eval/project.yaml --profile all \
  --split full --output runs/eval/final-all-full --env-file .env
python -m qwenpt.eval project run --config configs/eval/project.yaml --profile all \
  --split full --output runs/eval/final-all-full --env-file .env
python -m qwenpt.eval project score --config configs/eval/project.yaml --profile all \
  --split full --output runs/eval/final-all-full --env-file .env
```

`full` 完整覆盖冻结清单，不提供 `--limit` 缩短全集。前期反复查看的公开子集属于开发诊断数据；全集报告应单列该子集与未参与调参的留出部分，不能把所有题目都称为完全未见。

## 5. 完成度、恢复与报告

项目目录保存 `project_manifest.json`、`project_doctor.json`、`project_plan.json`、`project_run.json`、`project_score.json`，每项任务有独立子目录。

- `complete`：本次操作中所选任务全部完成；plan 成功不表示模型评测结束。
- `profile_coverage_complete`：完整覆盖 profile，未通过 `--tasks` 遗漏任务。
- `all_required_complete`：仅真实 `score`、`--profile all --split full` 且全部任务完成时为 true。缺失、环境异常或 mock 结果会阻止它变成 true。
- 逐项报告分母、尝试次数、得分和失败类型；规则准确率、执行成功率与 rubric 分数不混算。

单轮题库保存 `selection.json`、`manifest.json`、`predictions.jsonl`、生成汇总、错误日志，以及 `scores/<score-tag>/` 下的逐样本评分和汇总。IFBench 保留 strict / loose 及约束级分子分母；LCB 保存官方执行结果。原生任务保存任务 manifest、job / scenario plan、逐 trial 或环境证据、资源统计。

单轮题库重跑同一命令会按指纹跳过已完成样本、补做失败项；同一目录不能并发写入。模型权重、任务、数据、生成参数、适配器、环境或 agent 变化时使用新目录。原生 harness 的中断或失败作业保留原始日志，重试使用新目录；当前接口不导入手工 native resume 的新退出回执。只有匹配协议且成功退出的运行才能发布完整原生成绩。

评分未完整时，`final_mean_score` 为 null，`mean_on_scored` 只是暂时统计。输出预算耗尽和测试失败属于模型在固定预算内的结果；API / 环境 / 判分故障分别登记，不从分母消失。CLI 未完成时退出码为 `2`。

### 多次采样与 seed

`subset.seed` 只控制选题。单轮生成的 `generation.seed` 是**基础 seed**：程序按基础 seed、benchmark、question_id 派生题目偏移，再加入 `sample_index`，得到非负31位请求 seed。同题的32次采样使用32个不同 seed；HTTP 重试、断点补跑使用该次采样原来的 seed。题库顺序、subset/full 或 checkpoint 的变化不影响相同题目与采样序号的 seed，便于原版与训练版对照。

```yaml
generation:
  seed: 20261024
  temperature: 1.0
  max_tokens: 32768
benchmarks:
  aime2026:
    samples: 32
    scorer:
      type: math_equivalence
```

不支持 seed 的 API 请省略 `generation.seed`，或设为 `null`；不会自动发送 seed。`temperature: 0` 或服务忽略 seed 时，改变 seed 仍可能产生相同答案，不能将其当成32次独立随机试验。`avg@32` 是32次正确率的平均，不是 `pass@32`。

`manifest.json` 记录 `question-sample-seed-v1`，每条 prediction 记录实际 `request_seed`，同时保留服务返回的 `system_fingerprint`。升级前固定 seed 的重复采样必须换新目录重新生成，不能直接继续作为 avg@32 使用；程序拒绝混用新旧采样协议。旧的单次采样或未配置 seed 的有效答案可尝试独立 `score --score-tag <新标签>`，仍须通过数据与生成设置校验。

### 兼容性边界

| 环节 | 已支持与检查的契约 | 运行条件 |
|---|---|---|
| 模型／judge API | 单响应 Chat Completions；独立 endpoint/key；分离最终答案与 reasoning；保留服务指纹 | vLLM、SGLang 或远程服务需实际实现该协议；不自动改用 Responses API |
| 生成参数 | `max_tokens` 或 `max_completion_tokens`；支持展开 SDK 风格 `extra_body`，嵌套 seed 同样派生 | 重复字段、路由覆盖及两种 token 上限同时出现会被拒绝；不静默丢弃服务不支持的参数 |
| 拒答／过滤 | 合法拒答作为模型输出交给对应 scorer；内容过滤计零并保留分母 | judge 自身拒答不是有效评分；过滤的部分答案不获取正确分 |
| Windows | 配置、数据准备、选题、HTTP API、规则评测与离线测试 | 数学判等明确要求 Linux；交互环境和 Linux Docker 需另行部署 |
| Linux | 数学验证器运行探测、官方 IFBench 规则、Docker 中官方 LCB 判分 | 真实模型、网站、VM、模拟器仍须在评测主机完成接入测试 |
| Harbor／Pier | 固定 runner/agent 版本，显式 Chat Completions，严格原生退出回执与 task/trial 覆盖 | 固定源码与 Docker；适配器升级需新输出目录 |

视觉 judge 的图片只通过原生图片块发送，不把 base64 图片再次写入文本问题。所有任务分别保留缺失、解析错误、环境失败和模型失败；不同类型不混作一个准确率。

## 6. 单任务 API、离线示例与扩展 judge

原有 `plan|run|score --config <单任务YAML>` 继续可用。离线接口检查：

```bash
python -m qwenpt.eval plan --config configs/eval/demo.yaml --split subset --output runs/eval/demo-subset
python -m qwenpt.eval run --config configs/eval/demo.yaml --split subset --output runs/eval/demo-subset --backend mock
python -m qwenpt.eval score --config configs/eval/demo.yaml --split subset --output runs/eval/demo-subset
```

Mock 仅用于合成数据检查，不是模型成绩；原生交互任务拒绝 mock。自定义 JSONL 每行具有唯一 `benchmark + question_id`，以及 `prompt` 或标准 `messages`。messages 可含图像，但服务必须支持图像、资源必须可访问且版本固定。

| scorer.type | 输入与含义 |
|---|---|
| `multiple_choice` | 明确最终选项：独立选项、末行 Answer: B 或 boxed 选项 |
| `exact_match` | 短答案；默认忽略大小写并规范化空白，可显式调整 |
| `math_equivalence` | 最终数学答案，复用数学验证依赖，建议 boxed 格式 |
| `ifbench` | 官方约束 metadata；不需要参考答案；固定规则 checkout |
| `livecodebench` | 冻结执行 sample；不需要文本答案；固定 Docker image 和 grader |
| `llm_judge` | 参考答案判等或显式 rubric；单独配置 judge |

模型最终 content 与 reasoning_content 分开保存，评分不从长推理中搜索答案。生成参数按服务契约配置，例如 `max_tokens` 或 `max_completion_tokens`，不要同时设置两种上限。上下文不足时不会自动截短问题。

`dataset.expected_counts` 与完整题库一致；`subset.counts` 按固定 seed 和 stratum 均衡分层抽样，或使用 `subset.ids_file` 指定固定 ID（与 counts 二选一）。开发子集不保证按原题库比例抽样，其均分不能当作全集无偏估计。IFBench 约束参数和 LCB 私有测试保留在 metadata，不额外发给被测模型。

HLE、IMO-AnswerBench、ALE 等模型判分候选尚未纳入默认 core / scenarios。后续采用开源裁判或团队选定的新 API 裁判时，显式固定模型版本、rubric、解析、重试、预算及人工复核；没有默认 GPT-4o，也不假定某个服务一定提供某个型号。

```yaml
benchmarks:
  your_judged_task:
    scorer:
      type: llm_judge
      mode: correctness
      rubric: "依据标准答案判断最终回答是否正确，允许等价表达，拒绝事实矛盾。"
      structured_output: json_object
      judge_parameters:
        max_completion_tokens: 1024
judge:
  revision: judge-snapshot-or-frozen-evaluation-period
  base_url_env: JUDGE_BASE_URL
  model_env: JUDGE_MODEL
  api_key_env: JUDGE_API_KEY
  api_key_required: true
```

`correctness` 需要参考答案并输出二值正确率；`mode: rubric` 按量表输出 [0,1] 分数，不能称为准确率。`structured_output` 可选 none、json_object、json_schema，必须与服务支持一致。裁判错误保持未完成。

更换裁判后可复用 predictions，以 `score --score-tag judge-v2` 保存独立评分；原版和训练版使用相同 judge 协议。`demo_judge.yaml` 用 mock 生成后，**score 仍会调用配置的真实裁判服务**。通用 judge 不能代替 IFBench 规则、代码执行或环境状态检查，也不自动等价于候选 benchmark 的官方评分。

## 7. 历史 GPQA / AIME 与验证边界

[bolt_math_gpqa.yaml](../../configs/eval/bolt_math_gpqa.yaml) 的 54 题子集 / 288 题全集保留为历史题库兼容配置，**不是通用模型核心清单**。题库工具 `qwenpt.eval.prepare_benchmarks` 仍可使用。旧 GPU runner、monitor、summarize 已归档在 [legacy/benchmark](../../legacy/benchmark/README.md)，当前项目入口不使用旧 20 作业汇总协议。

150 题核心与独立场景依据当前飞书版本；早前讨论的 380 条扩展方案不作为已确认执行清单。新增 MMLU-Pro、长上下文、多模态或其他候选时，应先冻结来源和范围、接官方 scorer、建立原版基线，再纳入能力门禁。

```bash
python -m pytest tests/eval -q
```

测试覆盖选题、完整性、数据规范化、规则接口、Docker 隔离、原生配置和结果导入等工程契约。设置 `IFBENCH_CHECKOUT` 指向固定官方仓库，可额外运行真实规则依赖测试。离线样例、契约测试及官方样例响应判分均不代表项目模型基线已复现。

实际 API、Docker、网站、VM、模拟器与应用需在目标评测主机验证；数学依赖优先使用已验证的 Linux runtime。doctor 的 blocker 是需要补齐的条件，不能通过删任务、改小题量或填写虚构成绩绕过。
