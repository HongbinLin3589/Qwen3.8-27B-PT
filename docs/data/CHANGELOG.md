# 更新记录

## 仓库集成（2026-10-19）

- 核心模块迁至 `src/qwenpt/data`，统一使用根目录 `pyproject.toml` 安装。
- Python 导入和模块入口改为 `qwenpt.data`；保留 `datapipe` 命令。
- 配置、示例、文档和测试分别迁至 `configs/data`、`examples/data`、`docs/data`、`tests/data`。
- 更新动态后端注册路径与 verl 奖励函数引用；配置相对评测路径仍以配置文件所在目录解析。

## 0.2.1（2026-10-18）

- 修复批次首行缺列导致后续行的 tools、enable_thinking 等字段丢失；整批列并集写入，所有行读回校验，原子替换目标文件。
- SFT 严格校验布尔值、工具定义及参数 JSON Schema、调用配对/顺序、computer_use 坐标范围、媒体类型及占位符数量；默认检查本地媒体路径存在。
- 新增 `datapipe sft --input ... --out ...`，支持已准备好的 JSON、JSONL、Parquet 对话样本直接导入；extra_files 同时支持 Parquet。
- 保留多模态样本显式的空媒体列表；同名输入集合合并；不同工具 schema 自动分文件，SFT 标准列禁止转换成 JSON 字符串。
- 新增 SFT 合约回归测试，覆盖数学/GUI 样例、混合批次、参数约束、图片字节顺序和 CLI 往返。

## 0.2.0（2026-10-17）

- 导出格式可配置（`export.sft` / `export.rl`）：
  - 可改列名、`enable_thinking` 的取法、图片写路径还是嵌入字节、是否补空的媒体列、纯文本与多模态是否分文件。
  - RL 的追加提示、`extra_info` 字段、`ability` 映射也可配置。
- 新增 `datapipe validate`：检查任意 SFT / RL 文件是否符合格式。
- 导出后按 verl 的方式把文件读回来，结果写进 `export_report.json`。
- SFT 可以通过 `extra_files` 导入外部做好的样本。
- 导出器改成注册制（`@register_exporter`），配置项 `export.format` 选用哪个。
- 打包：新增 `pyproject.toml`，安装后可直接用 `datapipe` 命令；默认配置移到 `src/qwenpt/data/defaults.yaml`。
- 配置变化：`export.sft_from_reference` 改为 `export.sft.sources: [reference]`；`export.prompt_suffix`、`rl_verifier_types` 移到 `export.rl` 下（后者改名为 `verifier_types`）。

## 0.1.0（2026-10-17）

- 第 2–9 步流水线：去重、去污染、执行校验、验证器审查、泄漏检查、难度过滤（插件）、教师轨迹、存储字段。
- 数据下载：支持 hf / url / file / git 四种来源，以及 fields / mbpp / humaneval / rllm_code 四种格式转换。
- 判题方式：answer_match / python_asserts / python_function / stdio / command。
- verl 导出及奖励函数 `compute_score`。
- 实测：GSM8K + MBPP；POLARIS-53K + DeepCoder 代码题。
