# Assignment 4 实现与验证

由 Codex 参考只读 `cs336-assignments-refer/assignment4-data`，按当前 2026 handout 的算法接口补全。Modal 配置现从 `SUNET_ID` 环境变量读取；本地数据下载无需设置该变量，远端 Modal 入口仍要求显式设置。

## 实现范围

- `tests/adapters.py`：连接 HTML、语言识别、PII、NSFW/toxic、quality、Gopher、两种去重共 11 个接口。
- `cs336_data/`：迁移提取、PII、Gopher、两遍全局行去重；实现共享模型加载；修复参考质量预测的置信度语义；改进 MinHash/LSH 的独立 hash、短文本处理、连通簇及随机代表。
- `wet_files.py`：完成显式 `is_english` TODO，保持概率阈值 0.7。
- `scripts/train_quality_classifier.py`：用独立采集的正负 JSONL 训练 fastText，标签为 `hq/lq`；不使用测试样本训练，不自动开始训练。
- `scripts/filter_wet.py`：多进程 WET 过滤、PII 掩码和逐阶段统计；语言、有害、模型质量过滤按 CLI 开启。Gopher 默认开启。
- `scripts/tokenize_data.py`：JSONL 文档使用 GPT-2 编码，每篇追加 EOS，流式写入 `np.uint16` `.bin` 并报告 token 数。

## 模型资源

在本机下载 assignment 数据时运行 `uv run scripts/download_data.py --offline-only`，无需填写 `SUNET_ID`，资源写入项目的 `local-shared-data/`。不加 `--offline-only` 会继续下载完整 WET/Wikipedia 数据，体量更大。通过 Modal 运行时，先在 shell 设置真实 `SUNET_ID` 环境变量；不要把个人 ID 写进仓库。

默认在 `/shared-data/classifiers`、项目 `local-shared-data/classifiers`、项目 `data/classifiers` 中找模型，兼容参考实现旧名称。路径相对于项目而非 shell 当前目录。也可设置：

| 模型 | 环境变量 | 当前下载脚本的文件名 |
| --- | --- | --- |
| 语言 | `CS336_LANGUAGE_MODEL_PATH` | `lid.176.bin` |
| NSFW | `CS336_NSFW_MODEL_PATH` | `dolma_fasttext_nsfw_jigsaw_model.bin` |
| toxic | `CS336_TOXIC_MODEL_PATH` | `dolma_fasttext_hatespeech_jigsaw_model.bin` |
| quality | `CS336_QUALITY_MODEL_PATH` | 默认 `runs/quality_fasttext.bin`，需自行训练 |

前三个模型的官方来源已存在于 `scripts/download_data.py`。该脚本的 `--offline-only` 也会下载 WARC、WET 和验证集，需考虑真实数据体量。仅下载分类模型可在项目目录执行以下 macOS/Linux 命令；Windows CMD 可将 `export` 改为 `set`，并按行执行。本次已将三个官方模型下载到被 Git 忽略的 `local-shared-data/classifiers/`，总量约 2.1 GB；没有把二进制模型加入提交包。

```bash
mkdir -p local-shared-data/classifiers
curl -fL https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -o local-shared-data/classifiers/lid.176.bin
curl -fL https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin -o local-shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin
curl -fL https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin -o local-shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin
```

课程不提供预训练质量模型。本机已从 Wikipedia 优良条目 API 收集 1,200 篇正例，并从下载的 Common Crawl WET 样本独立抽取 1,200 篇负例，保存于被 Git 忽略的 `local-shared-data/quality-training/`。训练产物位于被忽略的 `runs/quality_fasttext.bin`。复现训练：

```bash
uv run python scripts/train_quality_classifier.py --positive local-shared-data/quality-training/positive.jsonl --negative local-shared-data/quality-training/negative.jsonl --output runs/quality_fasttext.bin --epochs 10
uv run python scripts/filter_wet.py input.warc.wet.gz --output-dir runs/filtered --workers 2
uv run python scripts/tokenize_data.py runs/filtered/00000.jsonl --output runs/filtered_tokens.bin
uv run pytest -q
```

有模型后可为过滤命令追加 `--check-language --check-harmful --min-quality 0.5`。每个过滤 worker 单独载入模型，workers 越多内存消耗越高。过滤不自动做全语料去重，可按研究方案调用独立去重函数。staff 的 LM 训练入口保持原样。

## 验证记录（2026-10-16）

首先借用只读参考目录现有 `.venv/bin/python` 的依赖执行目标项目测试；随后在目标项目创建隔离 `.venv` 安装最小验证依赖（未修改 pyproject/lock），最终使用目标 `.venv/bin/python` 重跑，结果一致。目标源代码由当前工作目录导入。

- 初次全套测试：16 passed，5 failed；5 项都因模型资源缺失而失败，未添加 skip 或伪预测。
- 补齐官方语言、NSFW、toxic 模型后，全套测试曾为 20 passed，1 failed；随后生成质量模型并单独运行 `uv run pytest tests/test_quality.py::test_classify_quality -q --tb=short`：1 passed。全套测试尚未在新模型生成后重跑。
- 质量模型使用 1,200 篇 Wikipedia 优良条目和 1,200 篇 Common Crawl 文档训练；没有使用测试夹具作为训练样本，也没有用规则伪造预测。
- 通过独立断言：不同短文不误合并、短文完全重复归为一簇、低质量标签置信度正确、fastText 单行输入/概率边界、三个新增脚本可导入。
- 通过临时合成 WET 的真实进程池 smoke：2 条记录输入，1 条通过、1 条被 Gopher 丢弃、email 被掩码，阶段统计正确。发现并修复参考 email 正则无法匹配句末 `hello@example.com.` 的问题。
- tokenizer 使用 mock 替代 GPT-2 下载，仅验证写出 `[10, 20, 50256]`，共 3 个 uint16 token / 6 bytes；未据此宣称真实 tokenizer 或全量数据已经验证。
- 未执行：质量分类真实语料采集/训练、2500 WET 全量过滤、GPT-2 资源下载、LM 训练和外部提交。
- 手册要求的真实数据人工分析、吞吐量、token 总数、最终 perplexity/曲线依赖实际数据及训练；这些结果没有编造。

以上为算法实现维护，不属于重要模型实验，没有生成实验结果声明。


## 最后独立复核与标准环境验证

独立复核后修正 MinHash 的标点归一化：按手册直接删除标点，而非插入空格，`Café's re-sume` 与 `cafes resume` 现在一致。两种去重都拒绝非空输出目录，防止上次输出残留被本轮淘汰的文档；扁平输出拒绝同名输入，防止无声覆盖。`uv run test/dedup_edge_checks.py` 的附着标点/重音和旧输出保护回归通过。

随后已在目标目录成功执行标准 `uv sync`，安装 pyproject/uv.lock 声明的完整依赖，再执行 `uv run pytest -q --tb=line`：**16 passed、5 failed，0 skipped**，与最小环境一致，完整输出见 `pytest-final.txt`。5 项仍仅因缺少四个真实分类模型；不得把模型缺失说成测试通过。没有修改依赖约束以绕过失败。未执行真实分类器训练或模型下载。


## 建议执行的 Git 命令

在本作业目录执行。以下仅为建议，本次没有执行暂存、提交或推送；未纳入无关 `.DS_Store` 文件。

```sh
git status --short
git diff
git add -- cs336_data/classifiers.py cs336_data/exact_deduplication.py cs336_data/extract_text.py cs336_data/harmful_content.py cs336_data/language_identification.py cs336_data/minhash_deduplication.py cs336_data/pii_masking.py cs336_data/quality_classifier.py cs336_data/quality_rules.py cs336_data/wet_files.py doc/implementation.md doc/pytest-final.txt scripts/filter_wet.py scripts/tokenize_data.py scripts/train_quality_classifier.py test/dedup_edge_checks.py tests/adapters.py
git diff --cached
git commit -m 'feat: implement assignment4-data TODOs'
```
