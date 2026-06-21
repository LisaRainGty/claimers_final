# CLAIMARC — 直播电商虚假宣传识别的检索增强对比学习框架（最终版）

本仓库收录 **CLAIMARC 新架构**（去除属性分块、BGE 全量微调 + 类别均衡监督对比）的全部
可复现资产：核心代码、最终数据集、实验结果、嵌入库与论文（中英双版）。组织目标是
"打开本文件夹即可继续修改与复现"。

> 架构基线（canonical，经用户确认）：BGE 全量微调（`--enc_train full --lr 1e-5`）
> + 方法 B（去属性分块的类别均衡监督对比 `--cl_no_attr_block --cl_class_balanced`）。
> 论文头条指标 **Acc 82.6 / F1 73.4 / AP 75.4 / AUC 90.4**。

## 目录结构

```
claimarc_final/
├── README.md                 # 本说明
├── requirements.txt          # 依赖（含版本）
├── requirements_lock.txt     # 锁定版本
├── env.example.sh            # 环境变量样例（复制为 env.sh 后填入 MATPOOL_API_KEY 等）
├── src/
│   ├── config.py             # 路径 / 模型 / 网关配置（全部读环境变量，无硬编码密钥）
│   ├── common/               # LLM 网关、embedding、IO 等公用模块（LLM 基线需要）
│   └── models/               # 新架构核心代码（精选 27 个文件，见下）
├── data/                     # 最终数据集（本地保留，.gitignore 不入库；见 data/README.md）
├── results/                  # 实验结果 JSON（入库）
│   ├── campaign6_results.jsonl   # 主表 / 消融（重定基新 canonical）
│   ├── campaign7_results.jsonl   # 决策路径 / RACL 细粒度消融
│   ├── campaign8_results.jsonl   # 双流 / 分类头 / 骨干消融
│   ├── alpha_sweep.json          # CLS×RKC 融合系数扫描
│   └── artifacts/                # 图表与案例分析的中间 JSON
├── embeddings/               # canonical 嵌入库 *.pt（本地保留，.gitignore 不入库）
└── paper/                    # 论文源文件（中英双版）+ 图
    ├── claimarc_paper.tex / .pdf         # 英文 LaTeX + 编译 PDF
    ├── claimarc_experiments.tex          # 实验部分独立源
    ├── claimarc_paper_zh.md / .docx      # 中文 Markdown + DOCX
    ├── RESULTS_LOG.md                    # 完整实验结果记录（口径与数值）
    └── figs/                             # 论文图（pdf + png）
```

### `src/models/` 核心文件
| 文件 | 作用 |
| --- | --- |
| `model.py` | CLAIMARC 模型：双流编码 + TwoStreamFusion + RACL + 可靠性加权 |
| `data.py` | 数据加载、分词、划分 |
| `train.py` | 训练 / 评估 / RKC / 阈值搜索（主入口，CLI） |
| `crossdomain.py` | 跨域评估（留一类 / 留主播） |
| `baselines.py` / `baselines_ft.py` / `baselines_frozen.py` / `baselines_neural.py` | 基线（冻结探针 / 微调 / 神经网络）|
| `run_llm_baselines.py` | LLM 基线（经 `common.llm` 网关）|
| `xdom_common.py` / `xdom_fold.py` / `xdom_inject.py` / `xdom_agg.py` / `xdom_llm.py` | 跨域协议、检索库注入、聚合 |
| `run_campaign6/7/8.py`, `run_campaign6_geo.py` | 论文各表的批处理实验编排 |
| `alpha_sweep.py` | CLS×RKC 融合系数扫描 |
| `make_figs.py` / `make_inject_fig.py` / `make_selective_fig.py` / `metrics_rich.py` | 论文图与富指标 |
| `case_confounder.py` / `case_counterfactual.py` | 案例 / 反事实分析 |
| `compile_results.py` | 汇编结果表 |

## 环境准备

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # 或 requirements_lock.txt 复现锁定版本
cp env.example.sh env.sh                 # 填入 MATPOOL_API_KEY（仅 LLM 基线需要）
source env.sh                            # 会设置 PYTHONPATH=src
```

主要依赖：PyTorch 2.6、transformers 4.57、sentence-transformers、scikit-learn、
umap-learn、matplotlib。训练在单卡 RTX 4090 上进行。

## 复现 canonical（新架构）

从 `src/` 目录运行（`PYTHONPATH` 已含 `src`）：

```bash
cd src
python -m models.train \
  --dataset ../data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl \
  --encoder_name BAAI/bge-large-zh-v1.5 \
  --enc_train full --lr 1e-5 \
  --tau 0.07 --lambda_cl 0.5 --Kp 3 --Kn 5 --loss bce \
  --warmup 3 --cl_epochs 6 \
  --cl_no_attr_block --cl_class_balanced \
  --seed 0
```

批量复现论文各表 / 图：

```bash
cd src
python -m models.run_campaign6     # 主表 + 重定基消融
python -m models.run_campaign7     # 决策路径 + RACL 细粒度消融
python -m models.run_campaign8     # 双流 / 分类头 / 骨干消融
python -m models.alpha_sweep       # CLS×RKC 融合扫描
python -m models.make_figs         # 重绘论文图（输出到 paper/figs）
```

## 数据与嵌入说明

按隐私要求，**最终数据集（直播话术 / 评论文本，约 56MB）与嵌入库 `*.pt` 不入 git**，
仅在本文件夹本地保留。克隆仓库后需从原始数据管线另行获取，或向作者索取。
字段说明见 [`data/README.md`](data/README.md)。

## 论文

- 英文：`paper/claimarc_paper.tex` →（`latexmk -xelatex`）`paper/claimarc_paper.pdf`
- 中文：`paper/claimarc_paper_zh.md` →（pandoc）`paper/claimarc_paper_zh.docx`
- 完整数值口径见 `paper/RESULTS_LOG.md`
