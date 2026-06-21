# 嵌入库说明

本目录存放 canonical 模型导出的句向量 / 原型嵌入库（`*.pt`），供 `alpha_sweep.py`、
几何图（`make_figs.py` 的 UMAP / 几何子图）与检索分析复用。

`*.pt` 文件**不纳入 git**（可由训练脚本重新导出），仅在本地保留。

| 文件 | 说明 |
| --- | --- |
| `emb_clarc_s0..s4.pt` | canonical（BGE 全量微调 + 方法 B）多种子嵌入库 |
| `emb_full.pt` | 全量条目嵌入（检索池）|
| `emb_nocl_s0.pt` | 去对比学习消融 |
| `emb_gneg_s0.pt` | 全集合随机负样本消融 |

重新生成：运行 `python -m models.run_campaign6`（带 `save` 的训练任务会导出 `clarc_*.pt`），
或在 `train.py` 中启用嵌入导出。
