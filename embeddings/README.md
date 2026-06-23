# 嵌入库说明

本目录存放 canonical 模型导出的句向量 / 检索嵌入库（`*.pt`）。
`*.pt` 文件**不纳入 git**（可由训练脚本重新导出），仅在本地保留。

## 当前内容：RQ2 表征几何三变体（§4.6）

由 `run_geom_campaign.py` 导出，每个 bundle 含 `train` / `val` / `test` 三划分的
`g`(检索嵌入) / `p`(前向概率) / `y` / `c` / `attr` / `pair_id`，供 `geom_probe2.py`
与 `make_geom_figs.py` 复用（训练集作检索索引、测试集作查询）。

| 文件 | 说明 |
| --- | --- |
| `emb_geom_none_s{0,1,2}.pt` | 无对比（仅 BCE）变体，3 种子 |
| `emb_geom_supcon_s{0,1,2}.pt` | 标准监督对比 SupCon（Khosla 2020）变体，3 种子 |
| `emb_geom_racl_s{0,1,2}.pt` | 本文 RACL（canonical 对比目标）变体，3 种子 |

## 重新生成

```bash
cd src
python -m models.run_geom_campaign \
  --dataset ../data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl \
  --outdir ../embeddings --seeds 0 1 2
```

其余实验（RQ1/RQ3）的临时嵌入由对应 campaign / xdom 脚本即时导出，不长期保留于此。
