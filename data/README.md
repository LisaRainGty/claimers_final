# 数据集说明

本目录根存放论文最终数据集;`raw/ processed/ index/ final/` 为流水线各层(各自带
README)。出于隐私/体积考虑,所有数据文本**不纳入 git**(见根 `.gitignore`),仅本地保留。

> 从原始数据到本目录最终数据集的完整构建链路见 [`../docs/DATA_PIPELINE.md`](../docs/DATA_PIPELINE.md)。
> 各层数据用同盘 `mv ../claimarc/data/<layer>/* <layer>/` 迁入。

## 文件

| 文件 | 规模 | 说明 |
| --- | --- | --- |
| `dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl` | ~18MB | 监督训练集（主实验使用）|
| `dataset_duallabel_FULLPOOL_PLUS_OBJNEG_all_20260615.jsonl` | ~38MB | 全量（含未监督 / 检索池条目）|

`supervised` 为论文主实验与消融的训练 / 评估口径；`all` 额外包含构成检索池的全部条目。

## 主要字段

| 字段 | 含义 |
| --- | --- |
| `pair_id` / `product_id` / `room_id` | 样本 / 商品 / 直播间标识 |
| `category` / `subcategory` / `attribute_id` / `attribute_name` | 类目与被核验属性 |
| `product_title` | 商品标题 |
| `y` | 客观核验标签（1=虚假宣传）|
| `y_perception` | 消费者感知标签 |
| `c` / `c_reliability` | 每样本可靠性权重 |
| `sample_role` | 样本角色（`supervised_main` 等）|
| `contrastive_mask` | 是否参与对比学习 |
| `claim` | 主播话术（passage + 带时间戳的 segments）|
| `evidence_params` / `evidence_ocr` | 证据：商品参数、详情图 OCR 等 |
| `arguments` | LLM 生成的支持 / 反驳 / 证据缺口论证（无标签泄漏）|

> 注：`y` 为客观核验标签，推理侧主路径用前向分类头（CLS）；RKC（检索近邻分类）与 CLS
> 配合支持人工 abstain。
