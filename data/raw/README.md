# data/raw — 原始数据

原始直播/电商数据(本地保留,**不入 git**,约 34GB)。从 `../claimarc/data/raw` 迁入即可
(同盘 `mv` 瞬时、零额外占用)：

```bash
mv "../claimarc/data/raw/"* "data/raw/"
```

| 子目录 | 内容 | 规模 |
| --- | --- | --- |
| `comment/` | 评论(≈28,556 条),Stage A 输入 | ≈5.6GB |
| `srt_cut/` | 直播切片字幕(主播话术来源),Stage B 输入 | ≈3.3GB |
| `product_images/` | 商品详情图(视觉证据),Stage C 输入 | ≈25GB |

下游用法见 `docs/DATA_PIPELINE.md`。
