"""campaign6 geometry add-on: save embeddings for the RQ2 geometry contrast under
the NEW canonical (full-FT + B). Runs after campaign6.

  c6geo_nocl      : canonical minus the contrastive objective (w/o RACL)
  c6geo_attrblock : canonical with attribute-blocked negatives (vs global/B)

Embeddings -> data/final/emb_c6/. Then geom_probe.py reads canon + these two.
"""
from __future__ import annotations
import json, os, subprocess, sys, time

PY = "/root/miniconda3/envs/clm/bin/python"
ROOT = os.path.expanduser("~/claimarc"); SRC = os.path.join(ROOT, "src")
DS = os.path.join(ROOT, "data/final/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl")
EMB = os.path.join(ROOT, "data/final/emb_c6")
ENC = "/root/models/bge-large-zh-v1.5"
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
       "PYTHONPATH": SRC, "TOKENIZERS_PARALLELISM": "false",
       "MODELSCOPE_CACHE": "/root/.cache/modelscope", "HF_HUB_OFFLINE": "1"}
FULL = ["--warmup", "3", "--cl_epochs", "6", "--bs", "12", "--accum", "3"]
CANON = ["--tau", "0.07", "--lambda_cl", "0.5", "--Kp", "3", "--Kn", "5",
         "--loss", "bce", "--encoder_name", ENC, "--enc_train", "full", "--lr", "1e-5"]
B = ["--cl_no_attr_block", "--cl_class_balanced"]

JOBS = [("c6geo_nocl", B + ["--no_cl"]),
        ("c6geo_attrblock", ["--cl_class_balanced"])]


def main():
    for tag, extra in JOBS:
        out = os.path.join(EMB, f"{tag}_s0.pt")
        if os.path.exists(out):
            print(f"[skip] {tag}", flush=True); continue
        cmd = [PY, "-m", "models.train", "--dataset", DS, "--seed", "0", "--tag", tag,
               *FULL, *CANON, *extra, "--save_emb", out]
        print(f"[RUN] {tag}", flush=True)
        subprocess.run(cmd, cwd=SRC, env=ENV)
    print("######## RUN_CAMPAIGN6_GEO COMPLETE ########", flush=True)


if __name__ == "__main__":
    main()
