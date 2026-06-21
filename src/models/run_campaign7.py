"""CLAIMARC campaign7: expanded ablation battery for the multi-table ablation section.

All jobs sit on the NEW canonical (BGE full fine-tuning + method B):
  full-FT (enc_train=full, lr1e-5) + RACL with attribute-blocking removed and
  class-balanced supervised contrast (--cl_no_attr_block --cl_class_balanced).
  Base schedule: bce / lambda_cl0.5 / tau0.07 / Kp3 Kn5 / warmup3 + cl6 / bs12 accum3.

New variants (campaign6 already covers no_cl / no_fusion / no_weight / attrblock /
args_only / lora / bert):

  Stream-design (dual-stream necessity):
    c7_claim_only   single stream, claim mirrored into both slots, no fusion
    c7_evid_only    single stream, evidence mirrored into both slots, no fusion

  Data/evidence composition:
    c7_sources_only raw retrieved sources only (no LLM argument view)

  RACL retrieval design (RGCL-style: Mei et al. 2024, Tables 3/F):
    c7_hardpos      hard positives (lowest-sim same-label) instead of pseudo-gold
    c7_kp1 / c7_kp5 number of pseudo-gold positives (1 / 5 vs canonical 3)
    c7_kn1 / c7_kn10 number of hard negatives (1 / 10 vs canonical 5)
    c7_negfilter    hard negatives restricted to same evidence-type
    c7_no_classbal  drop class-balanced weighting (plain global supervised contrast)

train RESULT lines -> campaign7_results.jsonl (resume by _job tag).
"""
from __future__ import annotations
import json, os, subprocess, sys, time

PY = "/root/miniconda3/envs/clm/bin/python"
ROOT = os.path.expanduser("~/claimarc")
SRC = os.path.join(ROOT, "src")
DS = os.path.join(ROOT, "data/final/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl")
OUT = os.path.join(ROOT, "data/final/campaign7_results.jsonl")
ENC = "/root/models/bge-large-zh-v1.5"
ENV = {**os.environ,
       "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
       "PYTHONPATH": SRC, "TOKENIZERS_PARALLELISM": "false",
       "MODELSCOPE_CACHE": "/root/.cache/modelscope", "HF_HUB_OFFLINE": "1"}

FULL = ["--warmup", "3", "--cl_epochs", "6", "--bs", "12", "--accum", "3"]
CANON = ["--tau", "0.07", "--lambda_cl", "0.5", "--Kp", "3", "--Kn", "5",
         "--loss", "bce", "--encoder_name", ENC, "--enc_train", "full", "--lr", "1e-5"]
B = ["--cl_no_attr_block", "--cl_class_balanced"]   # method B (new canonical add-on)
S3 = (0, 1, 2)


def tjob(tag, extra, seeds=S3):
    jobs = []
    for s in seeds:
        cmd = [PY, "-m", "models.train", "--dataset", DS, "--seed", str(s),
               "--tag", tag, *FULL, *CANON, *extra]
        jobs.append(("train", f"{tag}__s{s}", cmd))
    return jobs


JOBS = []
# ---- stream-design (dual-stream necessity) ----
JOBS += tjob("c7_claim_only", B + ["--no_fusion", "--stream_mode", "claim"])
JOBS += tjob("c7_evid_only",  B + ["--no_fusion", "--stream_mode", "evidence"])
# ---- data/evidence composition ----
JOBS += tjob("c7_sources_only", B + ["--evidence_policy", "sources_only"])
# ---- RACL retrieval design (RGCL-style) ----
JOBS += tjob("c7_hardpos",     B + ["--cl_hard_pos"])
JOBS += tjob("c7_kp1",         B + ["--Kp", "1"])
JOBS += tjob("c7_kp5",         B + ["--Kp", "5"])
JOBS += tjob("c7_kn1",         B + ["--Kn", "1"])
JOBS += tjob("c7_kn10",        B + ["--Kn", "10"])
JOBS += tjob("c7_negfilter",   B + ["--cl_neg_filter", "same_evtype"])
JOBS += tjob("c7_no_classbal", ["--cl_no_attr_block"])   # drop class balance, keep global contrast


def done_tags():
    if not os.path.exists(OUT):
        return set()
    tags = set()
    for line in open(OUT):
        try:
            tags.add(json.loads(line)["_job"])
        except Exception:
            pass
    return tags


def log_result(job_key, payload):
    payload["_job"] = job_key
    payload["_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_proc(job_key, cmd):
    print(f"\n{'='*72}\n[RUN] {job_key}\n{'='*72}", flush=True)
    t0 = time.time(); got = False
    try:
        p = subprocess.Popen(cmd, cwd=SRC, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1, env=ENV)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            if line.startswith("RESULT "):
                try:
                    log_result(job_key, json.loads(line[7:])); got = True
                except Exception as e:
                    print(f"[parse-err] {e}", flush=True)
        p.wait()
    except Exception as e:
        print(f"[JOB-ERR] {job_key}: {e}", flush=True)
    print(f"[DONE] {job_key} in {(time.time()-t0)/60:.1f} min (got={got})", flush=True)
    return got


def main():
    done = done_tags()
    print(f"[campaign7] {len(JOBS)} jobs, {len(done)} train-tags already done", flush=True)
    for kind, key, cmd in JOBS:
        if key in done:
            print(f"[skip] {key}", flush=True); continue
        got = run_proc(key, cmd)
        if not got:
            log_result(key, {"error": "no_result"})
    print("\n######## RUN_CAMPAIGN7 COMPLETE ########", flush=True)


if __name__ == "__main__":
    main()
