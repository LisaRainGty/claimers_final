"""CLAIMARC campaign6: re-base every paper table on the NEW canonical and add the
frozen-encoder library-injection cross-domain study.

NEW canonical (user-confirmed): BGE full fine-tuning + method B
  = full-FT (enc_train=full, lr1e-5) + RACL with attribute-blocking removed and
    class-balanced supervised-contrastive weighting (--cl_no_attr_block --cl_class_balanced).
  Base schedule: bce / lambda_cl0.5 / tau0.07 / K(3,5) / warmup3 + cl6 / bs12 accum3.

Waves (all resumable; one nohup drives the whole queue):
  W1a  c6_canon x3 seeds + --save_emb            -> main table, geometry, figures, selective
  W1b  xdom leave-one-category (10 folds, s0)     -> cross-domain panel (a) + injection
  W1c  xdom leave-20-streamers (s0,1,2)           -> cross-domain panel (b) + injection
  W2   re-based ablations (7 x 3 seeds)           -> ablation table (curated)
  W3   c-formula hyperparameter resweep (1 seed)  -> appendix c-sensitivity

train RESULT lines -> campaign6_results.jsonl (resume by _job tag).
xdom bundles -> data/final/xdom_c6/ (resume by .pt existence); offline analyzers read these.
"""
from __future__ import annotations
import json, os, subprocess, sys, time

PY = "/root/miniconda3/envs/clm/bin/python"
ROOT = os.path.expanduser("~/claimarc")
SRC = os.path.join(ROOT, "src")
DS = os.path.join(ROOT, "data/final/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl")
OUT = os.path.join(ROOT, "data/final/campaign6_results.jsonl")
EMB = os.path.join(ROOT, "data/final/emb_c6"); os.makedirs(EMB, exist_ok=True)
XD = os.path.join(ROOT, "data/final/xdom_c6"); os.makedirs(XD, exist_ok=True)
ENC = "/root/models/bge-large-zh-v1.5"
BERT = "/root/models/bert-base-chinese"
ROBERTA = "/root/models/chinese-roberta-wwm-ext"
ENV = {**os.environ,
       "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
       "PYTHONPATH": SRC, "TOKENIZERS_PARALLELISM": "false",
       "MODELSCOPE_CACHE": "/root/.cache/modelscope", "HF_HUB_OFFLINE": "1"}

FULL = ["--warmup", "3", "--cl_epochs", "6", "--bs", "12", "--accum", "3"]
CANON = ["--tau", "0.07", "--lambda_cl", "0.5", "--Kp", "3", "--Kn", "5",
         "--loss", "bce", "--encoder_name", ENC, "--enc_train", "full", "--lr", "1e-5"]
B = ["--cl_no_attr_block", "--cl_class_balanced"]   # method B (new canonical add-on)
S3 = (0, 1, 2)
CATS = ["apparel_and_underwear", "general", "baby_kids_and_pets", "shoes_and_bags",
        "food_and_beverages", "smart_home", "digital_and_electronics",
        "sports_and_outdoor", "beauty_and_personal_care", "jewelry_and_collectibles"]


def tjob(tag, extra, seeds=S3, save=False):
    jobs = []
    for s in seeds:
        cmd = [PY, "-m", "models.train", "--dataset", DS, "--seed", str(s),
               "--tag", tag, *FULL, *CANON, *extra]
        if save:
            cmd += ["--save_emb", os.path.join(EMB, f"{tag}_s{s}.pt")]
        jobs.append(("train", f"{tag}__s{s}", cmd))
    return jobs


def xjob(mode, holdout, seed):
    label = holdout[:24] if mode == "category" else mode
    bundle = os.path.join(XD, f"clarc_{mode}_{label}_s{seed}.pt")
    cmd = [PY, "-m", "models.xdom_fold", "--dataset", DS, "--mode", mode,
           "--outdir", XD, "--seed", str(seed), "--models", "clarc",
           "--warmup", "3", "--cl_epochs", "6", "--enc_train", "full", "--lr", "1e-5",
           "--cl_no_attr_block", "--cl_class_balanced"]
    if mode == "category":
        cmd += ["--holdout", holdout]
    return ("xdom", f"xdom_{mode}_{label}_s{seed}", cmd, bundle)


JOBS = []
# ---------- W1a: new canonical (3 seeds, save embeddings) ----------
JOBS += tjob("c6_canon", B, save=True)
# ---------- W1b: leave-one-category cross-domain (s0) ----------
for c in CATS:
    JOBS.append(xjob("category", c, 0))
# ---------- W1c: leave-20-streamers cross-domain (3 seeds) ----------
for s in S3:
    JOBS.append(xjob("rooms", "", s))
# ---------- W2: re-based ablations (curated core) ----------
JOBS += tjob("c6_no_cl", B + ["--no_cl"])
JOBS += tjob("c6_no_fusion", B + ["--no_fusion"])
JOBS += tjob("c6_no_weight", B + ["--no_weight"])
JOBS += tjob("c6_attrblock", ["--cl_class_balanced"])              # add attribute blocking back
JOBS += tjob("c6_args", B + ["--evidence_policy", "args_only"])    # argument view only
JOBS += tjob("c6_lora", ["--cl_no_attr_block", "--cl_class_balanced",
                         "--enc_train", "lora", "--lr", "2e-5"], save=True)  # efficient variant
JOBS += tjob("c6_bert", B + ["--backbone", "bert", "--encoder_name", BERT])
# ---------- W3: c-formula hyperparameter resweep (1 seed) ----------
for spec, tag in [("k=1.5,lambda=0.3,rho=0.4,phi=1.2", "c6_cf_k1p5"),
                  ("k=6,lambda=0.3,rho=0.4,phi=1.2", "c6_cf_k6"),
                  ("k=3,lambda=0.1,rho=0.4,phi=1.2", "c6_cf_lam0p1"),
                  ("k=3,lambda=0.6,rho=0.4,phi=1.2", "c6_cf_lam0p6"),
                  ("k=3,lambda=0.3,rho=0.2,phi=1.2", "c6_cf_rho0p2"),
                  ("k=3,lambda=0.3,rho=0.6,phi=1.2", "c6_cf_rho0p6"),
                  ("k=3,lambda=0.3,rho=0.4,phi=1.0", "c6_cf_phi1p0"),
                  ("k=3,lambda=0.3,rho=0.4,phi=1.5", "c6_cf_phi1p5")]:
    JOBS += tjob(tag, B + ["--c_recompute", spec], seeds=(0,))


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


def run_proc(job_key, cmd, want_prefixes=("RESULT ",)):
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
            elif line.startswith("CLARC_RES "):
                got = True  # xdom: bundle existence is the resume signal
        p.wait()
    except Exception as e:
        print(f"[JOB-ERR] {job_key}: {e}", flush=True)
    print(f"[DONE] {job_key} in {(time.time()-t0)/60:.1f} min (got={got})", flush=True)
    return got


def main():
    done = done_tags()
    print(f"[campaign6] {len(JOBS)} jobs, {len(done)} train-tags already done", flush=True)
    for item in JOBS:
        kind, key = item[0], item[1]
        if kind == "train":
            if key in done:
                print(f"[skip] {key}", flush=True); continue
            got = run_proc(key, item[2])
            if not got:
                log_result(key, {"error": "no_result"})
        else:  # xdom
            cmd, bundle = item[2], item[3]
            if os.path.exists(bundle):
                print(f"[skip] {key} (bundle exists)", flush=True); continue
            run_proc(key, cmd)
    print("\n######## RUN_CAMPAIGN6 COMPLETE ########", flush=True)


if __name__ == "__main__":
    main()
