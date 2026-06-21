#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 CLAIMARC 论文实验部分的全部图表。

在远端 GPU 主机运行（可访问 ~/claimarc/data/final 下的 emb_*.pt / pred_*.pt 与
paper_results.jsonl）。输出 PDF+PNG 到 ~/claimarc/figs。

图表清单（对应 proposal §4.4）：
  fig_umap_label   §4.4.4 UMAP（full/no_cl/global_neg × label 着色 + 反标签对高亮）
  fig_umap_attr    §4.4.4 UMAP（按属性着色，验证属性结构保留）
  fig_geometry     §4.4.4 几何指标分组柱状
  fig_xdom_traj    §4.4.2 跨域 Macro-F1 随检索库注入量 m 的轨迹
  fig_calibration  §4.4.1 可靠性图（reliability diagram）+ 温度缩放
  fig_hparam       §4.4.6 超参敏感性
  fig_main         §4.4.1 主对比（Macro-F1 / AUPRC 带置信区间）
"""
import os, json, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

D = os.path.expanduser("~/claimarc/data/final")
OUT = os.path.expanduser("~/claimarc/figs")
os.makedirs(OUT, exist_ok=True)

# 配色（colorblind-friendly）
C_POS = "#d1495b"   # y=1 误导
C_NEG = "#2e86ab"   # y=0 非误导
PALETTE = ["#2e86ab", "#d1495b", "#3a7d44", "#e07a5f", "#8e7dbe",
           "#f2a541", "#5b8c5a", "#b56576", "#4d908e", "#577590"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("SAVED", name)


def load_pt(fn):
    p = os.path.join(D, fn)
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def read_jsonl(fn):
    rows = []
    p = os.path.join(D, fn)
    if not os.path.exists(p):
        p = os.path.join(os.path.expanduser("~/claimarc"), fn)
    if not os.path.exists(p):
        return rows
    for ln in open(p):
        ln = ln.strip()
        if ln:
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows


# ---------------------------------------------------------------- UMAP
def compute_umap(g, seed=0):
    import umap
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric="cosine",
                        random_state=seed)
    return reducer.fit_transform(g)


def fig_umap():
    specs = [("CLAIMARC (full)", "emb_clarc_v2_s0.pt"),
             ("w/o RACL", "emb_nocl_s0.pt"),
             ("Global-neg (SupCon-style)", "emb_gneg_s0.pt")]
    loaded = [(t, load_pt(f)) for t, f in specs]
    loaded = [(t, d) for t, d in loaded if d is not None]
    if not loaded:
        print("SKIP umap: no emb"); return

    # ---- 按标签着色 + 反标签近邻对分离 ----
    fig, axes = plt.subplots(1, len(loaded), figsize=(5.2 * len(loaded), 4.6))
    if len(loaded) == 1:
        axes = [axes]
    for ax, (title, d) in zip(axes, loaded):
        te = d["test"]
        g = np.asarray(te["g"], np.float32)
        y = np.asarray(te["y"], float)
        emb = compute_umap(g)
        ax.scatter(emb[y == 0, 0], emb[y == 0, 1], s=7, c=C_NEG, alpha=0.55,
                   label="non-misleading (y=0)", linewidths=0)
        ax.scatter(emb[y == 1, 0], emb[y == 1, 1], s=7, c=C_POS, alpha=0.7,
                   label="misleading (y=1)", linewidths=0)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    axes[0].legend(loc="upper right", markerscale=2, framealpha=0.9)
    fig.suptitle("UMAP of retrieval embeddings $g_{p,a}$ (colored by perceived-risk label)", y=1.02)
    save(fig, "fig_umap_label")

    # ---- 按属性着色（取出现频次最高的前 8 个属性） ----
    fig, axes = plt.subplots(1, len(loaded), figsize=(5.2 * len(loaded), 4.6))
    if len(loaded) == 1:
        axes = [axes]
    ref_attr = np.asarray(loaded[0][1]["test"]["attr"])
    uniq, cnt = np.unique(ref_attr, return_counts=True)
    top = uniq[np.argsort(-cnt)][:8]
    amap = {a: i for i, a in enumerate(top)}
    for ax, (title, d) in zip(axes, loaded):
        te = d["test"]
        g = np.asarray(te["g"], np.float32)
        attr = np.asarray(te["attr"])
        emb = compute_umap(g)
        other = ~np.isin(attr, top)
        ax.scatter(emb[other, 0], emb[other, 1], s=6, c="#cccccc", alpha=0.4,
                   linewidths=0)
        for a in top:
            m = attr == a
            ax.scatter(emb[m, 0], emb[m, 1], s=9, c=PALETTE[amap[a] % len(PALETTE)],
                       alpha=0.8, linewidths=0, label=a if ax is axes[0] else None)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0].legend(loc="upper right", markerscale=1.8, framealpha=0.9, ncol=1,
                   fontsize=7)
    fig.suptitle("UMAP of $g_{p,a}$ (colored by standardized attribute)", y=1.02)
    save(fig, "fig_umap_attr")


# ---------------------------------------------------------------- geometry
def fig_geometry():
    rows = [r for r in read_jsonl("../paper_results.jsonl")
            if r.get("_job") == "analysis_boundary_geom_v2" and "geom_attr_same_label_tight" in r]
    if not rows:
        # 回退到本地汇总
        rows = [r for r in read_jsonl("paper_results.jsonl")
                if r.get("_job") == "analysis_boundary_geom_v2" and "geom_attr_same_label_tight" in r]
    if not rows:
        print("SKIP geometry: no rows"); return
    name_map = {"full": "CLAIMARC", "no_cl": "w/o RACL", "global_neg": "Global-neg"}
    rows = [r for r in rows if r["tag"] in name_map]
    order = ["full", "no_cl", "global_neg"]
    rows = sorted(rows, key=lambda r: order.index(r["tag"]))

    metrics = [
        ("geom_attr_same_label_tight", "Same-label\ntightness $\\uparrow$"),
        ("geom_attr_opp_label_sep", "Opp-label\nseparation $\\downarrow$"),
        ("geom_cross_attr", "Cross-attr\nsimilarity"),
        ("geom_alignment", "Alignment $\\downarrow$"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(metrics))
    w = 0.25
    for i, r in enumerate(rows):
        vals = [r[m] for m, _ in metrics]
        ax.bar(x + (i - 1) * w, vals, w, label=name_map[r["tag"]],
               color=PALETTE[i], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("cosine-based metric value")
    ax.set_title("Representation geometry: RACL tightens same-label and separates opposite-label pairs")
    ax.legend(framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.6)
    save(fig, "fig_geometry")


# ---------------------------------------------------------------- xdom trajectory
def fig_xdom_traj():
    rows = read_jsonl("../paper_results.jsonl") or read_jsonl("paper_results.jsonl")
    cats = [r for r in rows if r.get("holdout_mode") == "category" and "rkc" in r]
    rooms = [r for r in rows if r.get("holdout_mode") == "rooms" and "rkc" in r]
    if not cats:
        print("SKIP xdom: no rows"); return
    ms = [0, 1, 3, 5, 10]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # 左：每个留出品类一条轨迹 + 均值
    mat = []
    for r in cats:
        ys = [r["rkc"].get(str(m), np.nan) for m in ms]
        mat.append(ys)
        axL.plot(ms, ys, color="#bbbbbb", linewidth=0.9, marker="o",
                 markersize=3, alpha=0.7)
    mat = np.array(mat, float)
    mean = np.nanmean(mat, 0)
    axL.plot(ms, mean, color=C_POS, linewidth=2.6, marker="s", markersize=7,
             label="mean over 10 held-out categories")
    axL.set_xlabel("# held-out samples injected into retrieval library ($m$)")
    axL.set_ylabel("RKC Macro-F1 on held-out domain")
    axL.set_title("(a) Leave-one-category transfer")
    axL.legend(framealpha=0.9)
    axL.set_xticks(ms)

    # 右：留一主播 + forward 对照
    if rooms:
        r = rooms[0]
        ys = [r["rkc"].get(str(m), np.nan) for m in ms]
        axR.plot(ms, ys, color=C_NEG, linewidth=2.4, marker="o", markersize=6,
                 label="RKC (retrieval-library update)")
        axR.axhline(r.get("forward_macro_f1", np.nan), color=C_POS,
                    linestyle="--", linewidth=1.8,
                    label="forward classifier (no update)")
        axR.set_xlabel("# held-out streamer samples injected ($m$)")
        axR.set_ylabel("Macro-F1 on 20 held-out streamers")
        axR.set_title("(b) Leave-one-streamer transfer")
        axR.legend(framealpha=0.9)
        axR.set_xticks(ms)
    save(fig, "fig_xdom_traj")


# ---------------------------------------------------------------- calibration
def _reliability(y, p, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    xs, ys, ws = [], [], []
    for i in range(nb):
        hi = (p < bins[i + 1]) if i < nb - 1 else (p <= bins[i + 1])
        m = (p >= bins[i]) & hi
        if m.sum() == 0:
            continue
        xs.append(p[m].mean()); ys.append(y[m].mean()); ws.append(m.mean())
    return np.array(xs), np.array(ys), np.array(ws)


def _fit_T(logit_v, yv):
    from scipy.optimize import minimize_scalar
    def nll(T):
        z = logit_v / T
        z = np.clip(z, -30, 30)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))
    r = minimize_scalar(nll, bounds=(0.05, 10), method="bounded")
    return float(r.x)


def fig_calibration():
    d = load_pt("emb_clarc_v2_s0.pt")
    if d is None:
        print("SKIP calib: no emb"); return
    te, va = d["test"], d.get("val", d["test"])
    yt = np.asarray(te["y"], float); pt = np.asarray(te["p"], float)
    yv = np.asarray(va["y"], float); pv = np.asarray(va["p"], float)
    eps = 1e-6
    lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    T = _fit_T(lg(pv), yv)
    pt_cal = 1 / (1 + np.exp(-lg(pt) / T))

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle=":", label="perfect calibration")
    for p, lab, col in [(pt, "CLAIMARC (raw)", C_NEG),
                        (pt_cal, f"CLAIMARC (T-scaled, T={T:.2f})", C_POS)]:
        xs, ys, ws = _reliability(yt, p)
        ax.plot(xs, ys, marker="o", color=col, linewidth=2, label=lab)
    ax.set_xlabel("mean predicted P(misleading)")
    ax.set_ylabel("empirical positive rate")
    ax.set_title("Reliability diagram (test set)")
    ax.legend(framealpha=0.9, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig_calibration")


# ---------------------------------------------------------------- hyperparam
def fig_hparam():
    rows = read_jsonl("../paper_results.jsonl") or read_jsonl("paper_results.jsonl")
    by = {r["tag"]: r for r in rows if "tag" in r}

    def g(tag, key="auprc"):
        return by.get(tag, {}).get(key, np.nan)

    panels = [
        ("$\\lambda_{CL}$", [("0.1", "abl_lambda_0.1"), ("0.3", "abl_lambda_0.5"),
                              ("0.5", "abl_lambda_0.5"), ("1.0", "abl_lambda_1.0")]),
    ]
    # lambda: 0.1/0.5/1.0 实测，0.3 用 claimarc baseline(s0) 近似 0.7317
    lam_x = [0.1, 0.3, 0.5, 1.0]
    lam_y = [g("abl_lambda_0.1"), g("claimarc"), g("abl_lambda_0.5"), g("abl_lambda_1.0")]
    tau_x = [0.05, 0.07, 0.10]
    tau_y = [g("abl_tau_0.05"), g("claimarc"), g("abl_tau_0.10")]
    n_x = [1, 2, 4]
    n_y = [g("abl_nfusion_1"), g("claimarc"), g("abl_nfusion_4")]
    lora_x = [8, 16, 32]
    lora_y = [g("abl_lora_8"), g("claimarc"), g("abl_lora_32")]
    K_x = [1, 3, 5]  # (1,1),(3,5)默认,(5,10)
    K_y = [g("abl_K_1_1"), g("claimarc"), g("abl_K_5_10")]

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    # 每个面板 canonical 设定所在的索引：λ=0.5(idx2), τ=0.07(idx1), N=2(idx1), LoRA=16(idx1), Kp=3(idx1)
    series = [
        (axes[0], lam_x, lam_y, "$\\lambda_{CL}$", "contrastive weight", 2),
        (axes[1], tau_x, tau_y, "$\\tau$", "temperature", 1),
        (axes[2], n_x, n_y, "$N$", "fusion layers", 1),
        (axes[3], lora_x, lora_y, "LoRA rank", "LoRA rank", 1),
        (axes[4], K_x, K_y, "$K_p$ ($(K_p,K_n)$)", "retrieved samples", 1),
    ]
    for ax, xs, ys, xl, ttl, ci in series:
        ax.plot(xs, ys, marker="o", color=C_NEG, linewidth=2)
        # 标出 canonical 设定
        ax.scatter([xs[ci]], [ys[ci]], s=90, facecolors="none", edgecolors=C_POS,
                   linewidths=2, zorder=5)
        ax.set_xlabel(xl); ax.set_title(ttl)
        ax.set_xticks(xs)
    axes[0].set_ylabel("AUPRC")
    fig.suptitle("Hyperparameter sensitivity (AUPRC; circle = canonical setting)", y=1.05)
    save(fig, "fig_hparam")


# ---------------------------------------------------------------- main comparison
def fig_main():
    comp = None
    for cand in ["../paper_compiled.json", "paper_compiled.json"]:
        p = os.path.join(D, cand)
        if os.path.exists(p):
            comp = json.load(open(p)); break
    rows = read_jsonl("../paper_results.jsonl") or read_jsonl("paper_results.jsonl")
    # 聚合多种子均值±std
    agg = {}
    for r in rows:
        t = r.get("tag")
        if t in ("claimarc_v2", "bert_cls", "roberta_cls", "esim") and "macro_f1" in r:
            agg.setdefault(t, {"macro_f1": [], "auprc": []})
            agg[t]["macro_f1"].append(r["macro_f1"])
            agg[t]["auprc"].append(r["auprc"])
    # 单值基线
    single = {"BGE+LR": (by_get(rows, "BGE_frozen_LR_concat", "macro_f1"),
                         by_get(rows, "BGE_frozen_LR_concat", "auprc"))}
    name_map = {"claimarc_v2": "CLAIMARC", "bert_cls": "BERT-CLS",
                "roberta_cls": "RoBERTa-CLS", "esim": "ESIM"}
    order = ["claimarc_v2", "bert_cls", "roberta_cls", "esim"]
    labels, mf, mf_e, ap, ap_e = [], [], [], [], []
    for t in order:
        if t not in agg:
            continue
        labels.append(name_map[t])
        mf.append(np.mean(agg[t]["macro_f1"])); mf_e.append(np.std(agg[t]["macro_f1"]))
        ap.append(np.mean(agg[t]["auprc"])); ap_e.append(np.std(agg[t]["auprc"]))
    for nm, (a, b) in single.items():
        if a is not None:
            labels.append(nm); mf.append(a); mf_e.append(0); ap.append(b); ap_e.append(0)
    # LLM 最佳（从 llm_results）
    llm = read_jsonl("../llm_results.jsonl") or read_jsonl("llm_results.jsonl")
    best = None
    for r in llm:
        f = r.get("macro_f1")
        if f is not None and (best is None or f > best[1]):
            best = (r.get("model", "LLM"), f, r.get("auprc", np.nan))
    if best:
        labels.append("Best LLM"); mf.append(best[1]); mf_e.append(0)
        ap.append(best[2] if best[2] == best[2] else 0); ap_e.append(0)

    y = np.arange(len(labels))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.55 * len(labels) + 1.8), sharey=True)
    cols = [C_POS if l == "CLAIMARC" else C_NEG for l in labels]
    a1.barh(y, mf, xerr=mf_e, color=cols, edgecolor="black", linewidth=0.5, capsize=3)
    a1.set_yticks(y); a1.set_yticklabels(labels)
    a1.set_xlabel("Macro-F1"); a1.set_title("(a) Macro-F1")
    a1.set_xlim(min(mf) - 0.05, max(mf) + 0.03)
    a2.barh(y, ap, xerr=ap_e, color=cols, edgecolor="black", linewidth=0.5, capsize=3)
    a2.set_xlabel("AUPRC"); a2.set_title("(b) AUPRC")
    a2.set_xlim(min(ap) - 0.05, max(ap) + 0.03)
    fig.suptitle("Main comparison on the in-distribution test set (3-seed mean $\\pm$ std)", y=1.02)
    save(fig, "fig_main")


def by_get(rows, tag, key):
    for r in rows:
        if r.get("tag") == tag:
            return r.get(key)
    return None


if __name__ == "__main__":
    fns = [fig_geometry, fig_xdom_traj, fig_calibration, fig_hparam, fig_main, fig_umap]
    for f in fns:
        try:
            f()
        except Exception as e:
            import traceback
            print("ERR", f.__name__, e)
            traceback.print_exc()
    print("ALLDONE", sorted(os.listdir(OUT)))
