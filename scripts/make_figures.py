"""Publication figures, drawn ONLY from result files (no hand-typed numbers).
  python scripts/make_figures.py --root results/adni --out figures/ [--variants full,no_halluc,lr_concat]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mrxgnn.atlas import AD_SIGNATURE
from mrxgnn.metrics import all_metrics

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--out", default="figures")
ap.add_argument("--classes", nargs="+", default=["CN", "MCI", "AD"])
ap.add_argument("--variants", default="full,no_halluc,no_moddrop,lr_concat")
ap.add_argument("--watermark", default="", help="e.g. 'SYNTHETIC DATA - NOT RESULTS'")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
C = a.classes; pc = [f"p_{c}" for c in C]


def mark(fig):
    if a.watermark:
        fig.text(0.5, 0.5, a.watermark, ha="center", va="center", fontsize=16, color="#d03b3b",
                 alpha=0.25, rotation=20, weight="bold")


def load(v):
    p = os.path.join(a.root, v, "predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


# Fig A: robustness across missing-modality scenarios (per-seed mean +/- sd)
scen = [("observed", "Observed"), ("drop_csf", "No CSF"), ("drop_pet", "No PET"), ("mri_only", "MRI only")]
vs = [v for v in a.variants.split(",") if load(v) is not None]
fig, ax = plt.subplots(figsize=(6.2, 2.8)); mark(fig)
w = 0.8 / max(len(vs), 1)
for i, v in enumerate(vs):
    P = load(v); m, s = [], []
    for sc, _ in scen:
        vals = [all_metrics(q[pc].values, q.y.values, C)["bacc"] for _, q in P[P.scenario == sc].groupby("seed")]
        m.append(100 * np.mean(vals)); s.append(100 * np.std(vals))
    x = np.arange(len(scen)) + (i - (len(vs) - 1) / 2) * w
    ax.bar(x, m, w * 0.92, yerr=s, color=SERIES[i], label=v, capsize=2, error_kw={"lw": 0.8, "ecolor": INK2})
ax.set_xticks(range(len(scen))); ax.set_xticklabels([s for _, s in scen])
ax.set_ylabel("Balanced accuracy (%)"); ax.set_ylim(100 / len(C) - 5, 100)
ax.axhline(100 / len(C), color=INK2, lw=0.8, ls="--"); ax.text(3.45, 100 / len(C) + 1, "chance", color=INK2, fontsize=7, ha="right")
ax.legend(frameon=False, ncol=len(vs), loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=8)
fig.savefig(os.path.join(a.out, "fig_robustness.png")); plt.close(fig)

# Fig B: confusion matrix (full, observed, seed-averaged probabilities)
P = load("full")
if P is not None:
    Q = P[P.scenario == "observed"].groupby("subject_id").agg({**{c: "mean" for c in pc}, "y": "first"})
    cm = np.array(all_metrics(Q[pc].values, Q.y.values, C)["confusion"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.0, 2.7)); mark(fig)
    ax.imshow(cmn, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#ffffff", "#2a78d6", "#104281"]), vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(C)):
        for j in range(len(C)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n(n={int(cm[i, j])})", ha="center", va="center", fontsize=7,
                    color="#ffffff" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(C))); ax.set_xticklabels(C); ax.set_yticks(range(len(C))); ax.set_yticklabels(C)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.savefig(os.path.join(a.out, "fig_confusion.png")); plt.close(fig)

# Fig C: deletion faithfulness curve
ej = os.path.join(a.root, "full", "explain.json")
if os.path.exists(ej):
    E = json.load(open(ej))["explanations"]
    k = E[0]["deletion"]["k"]
    fig, ax = plt.subplots(figsize=(3.2, 2.6)); mark(fig)
    for key, lab, col in (("drop_top", "Top-k attributed ROIs", SERIES[0]), ("drop_random", "Random k ROIs", SERIES[1])):
        M = np.array([e["deletion"][key] for e in E])
        ax.plot(k, M.mean(0), "-o", color=col, lw=2, ms=4, label=lab)
        ax.fill_between(k, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=col, alpha=0.15, lw=0)
    ax.set_xscale("log", base=2); ax.set_xticks(k); ax.set_xticklabels(k)
    ax.set_xlabel("ROIs reset to normative (k)"); ax.set_ylabel("Drop in true-class prob.")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(a.out, "fig_deletion.png")); plt.close(fig)

# Fig D: top ROIs for AD subjects, marking AD-signature membership
ip = os.path.join(a.root, "full", "roi_importance.csv")
if os.path.exists(ip):
    I = pd.read_csv(ip)
    rois = [c for c in I.columns if c not in ("subject_id", "y", "seed", "fold")]
    ad = C.index("AD") if "AD" in C else len(C) - 1
    s = I[I.y == ad][rois].mean().sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(3.6, 3.6)); mark(fig)
    cols = [SERIES[0] if r in AD_SIGNATURE else "#b7b6b0" for r in s.index]
    ax.barh(range(len(s)), s.values, color=cols, height=0.7)
    ax.set_yticks(range(len(s))); ax.set_yticklabels([r.replace("_", " ") for r in s.index], fontsize=7)
    ax.set_xlabel("Mean |Integrated Gradients| (AD subjects)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label="a-priori AD-signature ROI"), Patch(color="#b7b6b0", label="other ROI")],
              frameon=False, fontsize=7, loc="lower right")
    fig.savefig(os.path.join(a.out, "fig_top_rois.png")); plt.close(fig)

# Fig E: PET reliance per ROI (1 - gate), subjects with PET
gp = os.path.join(a.root, "full", "gates.csv")
if os.path.exists(gp):
    G = pd.read_csv(gp)
    rois = [c for c in G.columns if c not in ("y", "m_pet", "seed", "fold")]
    fig, ax = plt.subplots(figsize=(6.2, 2.4)); mark(fig)
    for i, c in enumerate(C):
        g = 1 - G[(G.m_pet == 1) & (G.y == i)][rois].mean()
        ax.plot(range(len(rois)), g.values, lw=1.5, color=SERIES[i], label=c)
    ax.set_xticks(range(0, len(rois), 4)); ax.set_xticklabels([rois[j].replace("_", " ") for j in range(0, len(rois), 4)], rotation=75, fontsize=5.5)
    ax.set_ylabel("PET reliance (1 - g)"); ax.legend(frameon=False, ncol=3, fontsize=7)
    fig.savefig(os.path.join(a.out, "fig_gates.png")); plt.close(fig)

# Fig F: reliability diagram
if P is not None:
    Q = P[P.scenario == "observed"].groupby("subject_id").agg({**{c: "mean" for c in pc}, "y": "first"})
    prob, y = Q[pc].values, Q.y.values
    conf, corr = prob.max(1), (prob.argmax(1) == y)
    edges = np.linspace(0, 1, 11); mids, accs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.sum() >= 5:
            mids.append(conf[s].mean()); accs.append(corr[s].mean())
    fig, ax = plt.subplots(figsize=(2.8, 2.6)); mark(fig)
    ax.plot([0, 1], [0, 1], "--", color=INK2, lw=0.8)
    ax.plot(mids, accs, "-o", color=SERIES[0], lw=2, ms=4)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_xlim(0.3, 1); ax.set_ylim(0.3, 1)
    fig.savefig(os.path.join(a.out, "fig_calibration.png")); plt.close(fig)
print("figures written to", a.out)
