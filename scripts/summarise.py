"""Turn predictions and explanations into the paper's tables. Every number in the manuscript should
come from the files this script writes.
  python scripts/summarise.py --root results/adni --classes CN MCI AD
Outputs (in <root>/summary/): main_table.csv/.md, robustness_table.md, ablation_table.md,
explain_summary.md, stability.json, top_rois.csv.
"""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from mrxgnn.metrics import all_metrics, bootstrap_ci, paired_bootstrap_delta, paired_bootstrap_drop

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--classes", nargs="+", default=["CN", "MCI", "AD"])
ap.add_argument("--reference", default="full")
ap.add_argument("--B", type=int, default=1000)
a = ap.parse_args()
C = a.classes
pc = [f"p_{c}" for c in C]
out = os.path.join(a.root, "summary"); os.makedirs(out, exist_ok=True)
variants = sorted(d for d in os.listdir(a.root) if os.path.exists(os.path.join(a.root, d, "predictions.csv")))


def fmt(m, s):
    return f"{100*m:.1f} ± {100*s:.1f}"


def seed_stack(Q):
    """-> ids (sorted), y, probs [S, n, K], seeds. Only subjects present in every seed."""
    seeds = sorted(Q.seed.unique())
    ids = sorted(set.intersection(*[set(Q[Q.seed == s_].subject_id) for s_ in seeds]))
    probs = np.stack([Q[Q.seed == s_].drop_duplicates("subject_id").set_index("subject_id").loc[ids, pc].values for s_ in seeds])
    y = Q[Q.seed == seeds[0]].drop_duplicates("subject_id").set_index("subject_id").loc[ids, "y"].values
    return np.array(ids), y, probs, seeds


def bacc_fast(p, y):
    pred = p.argmax(-1)
    return np.mean([(pred[y == c] == c).mean() for c in range(len(C)) if (y == c).any()])


def seed_mean(probs, y, i, fn=bacc_fast):
    return float(np.mean([fn(pr[i], y[i]) for pr in probs]))


def auc_fn(p, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p[:, 1]) if len(C) == 2 else roc_auc_score(y, p, multi_class="ovr", average="macro")


# Point estimates: mean ± SD over repeats of per-repeat metrics.
# CIs and paired tests: subjects resampled with replacement; each replicate recomputes the SAME
# estimand (mean over repeats of the per-repeat metric), so CIs are centred on the reported means.
rng_boot = np.random.default_rng(0)
rows, cache = [], {}
for v in variants:
    P = pd.read_csv(os.path.join(a.root, v, "predictions.csv"))
    for sc in P.scenario.unique():
        Ps = P[P.scenario == sc]
        subsets = {"all": Ps}
        if sc == "observed":
            subsets["complete_MRI+PET"] = Ps[Ps.m_pet == 1]
            subsets["PET_missing"] = Ps[Ps.m_pet == 0]
        for sub, Q in subsets.items():
            if len(Q) == 0 or Q.y.nunique() < len(C):
                continue
            per_seed = [all_metrics(q[pc].values, q.y.values, C) for _, q in Q.groupby("seed")]
            ids, y, probs, seeds = seed_stack(Q)
            cache[(v, sc, sub)] = (ids, y, probs, seeds)
            r = {"variant": v, "scenario": sc, "subset": sub, "n": len(y)}
            for k in ["bacc", "auc", "macro_f1", "acc", "ece", "macro_prec", "macro_spec"] + [f"{m}_{c}" for m in ("sens", "spec", "prec") for c in C]:
                vals = np.array([m[k] for m in per_seed])
                r[k + "_mean"], r[k + "_sd"] = vals.mean(), vals.std()
            bs_b, bs_a = [], []
            for _ in range(a.B):
                i = rng_boot.integers(0, len(y), len(y))
                if len(np.unique(y[i])) < len(C):
                    continue
                bs_b.append(seed_mean(probs, y, i))
                if sub == "all" and sc in ("observed", "drop_pet"):
                    bs_a.append(seed_mean(probs, y, i, auc_fn))
            r["bacc_ci"] = f"[{100*np.percentile(bs_b, 2.5):.1f}, {100*np.percentile(bs_b, 97.5):.1f}]"
            if bs_a:
                r["auc_ci"] = f"[{100*np.percentile(bs_a, 2.5):.1f}, {100*np.percentile(bs_a, 97.5):.1f}]"
            rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(out, "main_table.csv"), index=False)


def md(df, cols):
    return df[cols].to_markdown(index=False) if len(df) else "(no rows)"


# Main table: observed scenario, all subjects
M = T[(T.scenario == "observed") & (T.subset == "all")].copy()
for k in ["bacc", "auc", "macro_f1", "ece"]:
    M[k] = [fmt(m, s) for m, s in zip(M[k + "_mean"], M[k + "_sd"])]
open(os.path.join(out, "main_table.md"), "w").write(
    "Mean ± SD over CV repeats; 95% subject-bootstrap CI of the same estimand.\n\n"
    + md(M, [c for c in ["variant", "n", "bacc", "bacc_ci", "auc", "auc_ci", "macro_f1", "ece"] if c in M]))

# Robustness table: BACC per scenario / subset
R = T.copy(); R["bacc"] = [fmt(m, s) for m, s in zip(R.bacc_mean, R.bacc_sd)]
R["col"] = R.scenario + ":" + R.subset
open(os.path.join(out, "robustness_table.md"), "w").write(
    "Balanced accuracy (%) under each missing-modality scenario.\n\n"
    + R.pivot(index="variant", columns="col", values="bacc").reset_index().to_markdown(index=False))

def paired_seed_bootstrap(groupA, groupB, stat):
    """Each group is a list of cache entries (ids, y, probs, seeds). Subjects common to all entries
    and seeds common to all entries are used; the metric per entry is the mean over seeds of
    per-seed balanced accuracy; stat(listA, listB) combines them. Returns (d, (lo, hi), p)."""
    entries = groupA + groupB
    ids = entries[0][0]
    for e in entries[1:]:
        ids = np.intersect1d(ids, e[0])
    seeds = sorted(set.intersection(*[set(e[3]) for e in entries]))
    aligned = []
    for e in entries:
        pos = np.searchsorted(e[0], ids)
        sidx = [e[3].index(s_) for s_ in seeds]
        aligned.append(e[2][sidx][:, pos])
    y = entries[0][1][np.searchsorted(entries[0][0], ids)]
    nA = len(groupA)
    f = lambda i: stat([seed_mean(pr, y, i) for pr in aligned[:nA]], [seed_mean(pr, y, i) for pr in aligned[nA:]])
    d0 = f(np.arange(len(y))); ds = []
    for _ in range(a.B):
        i = rng_boot.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == len(C):
            ds.append(f(i))
    ds = np.array(ds)
    return float(d0), (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))), float(min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean())))


# Ablation: paired bootstrap vs reference
ab = []
for v in variants:
    if v == a.reference:
        continue
    for sc in ("observed", "drop_pet"):
        kr, kv = (a.reference, sc, "all"), (v, sc, "all")
        if kr in cache and kv in cache:
            d, ci, p = paired_seed_bootstrap([cache[kr]], [cache[kv]], lambda A_, B_: A_[0] - B_[0])
            ab.append({"variant": v, "scenario": sc, "delta_bacc_ref_minus_variant": f"{100*d:+.1f}",
                       "ci": f"[{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]", "p": f"{p:.3f}",
                       "_d": d, "_lo": ci[0], "_hi": ci[1], "_p": p})
if ab:
    A = pd.DataFrame(ab)
    # Holm correction within each scenario, over all variants compared with the reference
    for sc, g in A.groupby("scenario"):
        order = g._p.sort_values().index
        m = len(order); run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * A.loc[i, "_p"]))
            A.loc[i, "p_holm"] = run
    A.to_csv(os.path.join(out, "ablation.csv"), index=False)
    ab = [{k: v for k, v in r.items() if not k.startswith("_")} | {"p_holm": f"{r['p_holm']:.3f}"} for r in A.to_dict("records")]
open(os.path.join(out, "ablation_table.md"), "w").write(
    f"Paired bootstrap: BACC({a.reference}) - BACC(variant); positive = reference better.\n"
    "Correct for multiple comparisons (Holm) when reporting.\n\n" + (pd.DataFrame(ab).to_markdown(index=False) if ab else ""))

# H2: robustness to PET withdrawal, reference vs comparators (difference in drops)
h2 = []
for v in ("no_halluc", "no_moddrop", "lr_concat", "svm_rbf_concat", "hgb_concat"):
    keys = [(a.reference, "observed", "all"), (a.reference, "drop_pet", "all"), (v, "observed", "all"), (v, "drop_pet", "all")]
    if all(k in cache for k in keys):
        # (drop of comparator) - (drop of reference); positive = reference degrades less
        d, ci, p = paired_seed_bootstrap([cache[keys[0]], cache[keys[1]]], [cache[keys[2]], cache[keys[3]]],
                                         lambda A_, B_: (B_[0] - B_[1]) - (A_[0] - A_[1]))
        h2.append({"comparator": v, "drop_comparator_minus_drop_ref_pp": round(100 * d, 2),
                   "ci_lo_pp": round(100 * ci[0], 2), "ci_hi_pp": round(100 * ci[1], 2), "p": round(p, 4)})
if h2:
    pd.DataFrame(h2).to_csv(os.path.join(out, "h2_robustness.csv"), index=False)

# Explanations of the reference model
ej = os.path.join(a.root, a.reference, "explain.json")
if os.path.exists(ej):
    E = json.load(open(ej))["explanations"]
    lines = ["## Faithfulness (deletion test): mean drop in true-class probability", "",
             "| k | top-k attributed | random k |", "|---|---|---|"]
    ks = E[0]["deletion"]["k"]
    for i, k in enumerate(ks):
        t = [e["deletion"]["drop_top"][i] for e in E]; r = [e["deletion"]["drop_random"][i] for e in E]
        lines.append(f"| {k} | {np.mean(t):.3f} ± {np.std(t):.3f} | {np.mean(r):.3f} ± {np.std(r):.3f} |")
    lines += ["", "## Anatomical plausibility (AD-signature ROIs, disease-class subjects)", ""]
    for key in E[0]["plausibility"]:
        vals = [e["plausibility"][key] for e in E]
        lines.append(f"- {key}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    ra = pd.DataFrame([r for e in E for r in e["rule_ablation"] if r.get("n_active", 0) > 0])
    if len(ra):
        lines += ["", "## Rule ablation (subjects where the rule is active)", "",
                  ra.groupby("rule")[["n_active", "mean_true_prob_drop", "bacc_with", "bacc_without"]].mean().round(3).to_markdown()]
    rl = os.path.join(a.root, a.reference, "rules_learned.csv")
    if os.path.exists(rl):
        L = pd.read_csv(rl)
        lines += ["", "## Learned vs guideline thresholds (mean ± SD over folds)", "",
                  L.groupby("rule").agg(t1_guideline=("t1_guideline", "first"), t1_learned=("t1_learned", "mean"),
                                        t1_sd=("t1_learned", "std")).round(4).to_markdown()]
    na = os.path.join(a.root, "no_anchor", "rules_learned.csv")
    if os.path.exists(rl) and os.path.exists(na):
        dev = lambda D: (100 * (D.t1_learned - D.t1_guideline).abs() / D.t1_guideline.abs().clip(lower=1e-9)).groupby(D.rule).mean()
        cmp_ = pd.DataFrame({"with_anchoring_%dev": dev(pd.read_csv(rl)), "without_anchoring_%dev": dev(pd.read_csv(na))}).round(2)
        lines += ["", "## Threshold drift from guideline (% of guideline value), with vs without anchoring", "", cmp_.to_markdown()]
    open(os.path.join(out, "explain_summary.md"), "w").write("\n".join(lines))
    EJ = {"deletion": {str(k): {"top_mean": float(np.mean([e["deletion"]["drop_top"][i] for e in E])),
                                "top_sd": float(np.std([e["deletion"]["drop_top"][i] for e in E])),
                                "random_mean": float(np.mean([e["deletion"]["drop_random"][i] for e in E])),
                                "random_sd": float(np.std([e["deletion"]["drop_random"][i] for e in E]))}
                       for i, k in enumerate(ks)},
          "h3_fraction_runs_top8_ge_2x_random": float(np.mean([e["deletion"]["drop_top"][ks.index(8)] >= 2 * e["deletion"]["drop_random"][ks.index(8)] for e in E])) if 8 in ks else None,
          "h4_auroc_runs": [e["plausibility"]["auroc_mean"] for e in E],
          "plausibility": {key: {"mean": float(np.mean([e["plausibility"][key] for e in E])),
                                 "sd": float(np.std([e["plausibility"][key] for e in E]))} for key in E[0]["plausibility"]}}
    if len(ra):
        EJ["rule_ablation"] = ra.groupby("rule")[["n_active", "mean_true_prob_drop", "bacc_with", "bacc_without"]].mean().to_dict("index")
    if os.path.exists(rl):
        L = pd.read_csv(rl)
        EJ["rules_learned"] = L.groupby("rule").agg(t1_guideline=("t1_guideline", "first"), t1_learned=("t1_learned", "mean"),
                                                    t1_sd=("t1_learned", "std")).to_dict("index")
    if os.path.exists(rl) and os.path.exists(na):
        EJ["threshold_drift_pct"] = cmp_.to_dict("index")
    json.dump(EJ, open(os.path.join(out, "explain_summary.json"), "w"), indent=1)

    I = pd.read_csv(os.path.join(a.root, a.reference, "roi_importance.csv"))
    rois = [c for c in I.columns if c not in ("subject_id", "y", "seed", "fold")]
    ad = C.index("AD") if "AD" in C else len(C) - 1
    G = I[I.y == ad].groupby(["seed", "fold"])[rois].mean()
    rhos = [spearmanr(G.iloc[i], G.iloc[j])[0] for i, j in itertools.combinations(range(len(G)), 2)]
    json.dump({"spearman_mean": float(np.mean(rhos)), "spearman_sd": float(np.std(rhos)), "pairs": len(rhos)},
              open(os.path.join(out, "stability.json"), "w"), indent=1)
    top = G.mean().sort_values(ascending=False)
    top.rename("mean_abs_IG").to_csv(os.path.join(out, "top_rois.csv"))
print(open(os.path.join(out, "main_table.md")).read())
print(open(os.path.join(out, "robustness_table.md")).read())
