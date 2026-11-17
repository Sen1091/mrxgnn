"""Collect every number the manuscript needs into one JSON (results/paper_numbers.json).
The manuscript builder (ms/build.js) replaces {{key}} placeholders with these values. Missing
keys stay as highlighted placeholders, so no number is ever typed by hand.
  python scripts/export_paper_numbers.py --internal results/adni3 --external results/external_oasis3 \
      --out results/paper_numbers.json
"""
import argparse, json, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--internal", required=True); ap.add_argument("--external")
ap.add_argument("--binary", nargs="*", default=[], help="label=results_dir pairs, e.g. cn_ad=results/adni_cn_ad")
ap.add_argument("--out", default="results/paper_numbers.json")
a = ap.parse_args()
S = os.path.join(a.internal, "summary")
N = {}
pfmt = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
f1 = lambda m, s: f"{100*m:.1f} ± {100*s:.1f}"
NAMES = {"lr_concat": "logistic regression", "svm_rbf_concat": "RBF-SVM", "hgb_concat": "gradient boosting"}


def table1(path, prefix):
    if not os.path.exists(path):
        return
    J = json.load(open(path))
    N[f"{prefix}.N"] = str(J["N"])
    for c, n in J["n"].items():
        N[f"{prefix}.n.{c}"] = str(n)
    N[f"{prefix}.pct_no_pet"] = f"{J['pct_without_pet']:.0f}"
    for row, v in J.items():
        if isinstance(v, dict) and "p" in v:
            for c, cell in v.items():
                N[f"{prefix}.{row}.{c}"] = cell if c != "p" else (f"{cell:.3g}" if cell == cell else "—")


table1(os.path.join(S, "table1.json"), "t1")
M = pd.read_csv(os.path.join(S, "main_table.csv"))
for _, r in M.iterrows():
    if r.scenario == "observed" and r.subset == "all":
        for k in ("bacc", "auc", "macro_f1", "ece"):
            N[f"main.{r.variant}.{k}"] = f1(r[k + "_mean"], r[k + "_sd"])
        for k in [c[:-5] for c in r.index if c.endswith("_mean") and (c.split("_")[0] in ("sens", "spec", "prec") or c[:-5] in ("macro_prec", "macro_spec"))]:
            if r[k + "_mean"] == r[k + "_mean"]:
                N[f"main.{r.variant}.{k}"] = f"{100*r[k + '_mean']:.1f}"
        for k in ("bacc", "auc"):
            if isinstance(r.get(k + "_ci"), str):
                N[f"main.{r.variant}.{k}_ci"] = r[k + "_ci"].strip("[]").replace(", ", "–")
    N[f"rob.{r.variant}.{r.scenario}.{r.subset}"] = f1(r.bacc_mean, r.bacc_sd)
    N[f"_rob.{r.variant}.{r.scenario}.{r.subset}"] = float(r.bacc_mean)

obs = M[(M.scenario == "observed") & (M.subset == "all")].set_index("variant")
base = [v for v in NAMES if v in obs.index]
if base:
    b = obs.loc[base, "bacc_mean"].idxmax()
    N["best_baseline.name"] = NAMES[b]; N["best_baseline.bacc"] = f1(obs.loc[b, "bacc_mean"], obs.loc[b, "bacc_sd"])
    dp = M[(M.scenario == "drop_pet") & (M.subset == "all")].set_index("variant")
    if b in dp.index:
        N["best_baseline.drop_pet"] = f"{100*dp.loc[b, 'bacc_mean']:.1f}"
    for v in ("full", "no_halluc", b):
        if v in dp.index and v in obs.index:
            N[f"delta_drop_pet.{v}"] = f"{100*(dp.loc[v,'bacc_mean']-obs.loc[v,'bacc_mean']):+.1f}"
    N["delta_drop_pet.best_baseline"] = N.get(f"delta_drop_pet.{b}", "")
if "full" in obs.index and base:
    N["full_minus_best_baseline"] = f"{100*(obs.loc['full','bacc_mean']-obs.loc[b,'bacc_mean']):+.1f}"

ab = os.path.join(S, "ablation.csv")
if os.path.exists(ab):
    for _, r in pd.read_csv(ab).iterrows():
        N[f"abl.{r.variant}.{r.scenario}.d"] = f"{100*r._d:+.1f}"
        N[f"abl.{r.variant}.{r.scenario}.p"] = pfmt(r.p_holm)
        N[f"abl.{r.variant}.{r.scenario}.ci"] = f"{100*r._lo:+.1f} to {100*r._hi:+.1f}"
        if base and r.variant == b and r.scenario == "observed":   # H1 non-inferiority (margin -2 pp)
            N["h1.d"], N["h1.ci"] = f"{100*r._d:+.1f}", f"{100*r._lo:+.1f} to {100*r._hi:+.1f}"
            N["h1.verdict"] = ("superior" if r._lo > 0 else "non-inferior" if r._lo > -0.02 else "not shown to be non-inferior")

ej = os.path.join(S, "explain_summary.json")
if os.path.exists(ej):
    E = json.load(open(ej))
    for k, v in E["deletion"].items():
        N[f"expl.del.{k}.top"] = f"{v['top_mean']:.3f} ± {v['top_sd']:.3f}"
        N[f"expl.del.{k}.rand"] = f"{v['random_mean']:.3f} ± {v['random_sd']:.3f}"
        N[f"expl.del.{k}.ratio"] = f"{v['top_mean']/max(v['random_mean'],1e-9):.1f}"
    P = E["plausibility"]
    N["expl.prec10"] = f"{100*P['precision@10']['mean']:.0f}"; N["expl.chance"] = f"{100*P['chance_precision']['mean']:.0f}"
    N["expl.auroc"] = f"{P['auroc_mean']['mean']:.2f} ± {P['auroc_mean']['sd']:.2f}"
    for rule, v in E.get("rule_ablation", {}).items():
        N[f"rule.{rule}.n_active"] = f"{v['n_active']:.0f}"; N[f"rule.{rule}.dprob"] = f"{v['mean_true_prob_drop']:+.3f}"
    for rule, v in E.get("rules_learned", {}).items():
        N[f"rule.{rule}.t1_learned"] = f"{v['t1_learned']:.4g} ± {v['t1_sd']:.2g}"
    dr = E.get("threshold_drift_pct")
    if dr:
        N["rule.max_drift_pct"] = f"{max(x['with_anchoring_%dev'] for x in dr.values()):.1f}"
h2 = os.path.join(S, "h2_robustness.csv")
if os.path.exists(h2):
    for _, r in pd.read_csv(h2).iterrows():
        N[f"h2.{r.comparator}.d"] = f"{r.drop_comparator_minus_drop_ref_pp:+.1f}"
        N[f"h2.{r.comparator}.ci"] = f"{r.ci_lo_pp:+.1f} to {r.ci_hi_pp:+.1f}"
        N[f"h2.{r.comparator}.p"] = pfmt(r.p)
if os.path.exists(ej) and E.get("h3_fraction_runs_top8_ge_2x_random") is not None:
    N["h3.frac"] = f"{100*E['h3_fraction_runs_top8_ge_2x_random']:.0f}"
st = os.path.join(S, "stability.json")
if os.path.exists(st):
    J = json.load(open(st)); N["expl.stab"] = f"{J['spearman_mean']:.2f} ± {J['spearman_sd']:.2f}"
tr = os.path.join(S, "top_rois.csv")
if os.path.exists(tr):
    t = pd.read_csv(tr).iloc[:5, 0].tolist()
    N["expl.top_rois"] = ", ".join(x.replace("lh_", "left ").replace("rh_", "right ") for x in t)

if a.external:
    ES = os.path.join(a.external, "summary")
    table1(os.path.join(ES, "table1.json"), "et1")
    et = os.path.join(ES, "external_table.csv")
    if os.path.exists(et):
        for _, r in pd.read_csv(et).iterrows():
            key = f"ext.{r.method}.{r.normalisation}.{r.scenario}"
            N[key + ".auc"] = r.auc_cn_vs_impaired; N[key + ".auc_ci"] = r.auc_ci.strip("[]").replace(", ", "–")
            N[key + ".bacc"] = r.bacc_3class

    h5 = os.path.join(ES, "h5_recalibration.json")
    if os.path.exists(h5):
        J = json.load(open(h5))
        N["h5.d"] = f"{J['delta_auc_recal_minus_norecal']:+.2f}"; N["h5.ci"] = f"{J['ci'][0]:+.2f} to {J['ci'][1]:+.2f}"
        N["h5.p"] = f"{J['p']:.3f}"
    ex = os.path.join(a.external, "external_explain.json")
    if os.path.exists(ex):
        XE = json.load(open(ex))["explanations"]
        if XE:
            N["ext.expl.auroc"] = f"{np.mean([e['plausibility']['auroc_mean'] for e in XE]):.2f}"
            i8 = XE[0]["deletion"]["k"].index(8)
            N["ext.expl.del8.ratio"] = f"{np.mean([e['deletion']['drop_top'][i8] for e in XE]) / max(np.mean([e['deletion']['drop_random'][i8] for e in XE]), 1e-9):.1f}"

for pair in a.binary:                      # secondary binary tasks (Online Resource S3)
    lab, root = pair.split("=", 1)
    bm = os.path.join(root, "summary", "main_table.csv")
    if os.path.exists(bm):
        for _, r in pd.read_csv(bm).iterrows():
            if r.scenario == "observed" and r.subset == "all":
                N[f"bin.{lab}.{r.variant}.bacc"] = f1(r.bacc_mean, r.bacc_sd)
                N[f"bin.{lab}.{r.variant}.auc"] = f1(r.auc_mean, r.auc_sd)
                N[f"bin.{lab}.{r.variant}.ece"] = f1(r.ece_mean, r.ece_sd)
                if isinstance(r.get("bacc_ci"), str):
                    N[f"bin.{lab}.{r.variant}.bacc_ci"] = r.bacc_ci.strip("[]").replace(", ", "–")
            if r.variant == "full" and r.subset == "all":
                N[f"bin.{lab}.full.{r.scenario}"] = f1(r.bacc_mean, r.bacc_sd)
        N[f"bin.{lab}.n"] = str(int(pd.read_csv(bm).n.max()))
    ba = os.path.join(root, "summary", "ablation.csv")
    if os.path.exists(ba):
        for _, r in pd.read_csv(ba).iterrows():
            N[f"bin.{lab}.d.{r.variant}.{r.scenario}"] = f"{100*r._d:+.1f}"
            N[f"bin.{lab}.ci.{r.variant}.{r.scenario}"] = f"{100*r._lo:+.1f} to {100*r._hi:+.1f}"
            N[f"bin.{lab}.p.{r.variant}.{r.scenario}"] = pfmt(r.p_holm)

envf = os.path.join(os.path.dirname(a.out) or ".", "environment.json")
if os.path.exists(envf):
    for k, v in json.load(open(envf)).items():
        N[f"env.{k}"] = v

N = {k: v for k, v in N.items() if not k.startswith("_")}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(N, open(a.out, "w"), indent=1, ensure_ascii=False)
print(f"{len(N)} numbers written to {a.out}")
