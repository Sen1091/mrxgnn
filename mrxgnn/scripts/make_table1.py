"""Table 1 (cohort characteristics) and flow counts from the generic input files.
  python scripts/make_table1.py --data data/adni_processed --out results/adni3/summary/table1 [--prefix adni]
Writes <out>.md and <out>.json. Group tests: one-way ANOVA (continuous) and chi-square (categorical).
"""
import argparse, json, os
import numpy as np, pandas as pd
from scipy.stats import f_oneway, chi2_contingency

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--classes", nargs="+", default=["CN", "MCI", "AD"])
a = ap.parse_args()
d = a.data
lab = pd.read_csv(os.path.join(d, "labels.csv"))
mri = pd.read_csv(os.path.join(d, "mri.csv")); pet = pd.read_csv(os.path.join(d, "pet.csv"))
cli = pd.read_csv(os.path.join(d, "clinical.csv"))
df = lab[lab.dx.isin(a.classes) & lab.subject_id.isin(mri.subject_id)].merge(cli, on="subject_id", how="left")
df["has_pet"] = df.subject_id.isin(pet.subject_id)
df["has_csf"] = df["abeta"].notna() if "abeta" in df else False
df["apoe_carrier"] = df["apoe4"] >= 1
G = [df[df.dx == c] for c in a.classes]
J, rows = {"N": int(len(df))}, []


def cont(name, col):
    if col not in df or df[col].notna().sum() == 0:
        return
    vals = [g[col].dropna() for g in G]
    cells = [f"{v.mean():.1f} ± {v.std():.1f}" for v in vals]
    p = f_oneway(*[v for v in vals if len(v) > 1]).pvalue if sum(len(v) > 1 for v in vals) > 1 else np.nan
    rows.append([name] + cells + [f"{p:.3g}"]); J[name] = dict(zip(a.classes, cells)) | {"p": p}


def cat(name, mask_col):
    cnt = [int(g[mask_col].sum()) for g in G]; tot = [len(g) for g in G]
    cells = [f"{c} ({100*c/max(t,1):.0f}%)" for c, t in zip(cnt, tot)]
    tab = np.array([cnt, [t - c for c, t in zip(cnt, tot)]])
    p = chi2_contingency(tab)[1] if (tab.sum(1) > 0).all() else np.nan
    rows.append([name] + cells + [f"{p:.3g}"]); J[name] = dict(zip(a.classes, cells)) | {"p": p}


rows.append(["N"] + [str(len(g)) for g in G] + [""]); J["n"] = {c: len(g) for c, g in zip(a.classes, G)}
cont("Age (years)", "age")
df["female"] = df["sex"] == 0; G = [df[df.dx == c] for c in a.classes]
cat("Female", "female"); cont("Education (years)", "educ"); cat("APOE ε4 carriers", "apoe_carrier")
cont("MMSE (not a model input)", "mmse"); cont("CDR (not a model input)", "cdr")
cat("PET available", "has_pet"); cat("CSF available", "has_csf")
J["pct_without_pet"] = float(100 * (1 - df.has_pet.mean()))
T = pd.DataFrame(rows, columns=["Characteristic"] + a.classes + ["p"])
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
open(a.out + ".md", "w").write(T.to_markdown(index=False))
json.dump(J, open(a.out + ".json", "w"), indent=1, default=float)
print(T.to_markdown(index=False))
