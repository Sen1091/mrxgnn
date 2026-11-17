"""Build labels / mri / pet / clinical CSVs from the ADNIMERGE2 R package (CDISC-style, 2025+)
plus the UCSF FreeSurfer 7 (UCSFFSX7), UC Berkeley amyloid 6-mm and UPenn Roche Elecsys tables.

Self-contained (needs pandas, numpy, pyreadr). Prints only aggregate counts, never participant rows.

  python scripts/prepare_adni2.py --raw ../ADNI_raw --out ../adni_processed

Implements the registered eligibility rules (OSF 10.17605/OSF.IO/J4KG6) plus deviations C1-C6:
  - phases ADNI1/GO/2/3 (ADSL.ORIGPROT); baseline = DXSUM VISCODE2 'bl' (fallback 'sc');
    enrolled only (ADSL.ENRLFL); SMC participants excluded (ARM table); Dementia -> AD
  - MRI: UCSFFSX7 session closest to baseline within --window-days, OVERALLQC Fail / Hippocampus
    Only excluded (blank = not failed); Desikan thickness (TA) + aseg volumes (SV), mapped by DATADIC text
  - PET: Berkeley 6-mm amyloid, FBP/FBB, qc_flag 0 (fail) / -2 (not processable) excluded, closest scan
    within --window-days; regional SUVR (whole-cerebellum reference); tracer kept for per-tracer W-scores
  - CSF: Elecsys closest within --window-days; censored values ('>1700', '<200') set to the bound
"""
import argparse, glob, os, re, sys, tarfile, tempfile
import numpy as np, pandas as pd

DK = ["bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus", "entorhinal",
      "fusiform", "inferiorparietal", "inferiortemporal", "isthmuscingulate", "lateraloccipital",
      "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal",
      "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
      "postcentral", "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate",
      "rostralmiddlefrontal", "superiorfrontal", "superiorparietal", "superiortemporal",
      "supramarginal", "frontalpole", "temporalpole", "transversetemporal", "insula"]
SUB = ["hippocampus", "amygdala", "thalamus", "caudate", "putamen", "pallidum", "accumbens"]
ROIS = [f"{h}_{r}" for h in ("lh", "rh") for r in DK + SUB]
FS_SUB = {"accumbens": "accumbensarea"}                       # aseg name differences
BK_SUB = {"accumbens": "ACCUMBENS_AREA", "thalamus": "THALAMUS_PROPER"}

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True, help="folder with ADNIMERGE2*.tar.gz and the three CSVs")
ap.add_argument("--out", required=True)
ap.add_argument("--phases", default="ADNI1,ADNIGO,ADNI2,ADNI3")
ap.add_argument("--window-days", type=int, default=180)
ap.add_argument("--qc-fail-values", default="Fail,Hippocampus Only")
ap.add_argument("--pet-tracers", default="FBP,FBB")
ap.add_argument("--pet-qc-exclude", default="0,-2")
ap.add_argument("--include-smc", action="store_true")
a = ap.parse_args()
R = a.raw
one = lambda pat: (glob.glob(os.path.join(R, pat)) or sys.exit(f"not found in {R}: {pat}"))[0]
log = []
def say(*s):
    msg = " ".join(str(x) for x in s); print(msg); log.append(msg)


# ---------- ADNIMERGE2 tables ----------
import pyreadr
def rda_tables(pkg, names):
    out = {}
    with tarfile.open(pkg) as tf, tempfile.TemporaryDirectory() as tmp:
        for n in names:
            m = [x for x in tf.getmembers() if x.name.endswith(f"/data/{n}.rda")]
            if not m:
                sys.exit(f"{n}.rda not in {pkg}")
            m = m[0]; m.name = f"{n}.rda"; tf.extract(m, tmp, filter="data")
            df = next(iter(pyreadr.read_r(os.path.join(tmp, m.name)).values()))
            for c in df.columns:
                if str(df[c].dtype) == "category":
                    df[c] = df[c].astype(str).replace({"nan": np.nan})
            out[n] = df
    return out

T = rda_tables(one("ADNIMERGE2*.tar.gz"), ["ADSL", "DXSUM", "ARM", "DATADIC"])
adsl, dxs, arm, dd = T["ADSL"], T["DXSUM"], T["ARM"], T["DATADIC"]
adsl["RID"] = pd.to_numeric(adsl.SUBJID, errors="coerce")
phases = [p.strip() for p in a.phases.split(",")]
say("ADSL subjects:", len(adsl), "| in phases", phases, ":", int(adsl.ORIGPROT.isin(phases).sum()))
s = adsl[adsl.ORIGPROT.isin(phases)].drop_duplicates("RID").copy()
n0 = len(s); s = s[s.ENRLFL.astype(str) == "Y"]
say("excluded: not enrolled (screen failures, ADSL.ENRLFL != Y):", n0 - len(s))

dxs["RID"] = pd.to_numeric(dxs.RID, errors="coerce")
dxs["EXAMDATE"] = pd.to_datetime(dxs.EXAMDATE, errors="coerce")
bl = dxs[dxs.VISCODE2 == "bl"].sort_values("EXAMDATE").drop_duplicates("RID")
sc = dxs[(dxs.VISCODE2 == "sc") & ~dxs.RID.isin(bl.RID)].sort_values("EXAMDATE").drop_duplicates("RID")
base = pd.concat([bl, sc])[["RID", "PTID", "EXAMDATE", "DIAGNOSIS"]]
say("baseline diagnosis from DXSUM: bl", len(bl), "| sc fallback", len(sc))
s = s.merge(base, on="RID", how="left")
s["dx"] = s.DIAGNOSIS.map({"CN": "CN", "MCI": "MCI", "Dementia": "AD"})
agree = s.dropna(subset=["dx", "DX"])
say("agreement DXSUM-baseline vs ADSL.DX:", f"{(agree.dx == agree.DX.replace({'DEM': 'AD'})).mean():.3f}", f"(n={len(agree)})")
n0 = len(s); s = s[s.dx.notna() & s.EXAMDATE.notna()]
say("excluded: no baseline diagnosis/date:", n0 - len(s))
arm["RID"] = pd.to_numeric(arm.RID, errors="coerce")
smc = set(arm.loc[arm.ARM.astype(str).str.contains("SMC"), "RID"])
if not a.include_smc:
    n0 = len(s); s = s[~s.RID.isin(smc)]; say("excluded: SMC enrolment arm:", n0 - len(s))
s["subject_id"] = s.PTID.fillna(s.USUBJID.str.replace("ADNI-", "", regex=False))
s["BL_DATE"] = s.EXAMDATE
ref = s[["RID", "BL_DATE"]]
say("eligible by diagnosis:", s.dx.value_counts().to_dict(), "| by phase:", s.ORIGPROT.value_counts().to_dict())


def closest(df, date_col, rid_col="RID"):
    d = df.copy()
    d[rid_col] = pd.to_numeric(d[rid_col], errors="coerce")
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.merge(ref, left_on=rid_col, right_on="RID", how="inner", suffixes=("", "_ref"))
    d["gap"] = (d[date_col] - d.BL_DATE).dt.days.abs()
    return d[d.gap <= a.window_days].sort_values(["RID", "gap"])


# ---------- MRI ----------
fsdd = dd[dd.TBLNAME.astype(str).str.upper() == "UCSFFSX7"][["FLDNAME", "TEXT"]].astype(str)
def fs_col(roi):
    h, r = roi.split("_", 1); side = "Left" if h == "lh" else "Right"
    if r in SUB:
        name, suf, kind = FS_SUB.get(r, r), "SV", "Subcortical Volume"
    else:
        name, suf, kind = r, "TA", "Thickness Average"
    hit = fsdd[fsdd.FLDNAME.str.endswith(suf) & fsdd.TEXT.str.startswith(kind)
               & (fsdd.TEXT.str.split(" of ").str[-1].str.lower() == (side + name).lower())]
    if len(hit) != 1:
        sys.exit(f"MRI mapping for {roi}: {len(hit)} matches")
    return hit.FLDNAME.iloc[0]
mri_map = {r: fs_col(r) for r in ROIS}
mri = pd.read_csv(one("UCSFFSX7*.csv"), low_memory=False)
fail = {v.strip().lower() for v in a.qc_fail_values.split(",")}
bad = mri.OVERALLQC.astype(str).str.strip().str.lower().isin(fail)
say("MRI rows:", len(mri), "| QC-failed removed:", int(bad.sum()), "| OVERALLQC counts:", mri.OVERALLQC.fillna("blank").value_counts().to_dict())
m = closest(mri[~bad], "EXAMDATE")
m[list(mri_map.values())] = m[list(mri_map.values())].apply(pd.to_numeric, errors="coerce")
m = m.dropna(subset=list(mri_map.values())).drop_duplicates("RID")
say("MRI baseline sessions:", len(m), "| FS version:", m.FSVER.astype(str).value_counts().to_dict(),
    "| field strength:", m.FIELD_STRENGTH.astype(str).value_counts().to_dict(),
    "| median gap days:", int(m.gap.median()) if len(m) else "-")
mri_out = m[["RID"] + list(mri_map.values())].rename(columns={v: k for k, v in mri_map.items()})

# ---------- PET ----------
def bk_col(roi):
    h, r = roi.split("_", 1)
    return (f"CTX_{h.upper()}_{r.upper()}_SUVR" if r in DK else
            f"{'LEFT' if h == 'lh' else 'RIGHT'}_{BK_SUB.get(r, r.upper())}_SUVR")
pet_map = {r: bk_col(r) for r in ROIS}
pet = pd.read_csv(one("UCBERKELEY_AMY_6MM*.csv"), low_memory=False)
miss = [c for c in pet_map.values() if c not in pet.columns]
if miss: sys.exit(f"PET columns missing: {miss[:5]}")
say("PET rows by tracer x qc_flag:", pet.groupby(["TRACER", "qc_flag"]).size().to_dict())
keep_tr = {t.strip().upper() for t in a.pet_tracers.split(",")}
qc_ex = {float(x) for x in a.pet_qc_exclude.split(",")}
p = pet[pet.TRACER.astype(str).str.upper().isin(keep_tr) & ~pd.to_numeric(pet.qc_flag, errors="coerce").isin(qc_ex)]
p = closest(p, "SCANDATE")
p[list(pet_map.values())] = p[list(pet_map.values())].apply(pd.to_numeric, errors="coerce")
p = p.dropna(subset=list(pet_map.values())).drop_duplicates("RID")
say("PET baseline scans:", len(p), "| tracer:", p.TRACER.value_counts().to_dict(),
    "| qc_flag:", p.qc_flag.value_counts().to_dict())
pet_out = p[["RID"] + list(pet_map.values()) + ["TRACER"]].rename(
    columns={**{v: k for k, v in pet_map.items()}, "TRACER": "tracer"})

# ---------- CSF ----------
num = lambda x: pd.to_numeric(x.astype(str).str.replace(r"[<>]", "", regex=True), errors="coerce")
csf = pd.read_csv(one("UPENNBIOMK_ROCHE_ELECSYS*.csv"), low_memory=False)
cens = int(csf.ABETA42.astype(str).str.contains("[<>]").sum())
c = closest(csf, "EXAMDATE")
c["abeta"], c["ptau"] = num(c.ABETA42), num(c.PTAU)
c = c.dropna(subset=["abeta", "ptau"]).drop_duplicates("RID")
say("CSF Elecsys rows:", len(csf), "(Abeta42 censored:", cens, ") | baseline matches:", len(c))

# ---------- assemble ----------
def apoe4(g):
    g = str(g)
    return np.nan if g in ("nan", "", "None") else float(g.count("4"))
clin = pd.DataFrame({"RID": s.RID, "subject_id": s.subject_id, "age": pd.to_numeric(s.AGE, errors="coerce"),
                     "sex": (s.SEX.astype(str) == "Male").astype(int),
                     "educ": pd.to_numeric(s.EDUC, errors="coerce"), "apoe4": s.APOE.map(apoe4),
                     "mmse": pd.to_numeric(s.get("MMSCORE"), errors="coerce")})
clin = clin.merge(c[["RID", "abeta", "ptau"]], on="RID", how="left")
clin["ptau_abeta"] = clin.ptau / clin.abeta
say("APOE e4 counts:", clin.apoe4.value_counts(dropna=False).to_dict())
ids = s[["RID", "subject_id"]]
labels = s[["subject_id", "dx", "SITEID", "ORIGPROT"]].rename(columns={"SITEID": "site", "ORIGPROT": "phase"})
mri_out = ids.merge(mri_out, on="RID")[["subject_id"] + ROIS]
pet_out = ids.merge(pet_out, on="RID")[["subject_id"] + ROIS + ["tracer"]]
os.makedirs(a.out, exist_ok=True)
for n, d in (("labels", labels), ("mri", mri_out), ("pet", pet_out), ("clinical", clin.drop(columns="RID"))):
    d.to_csv(os.path.join(a.out, f"{n}.csv"), index=False)
pd.DataFrame({"roi": ROIS, "mri_column": [mri_map[r] for r in ROIS], "pet_column": [pet_map[r] for r in ROIS]}
             ).to_csv(os.path.join(a.out, "roi_map_used.csv"), index=False)

lab = labels.set_index("subject_id")
has = lambda d: lab.index.isin(d.subject_id)
F = pd.DataFrame({"dx": lab.dx, "mri": has(mri_out), "pet": has(pet_out),
                  "csf": lab.index.isin(clin.subject_id[clin.abeta.notna()])})
say("\n=== FINAL (aggregate) ===")
say("N with MRI (analysis cohort):", int(F.mri.sum()))
say(F[F.mri].groupby("dx")[["pet", "csf"]].agg(["sum", "count"]).to_string())
say("PET tracer in cohort:", pet_out[pet_out.subject_id.isin(F.index[F.mri])].tracer.value_counts().to_dict())
open(os.path.join(a.out, "prepare_log.txt"), "w").write("\n".join(log) + "\n")
