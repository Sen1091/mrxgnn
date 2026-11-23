"""Build the generic input CSVs (labels / mri / pet / clinical) from ADNI study tables.

No image processing is needed: ADNI distributes regional FreeSurfer measures (UCSF) and regional
PET SUVRs (UC Berkeley). Download these from adni.loni.usc.edu (Study Files) after the Data Use
Agreement is approved:
  - ADNIMERGE.csv                     diagnosis, demographics, APOE4, CSF (baseline columns *_bl)
  - a UCSF FreeSurfer regional table  (cortical thickness / subcortical volume per ROI)
  - a UC Berkeley PET regional table  (FDG or amyloid; regional SUVR per ROI)

ADNI column names change between table versions. This script therefore needs an explicit ROI
mapping file (configs/roi_map.csv: roi, mri_column, pet_column), filled in from the tables' data
dictionaries. `--suggest` prints best-guess matches to start from; CHECK EVERY LINE by hand.

Usage:
  python scripts/prepare_adni.py --adnimerge ADNIMERGE.csv --mri UCSFFSX.csv --pet UCBERKELEY_PET.csv \
      --roi-map configs/roi_map.csv --out data/adni_processed [--include-smc] [--window-days 180]
  python scripts/prepare_adni.py --mri UCSFFSX.csv --pet UCBERKELEY_PET.csv --suggest
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from mrxgnn.atlas import ROIS

DX_MAP = {"CN": "CN", "SMC": "CN", "EMCI": "MCI", "LMCI": "MCI", "MCI": "MCI", "AD": "AD", "Dementia": "AD"}


def to_num(s):
    """ADNI CSF values may be strings such as '>1700' or '<200'."""
    return pd.to_numeric(s.astype(str).str.replace(r"[<>]", "", regex=True), errors="coerce")


def suggest(cols, roi):
    h, r = roi.split("_", 1)
    side = {"lh": ["LH", "LEFT", "L_"], "rh": ["RH", "RIGHT", "R_"]}[h]
    cand = [c for c in cols if r.upper() in c.upper().replace("-", "").replace("_", "")
            and any(s in c.upper() for s in side)]
    return cand[:3]


def pick_visit(df, rid_col, date_col, ref_dates, window):
    df = df.copy()
    if "VISCODE2" in df and window is None:
        return df[df.VISCODE2.isin(["bl", "sc", "scmri"])].drop_duplicates(rid_col)
    if "VISCODE" in df and window is None:
        return df[df.VISCODE.isin(["bl", "sc"])].drop_duplicates(rid_col)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.merge(ref_dates, left_on=rid_col, right_on="RID", how="inner")
    df["gap"] = (df[date_col] - df["BL_DATE"]).dt.days.abs()
    df = df[df.gap <= (window or 180)].sort_values("gap")
    return df.drop_duplicates(rid_col)


ap = argparse.ArgumentParser()
ap.add_argument("--adnimerge"); ap.add_argument("--mri", required=True); ap.add_argument("--pet", required=True)
ap.add_argument("--roi-map"); ap.add_argument("--out", default="data/adni_processed")
ap.add_argument("--include-smc", action="store_true", help="map SMC to CN (default: exclude SMC)")
ap.add_argument("--window-days", type=int, default=None, help="match scans to baseline by date instead of VISCODE")
ap.add_argument("--suggest", action="store_true")
ap.add_argument("--mri-qc-column", default="OVERALLQC",
                help="UCSF FreeSurfer overall QC column (pre-registered rule: failing scans are excluded); '' to disable")
ap.add_argument("--qc-fail-values", default="Fail,Hippocampus Only",
                help="comma-separated QC values treated as failing ('Hippocampus Only' = cortical regions unusable)")
ap.add_argument("--pet-tracer-column", default="TRACER",
                help="tracer column in the PET table (UC Berkeley 6mm: FBP/FBB); '' if single-tracer")
ap.add_argument("--pet-tracers", default="FBP,FBB", help="tracers to keep (FBP = florbetapir/AV45, FBB = florbetaben)")
ap.add_argument("--pet-window-days", type=int, default=180, help="PET must be within this many days of baseline")
ap.add_argument("--phases", default="ADNI1,ADNIGO,ADNI2,ADNI3",
                help="ADNI phases (ORIGPROT) to include, if ADNIMERGE has an ORIGPROT column")
a = ap.parse_args()

mri, pet = pd.read_csv(a.mri, low_memory=False), pd.read_csv(a.pet, low_memory=False)
if a.suggest:
    print("roi,mri_column,pet_column   # candidates only - verify against data dictionaries")
    for roi in ROIS:
        print(f"{roi},{'|'.join(suggest(mri.columns, roi))},{'|'.join(suggest(pet.columns, roi))}")
    sys.exit(0)

M = pd.read_csv(a.adnimerge, low_memory=False)
bl = M[M.VISCODE == "bl"].drop_duplicates("RID").copy()
if "ORIGPROT" in bl.columns and a.phases:
    bl = bl[bl.ORIGPROT.isin([p.strip() for p in a.phases.split(",")])]
if not a.include_smc:
    bl = bl[bl.DX_bl != "SMC"]
bl["dx"] = bl.DX_bl.map(DX_MAP)
bl = bl[bl.dx.notna()]
bl["subject_id"] = bl.PTID
bl["BL_DATE"] = pd.to_datetime(bl.EXAMDATE, errors="coerce")
ref = bl[["RID", "BL_DATE"]]

rmap = pd.read_csv(a.roi_map)
rois = list(rmap.roi)


def apply_mri_qc(df):
    """Pre-registered rule (OSF 10.17605/OSF.IO/J4KG6): MRI sessions failing the UCSF overall
    FreeSurfer QC are treated as missing MRI. Prints how many rows were removed."""
    col = a.mri_qc_column
    if not col:
        return df
    if col not in df.columns:
        print(f"WARNING: QC column '{col}' not found in the FreeSurfer table; check the data dictionary "
              f"and pass --mri-qc-column. No QC filtering applied.")
        return df
    fail = {v.strip().lower() for v in a.qc_fail_values.split(",")}
    bad = df[col].astype(str).str.strip().str.lower().isin(fail)
    print(f"MRI QC: removed {int(bad.sum())} of {len(df)} FreeSurfer rows with {col} in {sorted(fail)}")
    return df[~bad]


def regional(df, col_name, window=None, tracer_col=None):
    rid = "RID" if "RID" in df else df.columns[0]
    date = next((c for c in ("EXAMDATE", "SCANDATE") if c in df), None)
    if window is not None and date is None:
        raise SystemExit(f"{col_name}: a date window was requested but the table has no EXAMDATE/SCANDATE column")
    if window is not None:
        d = df.copy(); d[date] = pd.to_datetime(d[date], errors="coerce")
        d = d.merge(ref, left_on=rid, right_on="RID", how="inner", suffixes=("", "_ref"))
        d["gap"] = (d[date] - d["BL_DATE"]).dt.days.abs()
        d = d[d.gap <= window].sort_values("gap").drop_duplicates(rid)
    else:
        d = pick_visit(df, rid, date, ref, a.window_days)
    keep = [rid] + list(rmap[col_name]) + ([tracer_col] if tracer_col else [])
    out = d[keep].copy()
    out.columns = ["RID"] + rois + (["tracer"] if tracer_col else [])
    out[rois] = out[rois].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=rois)   # a subject has a modality only if all of its ROIs are present
    cols = ["subject_id"] + rois + (["tracer"] if tracer_col else [])
    return out.merge(bl[["RID", "subject_id"]], on="RID")[cols]


tcol = a.pet_tracer_column if a.pet_tracer_column and a.pet_tracer_column in pet.columns else None
if a.pet_tracer_column and not tcol:
    print(f"WARNING: tracer column '{a.pet_tracer_column}' not in the PET table; all PET treated as one tracer")
if tcol:
    keep_tr = {t.strip().upper() for t in a.pet_tracers.split(",")}
    pet = pet[pet[tcol].astype(str).str.upper().isin(keep_tr)]
    print("PET rows by tracer:", pet[tcol].value_counts().to_dict())
mri_out = regional(apply_mri_qc(mri), "mri_column")
pet_out = regional(pet, "pet_column", window=a.pet_window_days, tracer_col=tcol)
clin = pd.DataFrame({"subject_id": bl.subject_id, "age": bl.AGE,
                     "sex": (bl.PTGENDER == "Male").astype(int), "educ": bl.PTEDUCAT, "apoe4": bl.APOE4,
                     "abeta": to_num(bl.get("ABETA_bl", pd.Series(np.nan, index=bl.index))),
                     "ptau": to_num(bl.get("PTAU_bl", pd.Series(np.nan, index=bl.index))),
                     "mmse": bl.get("MMSE_bl")})   # mmse kept for Table 1 only; not a model input
clin["ptau_abeta"] = clin.ptau / clin.abeta
labels = bl[["subject_id", "dx", "SITE"]].rename(columns={"SITE": "site"})

os.makedirs(a.out, exist_ok=True)
for n, d in (("labels", labels), ("mri", mri_out), ("pet", pet_out), ("clinical", clin)):
    d.to_csv(os.path.join(a.out, f"{n}.csv"), index=False)
print(labels.dx.value_counts().to_string())
print(f"MRI available: {mri_out.subject_id.isin(labels.subject_id).sum()}  PET available: {pet_out.subject_id.isin(labels.subject_id).sum()}")
print(f"CSF available: {clin.abeta.notna().sum()}")
