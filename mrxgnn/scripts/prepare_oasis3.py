"""Build generic inputs (labels / mri / pet / clinical) for OASIS-3 external validation.

Access: register at NITRC and apply at https://sites.wustl.edu/oasisbrains/ (OASIS Data Use Terms).
Download the ADRC clinical data, demographics, FreeSurfer regional outputs and PUP regional PET SUVRs.

The column names are set in configs/oasis3_columns.yaml and the ROI names in configs/roi_map_oasis3.csv.
Both must be checked against the real headers.

Rules:
  - One MR session per participant: the first one with FreeSurfer output that has a clinical
    visit within +/-365 days.
  - Label from that clinical visit:
      CN      CDR 0 and diagnosis "Cognitively normal"
      MCI     proxy: CDR 0.5 with an AD-type diagnosis
      AD      CDR >= 1 with an AD-type diagnosis
    Anything else (other aetiologies, uncertain) is excluded and counted.
  - PET: nearest session of the chosen tracer within +/-365 days of the MR session.
  - CSF is not available, so those columns are NaN and the CSF rules switch off automatically.

  python scripts/prepare_oasis3.py --clinical ADRC.csv --demographics demo.csv --fs FS.csv --pup PUP.csv \
      --tracer AV45 --out data/oasis3_processed
  python scripts/prepare_oasis3.py --fs FS.csv --pup PUP.csv --suggest
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, yaml
from mrxgnn.atlas import ROIS

ap = argparse.ArgumentParser()
ap.add_argument("--clinical"); ap.add_argument("--demographics"); ap.add_argument("--fs", required=True)
ap.add_argument("--pup", required=True); ap.add_argument("--tracer", default="AV45")
ap.add_argument("--columns", default="configs/oasis3_columns.yaml")
ap.add_argument("--roi-map", default="configs/roi_map_oasis3.csv")
ap.add_argument("--out", default="data/oasis3_processed"); ap.add_argument("--suggest", action="store_true")
a = ap.parse_args()
K = yaml.safe_load(open(a.columns))
fs, pup = pd.read_csv(a.fs, low_memory=False), pd.read_csv(a.pup, low_memory=False)

if a.suggest:
    print("FreeSurfer columns:", list(fs.columns)[:60], "...")
    print("PUP columns:", list(pup.columns)[:60], "...")
    print("\nroi,mri_column,pet_column   # candidates - verify")
    for roi in ROIS:
        h, r = roi.split("_", 1)
        m = [c for c in fs.columns if r.lower() in c.lower().replace("_", "") and h in c.lower()][:2]
        side = "L" if h == "lh" else "R"
        p = [c for c in pup.columns if r.upper() in c.upper().replace("_", "") and (f"_{side}_" in c.upper() or h.upper() in c.upper())][:2]
        print(f"{roi},{'|'.join(m)},{'|'.join(p)}")
    sys.exit(0)


def days(label):
    m = re.search(r"_d(\d+)", str(label))
    return int(m.group(1)) if m else np.nan


cl = pd.read_csv(a.clinical, low_memory=False); de = pd.read_csv(a.demographics, low_memory=False)
kc, kd, kf, kp = K["clinical"], K["demographics"], K["freesurfer"], K["pup"]
cl["subj"], cl["day"] = cl[kc["subject"]], cl[kc["label"]].map(days)
fs["subj"], fs["day"] = fs[kf["subject"]], fs[kf["label"]].map(days)
pup["subj"], pup["day"] = pup[kp["subject"]], pup[kp["label"]].map(days)
pup = pup[pup[kp["tracer"]].astype(str).str.upper() == a.tracer.upper()]
rmap = pd.read_csv(a.roi_map); rois = list(rmap.roi)
W = K["windows_days"]


def label_of(row):
    cdr, dx = row[kc["cdr"]], str(row[kc["dx"]])
    for name, rule in K["labels"].items():
        if cdr in rule["cdr"] and any(s.lower() in dx.lower() for s in rule["dx_contains"]):
            return name.upper()
    return None


out_lab, out_mri, out_pet, out_cli, excl = [], [], [], [], {"no_clinical_match": 0, "other_dx": 0}
for subj, g in fs.sort_values("day").groupby("subj"):
    c = cl[cl.subj == subj]
    chosen = None
    for _, mr in g.iterrows():
        if c.empty:
            break
        j = (c.day - mr.day).abs().idxmin()
        if abs(c.loc[j, "day"] - mr.day) <= W["clinical_to_mr"]:
            chosen = (mr, c.loc[j]); break
    if chosen is None:
        excl["no_clinical_match"] += 1; continue
    mr, cv = chosen
    lab = label_of(cv)
    if lab is None:
        excl["other_dx"] += 1; continue
    out_lab.append({"subject_id": subj, "dx": lab, "site": "OASIS3"})
    out_mri.append([subj] + [pd.to_numeric(mr[c_], errors="coerce") for c_ in rmap.mri_column])
    p = pup[pup.subj == subj]
    if len(p):
        k = (p.day - mr.day).abs().idxmin()
        if abs(p.loc[k, "day"] - mr.day) <= W["pet_to_mr"]:
            out_pet.append([subj] + [pd.to_numeric(p.loc[k, c_], errors="coerce") for c_ in rmap.pet_column])
    d = de[de[kd["subject"]] == subj]
    age = float(cv[kc["age"]]) + mr.day / 365.25 if pd.notna(cv[kc["age"]]) else np.nan
    sex = np.nan
    apoe = np.nan
    educ = np.nan
    if len(d):
        sx = str(d.iloc[0][kd["sex"]]).strip().upper()
        sex = 1 if sx in ("M", "1", "MALE") else 0 if sx in ("F", "2", "FEMALE") else np.nan
        educ = pd.to_numeric(d.iloc[0][kd["educ"]], errors="coerce")
        g_ = str(d.iloc[0][kd["apoe"]]).split(".")[0]
        apoe = g_.count("4") if g_.isdigit() else np.nan
    out_cli.append({"subject_id": subj, "age": age, "sex": sex, "educ": educ, "apoe4": apoe,
                    "abeta": np.nan, "ptau": np.nan, "ptau_abeta": np.nan, "cdr": cv[kc["cdr"]]})

os.makedirs(a.out, exist_ok=True)
pd.DataFrame(out_lab).to_csv(os.path.join(a.out, "labels.csv"), index=False)
pd.DataFrame(out_mri, columns=["subject_id"] + rois).dropna().to_csv(os.path.join(a.out, "mri.csv"), index=False)
pd.DataFrame(out_pet, columns=["subject_id"] + rois).dropna().assign(tracer=a.tracer.upper()).to_csv(os.path.join(a.out, "pet.csv"), index=False)
pd.DataFrame(out_cli).to_csv(os.path.join(a.out, "clinical.csv"), index=False)
L = pd.DataFrame(out_lab)
print(L.dx.value_counts().to_string()); print("excluded:", excl); print("with PET:", len(out_pet))
