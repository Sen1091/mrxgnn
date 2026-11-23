"""ADNI now distributes ADNIMERGE only as an R package (ADNIMERGE_x.y.z.tar.gz). This script pulls the
`adnimerge` table out of it as a CSV, without needing R.

  python scripts/extract_adnimerge.py --pkg ../ADNI_raw/ADNIMERGE_x.y.z.tar.gz --out ../ADNI_raw/ADNIMERGE.csv

If the package stores its data in R's lazy-load format (data/Rdata.rdb), Python cannot read it. The
script then prints a two-line R alternative. In Colab: Runtime -> Change runtime type -> R, or run
`!apt-get -qq install r-base-core` and use Rscript.
"""
import argparse, os, sys, tarfile, tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True, help="path to ADNIMERGE_*.tar.gz")
ap.add_argument("--out", required=True, help="output CSV path")
ap.add_argument("--table", default="adnimerge", help="data object to export (default: adnimerge)")
a = ap.parse_args()

R_HINT = f"""
Use R instead (2 lines):
  Rscript -e "install.packages('{a.pkg}', repos=NULL, type='source')"
  Rscript -e "library(ADNIMERGE); write.csv({a.table}, '{a.out}', row.names=FALSE)"
"""

with tarfile.open(a.pkg) as tf:
    names = tf.getnames()
    data = [n for n in names if "/data/" in n]
    print(f"{len(data)} data files in package; e.g.", data[:8])
    rda = [n for n in data if os.path.basename(n).lower() in (f"{a.table}.rda", f"{a.table}.rdata")]
    if not rda:
        if any(n.endswith("Rdata.rdb") for n in data):
            sys.exit("Package uses R lazy-load data (Rdata.rdb), which Python cannot read." + R_HINT)
        sys.exit(f"No {a.table}.rda found in the package." + R_HINT)
    try:
        import pyreadr
    except ImportError:
        sys.exit("Install pyreadr first:  pip install pyreadr")
    with tempfile.TemporaryDirectory() as tmp:
        member = tf.getmember(rda[0])
        member.name = os.path.basename(member.name)       # extract flat, never outside tmp
        tf.extract(member, tmp, filter="data")
        res = pyreadr.read_r(os.path.join(tmp, member.name))
df = res[a.table] if a.table in res else next(iter(res.values()))
for c in df.columns:                      # R factors arrive as pandas categoricals: write as text
    if str(df[c].dtype) == "category":
        df[c] = df[c].astype(str).replace("nan", "")
for c in df.columns:                      # R stores integers as doubles: restore whole-number ids/counts
    if df[c].dtype.kind == "f" and df[c].notna().any() and (df[c].dropna() % 1 == 0).all() and c in ("RID", "SITE", "APOE4", "PTEDUCAT"):
        df[c] = df[c].astype("Int64")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(df)} rows x {df.shape[1]} columns")
need = ["RID", "PTID", "VISCODE", "DX_bl", "EXAMDATE", "AGE", "PTGENDER", "PTEDUCAT", "APOE4", "ABETA_bl", "PTAU_bl"]
missing = [c for c in need if c not in df.columns]
print("columns the pipeline needs:", "all present" if not missing else f"MISSING {missing} - send me the column list")
