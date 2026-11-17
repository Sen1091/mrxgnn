"""Run the full paper pipeline (Windows, macOS, Linux). Resumable: a variant
whose predictions.csv already exists is skipped, so you can stop (Ctrl+C) and restart any time.

  python run_all.py                       # everything, in pre-registered priority order
  python run_all.py --jobs 2              # run 2 variants at once (only if you have >= 8 CPU cores)
  python run_all.py --stage primary       # only what H1/H2 need: full, no_halluc, no_moddrop + baselines

Order: (1) primary: full model + baselines  (2) remaining ablations  (3) binary tasks
(4) OASIS-3 if data/oasis3_processed exists  (5) summaries, figures, paper_numbers.json, manuscript.
"""
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--data", default="data/adni_processed")
ap.add_argument("--root", default="results/adni3")
ap.add_argument("--ext", default="results/external_oasis3")
ap.add_argument("--stage", default="all", choices=["primary", "ablations", "binary", "all"])
ap.add_argument("--jobs", type=int, default=1)
a = ap.parse_args()
PY = sys.executable
os.environ["PYTHONUTF8"] = "1"   # Windows: write UTF-8 files (Greek letters, ±) instead of cp1252
D = [f"data_dir={a.data}"]


def run(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([PY, *args], check=True)


def variant(root, name, extra=()):
    if os.path.exists(os.path.join(root, name, "predictions.csv")):
        print(f"skip {root}/{name} (done)"); return
    run("scripts/run_ablations.py", "--config", a.config, "--root", root, "--only", name, "--set", *D, *extra)


def many(root, names, extra=()):
    if a.jobs > 1:
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(lambda n: variant(root, n, extra), names))
    else:
        for n in names:
            variant(root, n, extra)


os.makedirs(f"{a.root}/summary", exist_ok=True)
run("scripts/record_environment.py", "--out", "results/environment.json")
run("scripts/make_table1.py", "--data", a.data, "--out", f"{a.root}/summary/table1")
many(a.root, ["full", "no_halluc", "no_moddrop"])
if not os.path.exists(os.path.join(a.root, "lr_concat", "predictions.csv")):
    run("scripts/run_sklearn_baselines.py", "--config", a.config, "--root", a.root, "--set", *D)
if a.stage in ("ablations", "all"):
    many(a.root, ["mri_only", "pet_only", "no_clin_rules", "no_rules", "no_graph", "no_prior",
                  "concat_fusion", "no_anchor"])
run("scripts/summarise.py", "--root", a.root)
run("scripts/check_leakage.py", f"{a.root}/full/splits.csv")

if a.stage in ("binary", "all"):
    for c1, c2 in (("CN", "AD"), ("CN", "MCI")):
        B = f"results/adni_{c1.lower()}_{c2.lower()}"
        variant(B, "full", [f"classes=[{c1},{c2}]"])
        if not os.path.exists(os.path.join(B, "lr_concat", "predictions.csv")):
            run("scripts/run_sklearn_baselines.py", "--config", a.config, "--root", B, "--set", *D, f"classes=[{c1},{c2}]")
        run("scripts/summarise.py", "--root", B, "--classes", c1, c2)

ext = []
if a.stage == "all" and os.path.isdir("data/oasis3_processed"):
    run("scripts/make_table1.py", "--data", "data/oasis3_processed", "--out", f"{a.ext}/summary/table1")
    run("scripts/run_external.py", "--config", a.config, "--target", "data/oasis3_processed", "--out", a.ext,
        "--set", *D)
    ext = ["--external", a.ext]
run("scripts/make_figures.py", "--root", a.root, "--out", "figures/", "--variants", "full,no_halluc,no_moddrop,lr_concat")
run("scripts/export_paper_numbers.py", "--internal", a.root, *ext, "--binary", "cn_ad=results/adni_cn_ad", "cn_mci=results/adni_cn_mci", "--out", "results/paper_numbers.json")
import shutil
if shutil.which("node"):
    subprocess.run("npm install --silent", shell=True, cwd="manuscript")
    subprocess.run(["node", "build_cmpb.js", "../results/MR-XGNN_CMPB_filled.docx", "../results/paper_numbers.json", "../figures"], cwd="manuscript")
print("\nDone. Send me (Claude) ONLY: results/paper_numbers.json, the results/*/summary/ folders and figures/."
      "\nThese contain aggregate numbers only. Never send data/ or any predictions.csv / splits.csv.")
