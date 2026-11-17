"""External validation on OASIS-3 (or any second cohort in the generic format).
  python scripts/run_external.py --config configs/default.yaml --target data/oasis3_processed --out results/external_oasis3
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrxgnn.config import load
from mrxgnn.external import run_external, summarise_external

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--target", required=True); ap.add_argument("--out", default="results/external")
ap.add_argument("--calib-frac", type=float, default=0.5)
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load(a.config, a.set)
run_external(cfg, a.target, a.out, calib_frac=a.calib_frac)
print(summarise_external(a.out, cfg["classes"]).drop(columns=["_auc_mean", "_bacc_mean"]).to_string(index=False))
