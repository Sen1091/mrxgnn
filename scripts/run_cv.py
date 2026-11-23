"""Run one configuration.
  python scripts/run_cv.py --config configs/default.yaml --set out_dir=results/x model.use_pet=false
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrxgnn.config import load
from mrxgnn.train import run_cv

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
run_cv(load(a.config, a.set))
