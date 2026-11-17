"""Run the full model plus every ablation and baseline variant on the same folds and seeds.
  python scripts/run_ablations.py --config configs/default.yaml --root results/adni [--only full,no_graph]
Each variant writes to <root>/<variant>/. Then run scripts/summarise.py --root <root>.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrxgnn.config import load, apply_overrides
from mrxgnn.train import run_cv

VARIANTS = {
    # name: (overrides, description used in the ablation table)
    "full":          ([], "MR-XGNN (full)"),
    "mri_only":      (["model.use_pet=false", "model.use_clin=false", "use_rules=false"], "MRI graph only"),
    "pet_only":      (["model.use_mri=false", "model.use_clin=false", "use_rules=false"], "PET graph only"),
    "no_clin_rules": (["model.use_clin=false", "use_rules=false"], "MRI+PET, no clinical/rules"),
    "no_rules":      (["use_rules=false"], "w/o knowledge rules"),
    "no_graph":      (["model.use_graph=false"], "w/o graph attention (node-wise MLP)"),
    "no_prior":      (["model.use_prior=false"], "w/o anatomical prior"),
    "concat_fusion": (["model.fusion=concat"], "concat instead of gated fusion"),
    "no_halluc":     (["model.use_halluc=false"], "w/o PET hallucination (mask token)"),
    "no_moddrop":    (["train.moddrop=false", "model.use_halluc=false"], "w/o modality dropout/distillation"),
    "no_anchor":     (["train.lam_anchor=0.0"], "rules w/o threshold anchoring"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--root", default="results/ablations")
ap.add_argument("--only", default="")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
base = load(a.config, a.set)
names = a.only.split(",") if a.only else list(VARIANTS)
for n in names:
    ov, desc = VARIANTS[n]
    cfg = apply_overrides(base, ov + [f"out_dir={os.path.join(a.root, n)}"])
    print(f"=== {n}: {desc}", flush=True)
    run_cv(cfg)
