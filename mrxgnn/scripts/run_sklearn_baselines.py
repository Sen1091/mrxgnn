"""Strong classical baselines on the SAME folds, seeds and scenarios. Reviewers expect these,
because on ROI features a well-tuned linear model is often hard to beat.
  python scripts/run_sklearn_baselines.py --config configs/default.yaml --root results/adni
Writes <root>/<name>/predictions.csv in the same format as run_cv.py.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from mrxgnn.config import load
from mrxgnn.data import load_cohort, FoldNormaliser
from mrxgnn.train import make_splits, apply_scenario, CSF_VARS

MODELS = {
    "lr_concat": lambda s: LogisticRegressionCV(Cs=10, cv=3, max_iter=3000, class_weight="balanced", random_state=s),
    "svm_rbf_concat": lambda s: SVC(C=1.0, probability=True, class_weight="balanced", random_state=s),
    "hgb_concat": lambda s: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=s),
}


def feats(T):
    # W-scores (0 = normative, which doubles as mean imputation) + availability indicators
    return np.c_[T["mri"].numpy(), T["pet"].numpy(), T["m_pet"].numpy()[:, None],
                 (T["clin"] * T["clin_mask"]).numpy(), T["clin_mask"].numpy()]


ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--root", default="results/ablations")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load(a.config, a.set)
co = load_cohort(cfg["data_dir"], cfg["clinical_features"], cfg["classes"])
csf_idx = [i for i, c in enumerate(co.clin_names) if c in CSF_VARS]
for name, mk in MODELS.items():
    rows = []
    for seed in cfg["cv"]["seeds"]:
        for f, tr, va, te in make_splits(co.y, cfg["cv"]["folds"], seed, cfg["cv"]["val_frac"]):
            trv = np.sort(np.r_[tr, va])            # classical models tune internally, no early stopping
            norm = FoldNormaliser().fit(co, trv)
            Ttr, Tte = norm.transform(co, trv), norm.transform(co, te)
            clf = mk(seed).fit(feats(Ttr), Ttr["y"].numpy())
            for sc in cfg["scenarios"]:
                p = clf.predict_proba(feats(apply_scenario(Tte, sc, csf_idx)))
                df = pd.DataFrame(p, columns=[f"p_{c}" for c in cfg["classes"]])
                df.insert(0, "subject_id", co.ids[te]); df.insert(1, "y", co.y[te])
                df["seed"], df["fold"], df["scenario"], df["m_pet"] = seed, f, sc, co.m_pet[te]
                rows.append(df)
        print(name, "seed", seed, "done", flush=True)
    od = os.path.join(a.root, name); os.makedirs(od, exist_ok=True)
    pd.concat(rows).to_csv(os.path.join(od, "predictions.csv"), index=False)
