"""External validation: train on the source cohort (ADNI), test on the target cohort (OASIS-3).

Protocol (fixed in docs/analysis_plan.md):
  - For each seed: train on all source participants (85% train / 15% validation for early stopping).
  - Target: a random 50% of target CN participants form the *normative calibration* set, used only
    to fit target W-scores. They are EXCLUDED from evaluation. Everyone else is evaluated.
  - Primary external endpoint: AUC for CN vs cognitively impaired (target MCI-proxy + AD), with
    score = 1 - P(CN). OASIS-3 has no MCI label; CDR 0.5 with AD-type aetiology is the MCI proxy.
    Secondary: three-class balanced accuracy, and the same metrics with PET withheld.
  - Reference comparisons: (i) logistic regression trained on the same source features;
    (ii) MR-XGNN WITHOUT target normative recalibration (source normaliser applied directly).
"""
import json
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from .atlas import AD_SIGNATURE
from .data import load_cohort, CrossCohortNormaliser, FoldNormaliser
from .train import train_fold, predict, apply_scenario, CSF_VARS
from . import explain as X


def _lr_feats(T):
    return np.c_[T["mri"].numpy(), T["pet"].numpy(), T["m_pet"].numpy()[:, None],
                 (T["clin"] * T["clin_mask"]).numpy(), T["clin_mask"].numpy()]


def run_external(cfg, target_dir, out_dir, calib_frac=0.5, scenarios=("observed", "drop_pet", "mri_only")):
    os.makedirs(out_dir, exist_ok=True)
    src = load_cohort(cfg["data_dir"], cfg["clinical_features"], cfg["classes"])
    tgt = load_cohort(target_dir, cfg["clinical_features"], cfg["classes"])
    assert src.rois == tgt.rois, "source and target ROI columns must be identical (same atlas, same order)"
    rules = []
    if cfg.get("use_rules") and cfg.get("rules_file"):
        rules = [r for r in yaml.safe_load(open(cfg["rules_file"]))["rules"]
                 if r["var"].split(":", 1)[1] in (src.clin_names + src.rois)]
    csf_idx = [i for i, c in enumerate(src.clin_names) if c in CSF_VARS]
    sig = np.array([r in AD_SIGNATURE for r in src.rois])
    rows, expl, calib_log = [], [], []
    for seed in cfg["cv"]["seeds"]:
        idx = np.arange(len(src))
        tr, va = train_test_split(idx, test_size=cfg["cv"]["val_frac"], stratify=src.y, random_state=seed)
        model, src_norm, n_ep, vscore = train_fold(src, np.sort(tr), np.sort(va), cfg, rules, seed, csf_idx)
        rng = np.random.default_rng(seed)
        cn = np.where(tgt.y == 0)[0]
        calib = np.sort(rng.choice(cn, int(round(calib_frac * len(cn))), replace=False))
        ev = np.setdiff1d(np.arange(len(tgt)), calib)
        calib_log.append({"seed": seed, "n_calib_cn": int(len(calib)), "n_eval": int(len(ev)), "epochs": n_ep})
        norm_re = CrossCohortNormaliser().fit_cross(src, np.sort(tr), tgt, calib)
        Tsrc = src_norm.transform(src, np.sort(np.r_[tr, va]))
        lr = LogisticRegressionCV(Cs=10, cv=3, max_iter=3000, class_weight="balanced", random_state=seed)
        lr.fit(_lr_feats(Tsrc), Tsrc["y"].numpy())
        for recal, norm in (("recalibrated", norm_re), ("no_recalibration", src_norm)):
            T = norm.transform(tgt, ev)
            for sc in scenarios:
                Ts = apply_scenario(T, sc, csf_idx)
                for method, p in (("MR-XGNN", predict(model, Ts)), ("LogReg", lr.predict_proba(_lr_feats(Ts)))):
                    df = pd.DataFrame(p, columns=[f"p_{c}" for c in cfg["classes"]])
                    df.insert(0, "subject_id", tgt.ids[ev]); df.insert(1, "y", tgt.y[ev])
                    df["seed"], df["scenario"], df["method"], df["normalisation"] = seed, sc, method, recal
                    df["m_pet"] = tgt.m_pet[ev]
                    rows.append(df)
        if cfg.get("explain", True):
            T = norm_re.transform(tgt, ev)
            am, ap = X.integrated_gradients(model, T, T["y"], steps=cfg.get("ig_steps", 32))
            imp = am.abs() + ap.abs()
            disease = [i for i, c in enumerate(cfg["classes"]) if c != "CN"]
            expl.append({"seed": seed, "deletion": X.deletion_curve(model, T, imp),
                         "plausibility": X.plausibility(imp.numpy(), sig, tgt.y[ev], disease),
                         "rule_ablation": X.rule_ablation(model, T, [r["name"] for r in rules])})
        print(f"external seed {seed}: trained {n_ep} epochs; calib CN={len(calib)} eval={len(ev)}", flush=True)
    P = pd.concat(rows)
    P.to_csv(os.path.join(out_dir, "predictions_external.csv"), index=False)
    json.dump({"calibration": calib_log, "explanations": expl}, open(os.path.join(out_dir, "external_explain.json"), "w"), indent=1)
    return P


def summarise_external(out_dir, classes=("CN", "MCI", "AD"), B=1000):
    from .metrics import all_metrics
    P = pd.read_csv(os.path.join(out_dir, "predictions_external.csv"))
    pc = [f"p_{c}" for c in classes]
    rows = []
    for (m, nrm, sc), Q in P.groupby(["method", "normalisation", "scenario"]):
        aucs, baccs = [], []
        for _, q in Q.groupby("seed"):
            yb = (q.y.values > 0).astype(int)
            aucs.append(roc_auc_score(yb, 1 - q["p_CN"].values))
            if q.y.nunique() == len(classes):
                baccs.append(all_metrics(q[pc].values, q.y.values, list(classes))["bacc"])
        # bootstrap CI on seed-averaged scores (subjects may differ across seeds; use common subjects)
        G = Q.groupby("subject_id").agg(y=("y", "first"), s=("p_CN", "mean"), k=("seed", "nunique"))
        rng = np.random.default_rng(0); yb = (G.y.values > 0).astype(int); s = 1 - G.s.values; bs = []
        for _ in range(B):
            i = rng.integers(0, len(G), len(G))
            if yb[i].min() != yb[i].max():
                bs.append(roc_auc_score(yb[i], s[i]))
        rows.append({"method": m, "normalisation": nrm, "scenario": sc, "n_subjects": len(G),
                     "auc_cn_vs_impaired": f"{100*np.mean(aucs):.1f} ± {100*np.std(aucs):.1f}",
                     "auc_ci": f"[{100*np.percentile(bs,2.5):.1f}, {100*np.percentile(bs,97.5):.1f}]",
                     "bacc_3class": f"{100*np.mean(baccs):.1f} ± {100*np.std(baccs):.1f}" if baccs else "n/a",
                     "_auc_mean": np.mean(aucs), "_bacc_mean": np.mean(baccs) if baccs else np.nan})
    S = pd.DataFrame(rows)
    os.makedirs(os.path.join(out_dir, "summary"), exist_ok=True)
    # H5: recalibrated vs not (MR-XGNN, observed), paired bootstrap of AUC on seed-averaged scores
    try:
        g = lambda nrm: P[(P.method == "MR-XGNN") & (P.normalisation == nrm) & (P.scenario == "observed")] \
            .groupby("subject_id").agg(y=("y", "first"), s=("p_CN", "mean"))
        A, Bn = g("recalibrated"), g("no_recalibration")
        common = A.index.intersection(Bn.index); A, Bn = A.loc[common], Bn.loc[common]
        yb = (A.y.values > 0).astype(int); sa, sb = 1 - A.s.values, 1 - Bn.s.values
        rng = np.random.default_rng(1); ds = []
        for _ in range(B):
            i = rng.integers(0, len(yb), len(yb))
            if yb[i].min() != yb[i].max():
                ds.append(roc_auc_score(yb[i], sa[i]) - roc_auc_score(yb[i], sb[i]))
        ds = np.array(ds); d0 = roc_auc_score(yb, sa) - roc_auc_score(yb, sb)
        json.dump({"delta_auc_recal_minus_norecal": float(d0), "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                   "p": float(min(1, 2 * min((ds <= 0).mean(), (ds >= 0).mean())))},
                  open(os.path.join(out_dir, "summary", "h5_recalibration.json"), "w"), indent=1)
    except Exception as e:  # noqa
        print("H5 comparison skipped:", e)
    S.to_csv(os.path.join(out_dir, "summary", "external_table.csv"), index=False)
    open(os.path.join(out_dir, "summary", "external_table.md"), "w").write(
        "External validation (train source, test target). AUC: CN vs impaired, score = 1 - P(CN).\n\n"
        + S.drop(columns=["_auc_mean", "_bacc_mean"]).to_markdown(index=False))
    return S
