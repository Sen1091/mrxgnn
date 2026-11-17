"""Subject-level repeated stratified cross-validation with modality dropout and evaluation under
each missing-modality scenario."""
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from .atlas import AD_SIGNATURE
from .data import load_cohort, FoldNormaliser, batches
from .graph import covariance_prior
from .model import MRXGNN
from .metrics import all_metrics
from . import explain as X

CSF_VARS = ("abeta", "ptau", "tau", "ptau_abeta")


def make_splits(y, folds, seed, val_frac):
    """Yields (train, val, test) index arrays. There is one row per subject, so a subject can
    never appear in two partitions; scripts/check_leakage.py verifies this on the saved splits."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for f, (trv, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr, va = train_test_split(trv, test_size=val_frac, stratify=y[trv], random_state=seed * 100 + f)
        yield f, np.sort(tr), np.sort(va), np.sort(te)


def apply_scenario(b, scenario, csf_idx):
    b = dict(b)
    if scenario in ("drop_pet", "mri_only"):
        b["m_pet"] = torch.zeros_like(b["m_pet"]); b["pet"] = torch.zeros_like(b["pet"])
    if scenario == "drop_csf" and csf_idx:
        cm = b["clin_mask"].clone(); cm[:, csf_idx] = 0; b["clin_mask"] = cm
    if scenario == "mri_only":
        b["clin_mask"] = torch.zeros_like(b["clin_mask"])
    if scenario != "observed":
        b["clin"] = b["clin"] * b["clin_mask"]; b["clin_raw"] = b["clin_raw"] * b["clin_mask"]
    return b


def random_drop(b, tc, csf_idx, gen):
    b = dict(b)
    n = len(b["y"])
    keep_pet = (torch.rand(n, generator=gen) > tc["p_drop_pet"]).float()
    b["m_pet"] = b["m_pet"] * keep_pet
    drop_mri = (torch.rand(n, generator=gen) < tc["p_drop_mri"]).float() * b["m_pet"]  # only if PET remains
    b["m_mri"] = b["m_mri"] * (1 - drop_mri)
    b["pet"] = b["pet"] * b["m_pet"].view(-1, 1); b["mri"] = b["mri"] * b["m_mri"].view(-1, 1)
    if csf_idx:
        keep_csf = (torch.rand(n, generator=gen) > tc["p_drop_csf"]).float().view(-1, 1)
        cm = b["clin_mask"].clone(); cm[:, csf_idx] = cm[:, csf_idx] * keep_csf; b["clin_mask"] = cm
        b["clin"] = b["clin"] * cm; b["clin_raw"] = b["clin_raw"] * cm
    return b


@torch.no_grad()
def predict(model, T, bs=256):
    model.eval()
    return torch.cat([model(b)["logits"].softmax(-1) for b in batches(T, bs, False)]).numpy()


def train_fold(co, tr, va, cfg, rules, seed, csf_idx):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    mc, tc = cfg["model"], cfg["train"]
    norm = FoldNormaliser().fit(co, tr)
    Ttr, Tva = norm.transform(co, tr), norm.transform(co, va)
    prior = covariance_prior(Ttr["mri"].numpy(), Ttr["m_mri"].numpy(), co.rois, k=mc.get("prior_k", 8))
    model = MRXGNN(len(co.rois), len(co.clin_names), prior, classes=cfg["classes"], d=mc["d"],
                   layers=mc["layers"], heads=mc["heads"], dropout=mc["dropout"],
                   use_mri=mc["use_mri"], use_pet=mc["use_pet"], use_clin=mc["use_clin"],
                   use_graph=mc["use_graph"], use_prior=mc["use_prior"], use_halluc=mc["use_halluc"],
                   fusion=mc["fusion"], rules=rules if cfg.get("use_rules") else None,
                   clin_names=co.clin_names, rois=co.rois)
    # rule parameters get a larger learning rate (default 10x): their gradients are small, and with
    # the base rate the thresholds barely move (seen in the synthetic dry run)
    rule_p = list(model.rules.parameters()) if model.rules is not None else []
    rid = {id(p) for p in rule_p}
    groups = [{"params": [p for p in model.parameters() if id(p) not in rid], "lr": tc["lr"], "weight_decay": tc["wd"]}]
    if rule_p:
        groups.append({"params": rule_p, "lr": tc["lr"] * tc.get("rule_lr_mult", 10.0), "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    cnt = np.bincount(Ttr["y"].numpy(), minlength=len(cfg["classes"]))
    cw = torch.tensor(len(Ttr["y"]) / (len(cnt) * np.maximum(cnt, 1)), dtype=torch.float32)
    best, best_score, bad = None, -1, 0
    for ep in range(tc["epochs"]):
        model.train()
        for b in batches(Ttr, tc["batch"], True, gen):
            out = model(b)
            loss = F.cross_entropy(out["logits"], b["y"], weight=cw)
            if tc.get("moddrop", True):
                bd = random_drop(b, tc, csf_idx, gen)
                od = model(bd)
                loss = loss + F.cross_entropy(od["logits"], b["y"], weight=cw)
                loss = loss + tc["lam_kd"] * F.kl_div(od["logits"].log_softmax(-1),
                                                      out["logits"].detach().softmax(-1), reduction="batchmean")
            if out["h_p_hat"] is not None and b["m_pet"].sum() > 0:
                m = b["m_pet"] > 0
                loss = loss + tc["lam_hall"] * F.mse_loss(out["h_p_hat"][m], out["h_p_real"][m].detach())
            if model.rules is not None:
                loss = loss + tc["lam_anchor"] * model.rules.anchor_penalty()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        # model selection: mean val balanced accuracy over observed and PET-dropped inputs
        pv = predict(model, Tva)
        pvd = predict(model, apply_scenario(Tva, "drop_pet", csf_idx))
        score = 0.5 * (all_metrics(pv, Tva["y"].numpy(), cfg["classes"])["bacc"]
                       + all_metrics(pvd, Tva["y"].numpy(), cfg["classes"])["bacc"])
        if score > best_score + 1e-4:
            best_score, best, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best)
    return model, norm, ep + 1, best_score


def run_cv(cfg):
    out_dir = cfg["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    co = load_cohort(cfg["data_dir"], cfg["clinical_features"], cfg["classes"])
    rules = []
    if cfg.get("use_rules") and cfg.get("rules_file"):
        rules = [r for r in yaml.safe_load(open(cfg["rules_file"]))["rules"]
                 if r["var"].split(":", 1)[1] in (co.clin_names + co.rois)]
    csf_idx = [i for i, c in enumerate(co.clin_names) if c in CSF_VARS]
    sig = np.array([r in AD_SIGNATURE for r in co.rois])
    disease = [i for i, c in enumerate(cfg["classes"]) if c != "CN"]
    preds, splits, logs, expl, imp_rows, rule_rows, gate_rows = [], [], [], [], [], [], []
    t0 = time.time()
    for seed in cfg["cv"]["seeds"]:
        for f, tr, va, te in make_splits(co.y, cfg["cv"]["folds"], seed, cfg["cv"]["val_frac"]):
            for part, idx in (("train", tr), ("val", va), ("test", te)):
                splits.append(pd.DataFrame({"seed": seed, "fold": f, "part": part, "subject_id": co.ids[idx]}))
            model, norm, n_ep, vscore = train_fold(co, tr, va, cfg, rules, seed * 1000 + f, csf_idx)
            Tte = norm.transform(co, te)
            for sc in cfg["scenarios"]:
                p = predict(model, apply_scenario(Tte, sc, csf_idx))
                df = pd.DataFrame(p, columns=[f"p_{c}" for c in cfg["classes"]])
                df.insert(0, "subject_id", co.ids[te]); df.insert(1, "y", co.y[te])
                df["seed"], df["fold"], df["scenario"] = seed, f, sc
                df["m_pet"] = co.m_pet[te]
                df["m_csf"] = (Tte["clin_mask"][:, csf_idx].sum(1) > 0).numpy().astype(int) if csf_idx else 0
                preds.append(df)
            logs.append({"seed": seed, "fold": f, "epochs": n_ep, "val_score": vscore})
            if cfg.get("explain", True):
                am, ap = X.integrated_gradients(model, Tte, Tte["y"], steps=cfg.get("ig_steps", 32))
                imp = (am.abs() + ap.abs())
                e = {"seed": seed, "fold": f,
                     "deletion": X.deletion_curve(model, Tte, imp),
                     "plausibility": X.plausibility(imp.numpy(), sig, co.y[te], disease),
                     "rule_ablation": X.rule_ablation(model, Tte, [r["name"] for r in rules])}
                expl.append(e)
                d = pd.DataFrame(imp.numpy(), columns=co.rois)
                d.insert(0, "subject_id", co.ids[te]); d.insert(1, "y", co.y[te]); d["seed"], d["fold"] = seed, f
                imp_rows.append(d)
                with torch.no_grad():
                    o = model(Tte)
                g = pd.DataFrame(o["gate"].numpy(), columns=co.rois)
                g.insert(0, "y", co.y[te]); g["m_pet"] = co.m_pet[te]; g["seed"], g["fold"] = seed, f
                gate_rows.append(g)
                if model.rules is not None:
                    t1, t2 = model.rules.thresholds()
                    for k, r in enumerate(rules):
                        rule_rows.append({"seed": seed, "fold": f, "rule": r["name"], "t1_guideline": r["t1"],
                                          "t1_learned": float(t1[k]), "t2_learned": float(t2[k]),
                                          "w": model.rules.W[k].detach().numpy().round(4).tolist()})
            print(f"seed {seed} fold {f}: epochs={n_ep} val={vscore:.3f} ({time.time()-t0:.0f}s)", flush=True)
    P = pd.concat(preds); P.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    pd.concat(splits).to_csv(os.path.join(out_dir, "splits.csv"), index=False)
    if imp_rows:
        pd.concat(imp_rows).to_csv(os.path.join(out_dir, "roi_importance.csv"), index=False)
        pd.concat(gate_rows).to_csv(os.path.join(out_dir, "gates.csv"), index=False)
    if rule_rows:
        pd.DataFrame(rule_rows).to_csv(os.path.join(out_dir, "rules_learned.csv"), index=False)
    json.dump({"train_log": logs, "explanations": expl}, open(os.path.join(out_dir, "explain.json"), "w"), indent=1)
    return P
