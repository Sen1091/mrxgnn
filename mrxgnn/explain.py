"""Explanations and how they are evaluated. Evaluating the explanations is a core part of the
method, not decoration.

- ROI attributions: Integrated Gradients on the MRI/PET W-scores. The baseline is 0, which means a
  normative, age/sex-expected CN brain, so an attribution reads "deviation from normal pushes the
  prediction towards class c".
- Faithfulness (deletion test): reset the top-k attributed ROIs to normative (0) and measure the
  drop in true-class probability, compared with resetting k random ROIs. A faithful explanation
  gives a much larger drop.
- Anatomical plausibility: precision@k and AUROC of ROI importance against an a-priori AD-signature
  ROI set, which is never used in training, compared with chance.
- Rule ablation: switch off each rule, then measure the change in true-class probability and in
  balanced accuracy over the subjects where that rule is active.
- Stability: Spearman correlation of group-level ROI rankings across folds and seeds (see
  scripts/summarise.py).
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


def _prob(model, b, **kw):
    return model(b, **kw)["logits"].softmax(-1)


def integrated_gradients(model, b, target, steps=32):
    model.eval()
    x_m, x_p = b["mri"], b["pet"]
    tot_m, tot_p = torch.zeros_like(x_m), torch.zeros_like(x_p)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        bm = dict(b)
        bm["mri"] = (a * x_m).requires_grad_(True)
        bm["pet"] = (a * x_p).requires_grad_(True)
        out = model(bm)["logits"]
        sel = out.gather(1, target.view(-1, 1)).sum()
        gm, gp = torch.autograd.grad(sel, [bm["mri"], bm["pet"]], allow_unused=True)
        if gm is not None:
            tot_m += gm
        if gp is not None:
            tot_p += gp
    att_m = (x_m * tot_m / steps) * b["m_mri"].view(-1, 1)
    att_p = (x_p * tot_p / steps) * b["m_pet"].view(-1, 1)
    return att_m.detach(), att_p.detach()


@torch.no_grad()
def deletion_curve(model, b, importance, ks=(1, 2, 4, 8, 16), n_random=5, seed=0):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    y = b["y"]
    p0 = _prob(model, b).gather(1, y.view(-1, 1)).squeeze(1)
    R = importance.shape[1]
    order = importance.argsort(1, descending=True)
    res = {"k": list(ks), "drop_top": [], "drop_random": []}
    for k in ks:
        bm = dict(b)
        keep = torch.ones_like(b["mri"])
        keep.scatter_(1, order[:, :k], 0.0)
        bm["mri"], bm["pet"] = b["mri"] * keep, b["pet"] * keep
        res["drop_top"].append(float((p0 - _prob(model, bm).gather(1, y.view(-1, 1)).squeeze(1)).mean()))
        dr = []
        for _ in range(n_random):
            rnd = torch.stack([torch.randperm(R, generator=g)[:k] for _ in range(len(y))])
            keep = torch.ones_like(b["mri"]); keep.scatter_(1, rnd, 0.0)
            bm["mri"], bm["pet"] = b["mri"] * keep, b["pet"] * keep
            dr.append(float((p0 - _prob(model, bm).gather(1, y.view(-1, 1)).squeeze(1)).mean()))
        res["drop_random"].append(float(np.mean(dr)))
    return res


def plausibility(importance, is_signature, y, disease_classes, ks=(5, 10, 20)):
    """importance: [N, R] (abs attribution). Evaluated on subjects of disease classes only."""
    imp = importance[np.isin(y, disease_classes)]
    sig = np.asarray(is_signature, bool)
    out = {"chance_precision": float(sig.mean())}
    order = np.argsort(-imp, 1)
    for k in ks:
        out[f"precision@{k}"] = float(sig[order[:, :k]].mean())
    aucs = [roc_auc_score(sig, row) for row in imp if row.std() > 0]
    out["auroc_mean"] = float(np.mean(aucs)) if aucs else float("nan")
    return out


@torch.no_grad()
def rule_ablation(model, b, rule_names):
    if model.rules is None:
        return []
    model.eval()
    y = b["y"].numpy()
    base = model(b)
    p0 = base["logits"].softmax(-1)
    active = (base["rule_contrib"].abs().sum(-1) > 1e-6).numpy()   # [B, K]
    out = []
    K = len(rule_names)
    for k in range(K):
        dis = torch.zeros(K); dis[k] = 1
        p1 = model(b, disabled_rules=dis)["logits"].softmax(-1)
        a = active[:, k]
        if a.sum() == 0:
            out.append({"rule": rule_names[k], "n_active": 0}); continue
        yt = torch.tensor(y)
        dprob = (p0.gather(1, yt.view(-1, 1)) - p1.gather(1, yt.view(-1, 1))).squeeze(1).numpy()[a]
        out.append({"rule": rule_names[k], "n_active": int(a.sum()),
                    "mean_true_prob_drop": float(dprob.mean()),
                    "bacc_with": float(balanced_accuracy_score(y[a], p0.argmax(1).numpy()[a])),
                    "bacc_without": float(balanced_accuracy_score(y[a], p1.argmax(1).numpy()[a]))})
    return out
