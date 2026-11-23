"""Metrics with bootstrap confidence intervals. Balanced accuracy is the primary metric, because
ADNI classes are imbalanced."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix


def ece(prob, y, bins=10):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs((pred[s] == y[s]).mean() - conf[s].mean())
    return float(e)


def all_metrics(prob, y, classes):
    pred = prob.argmax(1)
    K = len(classes)
    out = {"n": int(len(y)), "bacc": balanced_accuracy_score(y, pred), "acc": float((pred == y).mean()),
           "macro_f1": f1_score(y, pred, average="macro"), "ece": ece(prob, y)}
    try:
        out["auc"] = roc_auc_score(y, prob[:, 1]) if K == 2 else roc_auc_score(y, prob, multi_class="ovr", average="macro")
    except ValueError:
        out["auc"] = float("nan")
    cm = confusion_matrix(y, pred, labels=list(range(K)))
    for i, c in enumerate(classes):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        out[f"sens_{c}"] = tp / max(tp + fn, 1)
        out[f"spec_{c}"] = tn / max(tn + fp, 1)
        out[f"prec_{c}"] = tp / max(tp + fp, 1)
    out["macro_prec"] = float(np.mean([out[f"prec_{c}"] for c in classes]))
    out["macro_spec"] = float(np.mean([out[f"spec_{c}"] for c in classes]))
    out["confusion"] = cm.tolist()
    return {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v) for k, v in out.items()}


def bootstrap_ci(prob, y, classes, keys=("bacc", "auc", "macro_f1"), B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = {k: [] for k in keys}
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < len(classes):
            continue
        m = all_metrics(prob[i], y[i], classes)
        for k in keys:
            vals[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


def paired_bootstrap_delta(prob_a, prob_b, y, classes, key="bacc", B=2000, seed=0):
    """Paired bootstrap for metric(A) - metric(B) on the same subjects. Returns (delta, CI, p)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d0 = all_metrics(prob_a, y, classes)[key] - all_metrics(prob_b, y, classes)[key]
    ds = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < len(classes):
            continue
        ds.append(all_metrics(prob_a[i], y[i], classes)[key] - all_metrics(prob_b[i], y[i], classes)[key])
    ds = np.array(ds)
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return float(d0), (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))), float(min(p, 1.0))


def paired_bootstrap_drop(a_obs, a_drop, b_obs, b_drop, y, classes, key="bacc", B=2000, seed=0):
    """H2: is model A's loss (observed -> PET withheld) smaller than model B's?
    Returns (drop_B - drop_A, CI, p). A positive value means A degrades less."""
    rng = np.random.default_rng(seed)
    f = lambda i: ((all_metrics(b_obs[i], y[i], classes)[key] - all_metrics(b_drop[i], y[i], classes)[key])
                   - (all_metrics(a_obs[i], y[i], classes)[key] - all_metrics(a_drop[i], y[i], classes)[key]))
    n = len(y); d0 = f(np.arange(n)); ds = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) == len(classes):
            ds.append(f(i))
    ds = np.array(ds)
    return float(d0), (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))), float(min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean())))
