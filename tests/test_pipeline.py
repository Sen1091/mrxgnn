import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, yaml
from mrxgnn.synthetic import write_cohort
from mrxgnn.data import load_cohort, FoldNormaliser
from mrxgnn.graph import covariance_prior
from mrxgnn.model import MRXGNN
from mrxgnn.config import load
from mrxgnn.train import run_cv, make_splits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATS = ["age", "sex", "educ", "apoe4", "abeta", "ptau", "ptau_abeta"]


def _setup(tmp_path):
    d = str(tmp_path / "syn")
    write_cohort(d, n=160, seed=1)
    return d


def test_splits_are_disjoint():
    y = np.repeat([0, 1, 2], 40)
    for f, tr, va, te in make_splits(y, 5, 0, 0.15):
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
        assert len(tr) + len(va) + len(te) == len(y)


def test_normaliser_uses_train_only(tmp_path):
    co = load_cohort(_setup(tmp_path), FEATS)
    tr = np.arange(0, 100); te = np.arange(100, len(co))
    n1 = FoldNormaliser().fit(co, tr)
    co.mri[te] += 100.0                       # perturbing the test data must not change the fitted stats
    n2 = FoldNormaliser().fit(co, tr)
    assert np.allclose(n1.img["mri"][""][0], n2.img["mri"][""][0])


def test_rules_inactive_when_input_missing(tmp_path):
    co = load_cohort(_setup(tmp_path), FEATS)
    idx = np.arange(len(co))
    T = FoldNormaliser().fit(co, idx).transform(co, idx)
    rules = yaml.safe_load(open(os.path.join(ROOT, "configs/rules.yaml")))["rules"]
    prior = covariance_prior(T["mri"].numpy(), T["m_mri"].numpy(), co.rois)
    m = MRXGNN(len(co.rois), len(FEATS), prior, rules=rules, clin_names=FEATS, rois=co.rois)
    T["clin_mask"][:] = 0; T["m_mri"][:] = 0
    out = m(T)
    assert torch.allclose(out["rule_contrib"], torch.zeros_like(out["rule_contrib"]))


def test_runs_without_any_pet(tmp_path):
    co = load_cohort(_setup(tmp_path), FEATS)
    idx = np.arange(len(co))
    T = FoldNormaliser().fit(co, idx).transform(co, idx)
    T["m_pet"][:] = 0
    prior = covariance_prior(T["mri"].numpy(), T["m_mri"].numpy(), co.rois)
    m = MRXGNN(len(co.rois), len(FEATS), prior)
    p = m(T)["logits"].softmax(-1)
    assert torch.isfinite(p).all() and p.shape == (len(co), 3)


def test_end_to_end_and_leakage_check(tmp_path):
    d = _setup(tmp_path)
    out = str(tmp_path / "res")
    cfg = load(os.path.join(ROOT, "configs/default.yaml"),
               [f"data_dir={d}", f"out_dir={out}", "cv.seeds=[0]", "cv.folds=2", "train.epochs=2", "ig_steps=2"])
    P = run_cv(cfg)
    assert set(P.scenario) == set(cfg["scenarios"])
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/check_leakage.py"), os.path.join(out, "splits.csv")])
    assert r.returncode == 0


def test_external_validation_and_export(tmp_path):
    from mrxgnn.synthetic import write_external_cohort
    from mrxgnn.external import run_external, summarise_external
    src = _setup(tmp_path)
    tgt = str(tmp_path / "ext"); write_external_cohort(tgt, n=150, seed=3)
    cfg = load(os.path.join(ROOT, "configs/default.yaml"),
               [f"data_dir={src}", "cv.seeds=[0]", "train.epochs=2", "ig_steps=2"])
    out = str(tmp_path / "ext_res")
    P = run_external(cfg, tgt, out)
    S = summarise_external(out, B=20)
    assert set(S.normalisation) == {"recalibrated", "no_recalibration"}
    # calibration CN participants must never be evaluated
    import json
    calib = json.load(open(os.path.join(out, "external_explain.json")))["calibration"][0]
    assert P[(P.seed == 0) & (P.method == "MR-XGNN") & (P.normalisation == "recalibrated") & (P.scenario == "observed")].shape[0] == calib["n_eval"]


def test_pet_wscores_are_per_tracer(tmp_path):
    """Two amyloid tracers on different SUVR scales must each be normalised to their own CN reference."""
    import pandas as pd
    d = _setup(tmp_path)
    pet = pd.read_csv(os.path.join(d, "pet.csv"))
    half = np.arange(len(pet)) % 2 == 1
    rois = [c for c in pet.columns if c != "subject_id"]
    pet.loc[half, rois] = pet.loc[half, rois] * 1.6 + 0.4          # second tracer, different scale
    pet["tracer"] = np.where(half, "FBB", "FBP")
    pet.to_csv(os.path.join(d, "pet.csv"), index=False)
    co = load_cohort(d, FEATS)
    idx = np.arange(len(co))
    norm = FoldNormaliser().fit(co, idx)
    assert set(norm.img["pet"]) == {"FBP", "FBB"}
    T = norm.transform(co, idx)
    cn = (co.y == 0) & (co.m_pet > 0)
    for tr in ("FBP", "FBB"):
        sel = cn & (co.pet_tracer == tr)
        assert abs(T["pet"].numpy()[sel].mean()) < 0.2       # CN of each tracer centred near 0
