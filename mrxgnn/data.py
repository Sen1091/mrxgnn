"""Loading, alignment, missing-modality masks and leakage-safe normalisation.

Input format (one row per subject, baseline visit only):
  labels.csv    subject_id, dx (CN|MCI|AD), site
  mri.csv       subject_id, <ROI columns>      subjects without usable MRI are simply absent
  pet.csv       subject_id, <ROI columns>      subjects without PET are simply absent
  clinical.csv  subject_id, age, sex, educ, apoe4, abeta, ptau, ptau_abeta, mmse, ... (NaN = missing)

All statistics (W-score regression, z-scoring) are fitted on the TRAINING subjects of each
fold only. W-scores are fitted on training-fold CN subjects, then applied to everyone.
"""
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
import torch

CLASSES = ["CN", "MCI", "AD"]


@dataclass
class Cohort:
    ids: np.ndarray            # [N]
    y: np.ndarray              # [N] int
    site: np.ndarray           # [N]
    mri: np.ndarray            # [N, R] raw (NaN rows if missing)
    pet: np.ndarray            # [N, R]
    m_mri: np.ndarray          # [N] 1 = available
    m_pet: np.ndarray
    clin: np.ndarray           # [N, C] raw, NaN = missing
    clin_names: list
    covars: np.ndarray         # [N, k] age, sex (for W-scores)
    rois: list
    pet_tracer: np.ndarray = None   # [N] PET tracer label ("" if none/unknown)

    def __len__(self):
        return len(self.ids)


def load_cohort(datadir, clinical_features, classes=CLASSES, require_mri=True):
    lab = pd.read_csv(os.path.join(datadir, "labels.csv"))
    lab = lab[lab.dx.isin(classes)].drop_duplicates("subject_id")
    mri = pd.read_csv(os.path.join(datadir, "mri.csv")).drop_duplicates("subject_id")
    pet = pd.read_csv(os.path.join(datadir, "pet.csv")).drop_duplicates("subject_id")
    clin = pd.read_csv(os.path.join(datadir, "clinical.csv")).drop_duplicates("subject_id")
    rois = [c for c in mri.columns if c != "subject_id"]
    assert [c for c in pet.columns if c not in ("subject_id", "tracer")] == rois, "MRI/PET ROI columns must match"
    if require_mri:
        lab = lab[lab.subject_id.isin(mri.subject_id)]
    ids = lab.subject_id.values
    M = mri.set_index("subject_id").reindex(ids)[rois].values.astype(float)
    P = pet.set_index("subject_id").reindex(ids)[rois].values.astype(float)
    C = clin.set_index("subject_id").reindex(ids)
    tracer = (pet.set_index("subject_id").reindex(ids)["tracer"].fillna("").astype(str).values
              if "tracer" in pet.columns else np.full(len(ids), "", dtype=object))
    for need in ("age", "sex"):
        assert need in C.columns, f"clinical.csv needs '{need}' for W-scores"
    return Cohort(ids=ids, y=lab.dx.map({c: i for i, c in enumerate(classes)}).values,
                  site=lab["site"].values if "site" in lab else np.zeros(len(ids)),
                  mri=M, pet=P,
                  m_mri=(~np.isnan(M).any(1)).astype(np.float32),
                  m_pet=(~np.isnan(P).any(1)).astype(np.float32),
                  clin=C[clinical_features].values.astype(float), clin_names=list(clinical_features),
                  covars=C[["age", "sex"]].values.astype(float), rois=rois, pet_tracer=tracer)


def _groups(co):
    """Normalisation groups per modality: MRI is one group; PET is grouped by tracer, because
    amyloid tracers (e.g. florbetapir vs florbetaben, or PIB) have different SUVR scales."""
    tr = co.pet_tracer if co.pet_tracer is not None else np.full(len(co), "", dtype=object)
    return {"mri": np.full(len(co), "", dtype=object), "pet": np.asarray(tr, dtype=object)}


def _fit_w(X, covars, s):
    D = np.c_[np.ones(len(s)), covars[s]]
    beta, *_ = np.linalg.lstsq(D, X[s], rcond=None)
    res = X[s] - D @ beta
    return beta, res.std(0) + 1e-6


class FoldNormaliser:
    """W-scores for imaging (regress ROI ~ age + sex on training CN subjects; PET separately for
    each tracer) and z-scores for clinical variables, all fitted on the training fold only."""

    def fit(self, co: Cohort, idx):
        tr = np.asarray(idx)
        cn = tr[co.y[tr] == 0]
        G = _groups(co)
        self.img = {}
        for name, X, m in (("mri", co.mri, co.m_mri), ("pet", co.pet, co.m_pet)):
            self.img[name] = {}
            for g in sorted(set(G[name][tr[m[tr] > 0]])):
                s = cn[(m[cn] > 0) & (G[name][cn] == g)]
                if len(s) < 10:  # fall back to all training subjects with the modality/tracer
                    s = tr[(m[tr] > 0) & (G[name][tr] == g)]
                if len(s) >= 3:
                    self.img[name][g] = _fit_w(X, co.covars, s)
        c = co.clin[tr]
        self.c_mu, self.c_sd = np.nanmean(c, 0), np.nanstd(c, 0) + 1e-6
        self.c_mu = np.nan_to_num(self.c_mu); self.c_sd = np.nan_to_num(self.c_sd, nan=1.0)
        return self

    def transform(self, co: Cohort, idx):
        idx = np.asarray(idx)
        out = {}
        D = np.c_[np.ones(len(idx)), co.covars[idx]]
        G = _groups(co)
        for name, X, m in (("mri", co.mri, co.m_mri), ("pet", co.pet, co.m_pet)):
            W = np.zeros((len(idx), X.shape[1]))
            avail = m[idx].astype(np.float32).copy()
            for g, (beta, sd) in self.img[name].items():
                rows = np.where(G[name][idx] == g)[0]
                W[rows] = (X[idx][rows] - D[rows] @ beta) / sd
            # a modality whose tracer was never seen in training has no reference -> treated as missing
            seen = np.isin(G[name][idx], list(self.img[name].keys()))
            avail = avail * seen
            out[name] = np.nan_to_num(W * avail[:, None]).astype(np.float32)
            out["m_" + name] = avail.astype(np.float32)
        raw = co.clin[idx]
        out["clin_mask"] = (~np.isnan(raw)).astype(np.float32)
        out["clin"] = np.nan_to_num((raw - self.c_mu) / self.c_sd).astype(np.float32)
        out["clin_raw"] = np.nan_to_num(raw).astype(np.float32)   # physical units, for rules
        out["y"] = co.y[idx].astype(np.int64)
        return {k: torch.from_numpy(v) for k, v in out.items()}


def batches(t, bs, shuffle, gen=None):
    n = len(t["y"])
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = order[i:i + bs]
        yield {k: v[j] for k, v in t.items()}


class CrossCohortNormaliser(FoldNormaliser):
    """External validation. Imaging W-scores for the TARGET cohort are fitted on a calibration
    subset of target CN participants, which are then excluded from evaluation. This per-cohort,
    per-tracer normative reference removes scanner, pipeline and tracer offsets; for example PIB
    and AV45 SUVRs sit on different scales, and each is expressed as SD from its own CN norm.
    Clinical variables keep the SOURCE training statistics, because they are in physical units."""

    def fit_cross(self, source: Cohort, source_idx, target: Cohort, calib_idx):
        src = FoldNormaliser().fit(source, source_idx)
        tgt = FoldNormaliser().fit(target, calib_idx)   # calib_idx should contain CN subjects only
        self.img = tgt.img
        self.c_mu, self.c_sd = src.c_mu, src.c_sd
        return self
