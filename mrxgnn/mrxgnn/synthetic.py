"""Synthetic ADNI-like cohort for TESTING THE PIPELINE ONLY.

Numbers produced on this data must never appear in a manuscript. It exists so the code can be
run end-to-end, and the tests can pass, before ADNI access is granted.

It generates ROI-level MRI (regional volume/thickness) and PET (regional SUVR) features, clinical
variables and realistic missingness (PET and CSF are often absent in ADNI), in the same generic
CSV format that scripts/prepare_adni.py writes for real data.
"""
import numpy as np
import pandas as pd

from .atlas import ROIS, AD_SIGNATURE


def make_cohort(n=600, seed=0, p_pet_missing=0.4, p_csf_missing=0.5, p_mri_missing=0.03,
                n_sites=6, effect=0.8, pet_scale=1.0, pet_offset=0.0, mri_offset=0.0, id_prefix="SYN"):
    """pet_scale/pet_offset/mri_offset simulate a second cohort (other scanner, pipeline or tracer)."""
    rng = np.random.default_rng(seed)
    R = len(ROIS)
    sig = np.array([r in AD_SIGNATURE for r in ROIS], float)
    labels = rng.choice(["CN", "MCI", "AD"], size=n, p=[0.40, 0.33, 0.27])
    sev = np.where(labels == "CN", np.clip(rng.normal(0.05, 0.12, n), 0, None),
          np.where(labels == "MCI", rng.uniform(0.15, 0.75, n),
                   np.clip(rng.normal(0.9, 0.18, n), 0.4, None)))
    age = np.clip(rng.normal(73, 7, n) + 2 * sev, 55, 95)
    sex = rng.integers(0, 2, n)
    site = rng.integers(0, n_sites, n)
    site_mri = rng.normal(0, 0.3, (n_sites, R))
    site_pet = rng.normal(0, 0.2, (n_sites, R))

    # low-rank correlated biological noise (global size, network-level factors)
    F = rng.normal(0, 1, (4, R)) * 0.5
    lat = rng.normal(0, 1, (n, 4))
    # regional disease weights: signature ROIs strongest, plus a diffuse component
    w_mri = effect * (1.2 * sig + 0.25 * rng.uniform(0, 1, R))
    w_pet = effect * (1.0 * sig + 0.35 * rng.uniform(0, 1, R))
    # PET is more sensitive early (MCI) -> nonlinear in severity
    mri = (10 - 0.03 * (age[:, None] - 73) + 0.4 * sex[:, None]
           - sev[:, None] ** 1.5 * w_mri * 1.6 + lat @ F + site_mri[site]
           + rng.normal(0, 0.9, (n, R)))
    pet = (1.3 - 0.004 * (age[:, None] - 73) - np.sqrt(sev)[:, None] * w_pet * 0.35
           + 0.1 * (lat @ F) + site_pet[site] * 0.3 + rng.normal(0, 0.12, (n, R)))

    mri = mri + mri_offset
    pet = pet_offset + pet_scale * pet
    apoe4 = np.array([rng.choice([0, 1, 2], p=[0.7 - 0.4 * s, 0.25 + 0.25 * s, 0.05 + 0.15 * s])
                      for s in np.clip(sev, 0, 1)])
    abeta = np.clip(rng.normal(1300 - 700 * np.clip(sev, 0, 1.2) - 80 * apoe4, 280), 200, 2500)
    ptau = np.clip(rng.normal(18 + 22 * np.clip(sev, 0, 1.2), 7), 5, 90)
    educ = np.clip(np.round(rng.normal(16, 2.8, n)), 6, 20)
    mmse = np.clip(np.round(29.2 - 7 * sev + rng.normal(0, 1.2, n)), 10, 30)

    ids = [f"{id_prefix}{i:04d}" for i in range(n)]
    labels_df = pd.DataFrame({"subject_id": ids, "dx": labels, "site": site})
    mri_df = pd.DataFrame(mri, columns=ROIS); mri_df.insert(0, "subject_id", ids)
    pet_df = pd.DataFrame(pet, columns=ROIS); pet_df.insert(0, "subject_id", ids)
    clin = pd.DataFrame({"subject_id": ids, "age": age, "sex": sex, "educ": educ, "apoe4": apoe4,
                         "abeta": abeta, "ptau": ptau, "ptau_abeta": ptau / abeta, "mmse": mmse})
    # missingness: PET missing more often for CN; CSF missing at random
    pet_miss = rng.uniform(size=n) < p_pet_missing * np.where(labels == "CN", 1.2, 0.85)
    csf_miss = rng.uniform(size=n) < p_csf_missing
    mri_miss = rng.uniform(size=n) < p_mri_missing
    pet_df = pet_df[~pet_miss]
    mri_df = mri_df[~mri_miss]
    clin.loc[csf_miss, ["abeta", "ptau", "ptau_abeta"]] = np.nan
    return {"labels": labels_df, "mri": mri_df.reset_index(drop=True),
            "pet": pet_df.reset_index(drop=True), "clinical": clin}


def write_external_cohort(outdir, n=400, seed=7):
    """Shifted second cohort (OASIS-3-like): PIB-like PET scale, MRI offset, no CSF."""
    return write_cohort(outdir, n=n, seed=seed, p_pet_missing=0.3, p_csf_missing=1.0, n_sites=1,
                        pet_scale=1.6, pet_offset=0.4, mri_offset=-0.8, id_prefix="EXT")


def write_cohort(outdir, **kw):
    import os
    os.makedirs(outdir, exist_ok=True)
    c = make_cohort(**kw)
    for k, df in c.items():
        df.to_csv(os.path.join(outdir, f"{k}.csv"), index=False)
    return c
