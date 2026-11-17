# Pre-specified analysis plan: MR-XGNN (ADNI internal, OASIS-3 external)

**Status:** frozen on 4 November 2026, **before** any ADNI or OASIS-3 data were downloaded.
**Registered** on OSF on 4 November 2026 (OSF Preregistration template): https://osf.io/f95pb, DOI https://doi.org/10.17605/OSF.IO/J4KG6. After freezing, every change goes in
`docs/deviations.md` with its date and reason, and the manuscript reports the deviations.

## 1. Data
- **Development cohort:** ADNI 1/GO/2/3.
  - Baseline visit only; MRI with regional FreeSurfer measures required.
  - Diagnoses: CN, MCI (EMCI+LMCI) and AD. SMC is excluded (primary analysis).
  - PET: amyloid (AV45 or FBB) regional SUVR within ±180 days of baseline; otherwise treated as missing.
  - CSF: Elecsys Aβ42 and p-tau181 within ±180 days; otherwise treated as missing.
  - Nobody is excluded for missing PET or CSF.
  - MRI sessions flagged as failing the UCSF overall FreeSurfer quality control are treated as missing MRI (and the participant excluded, since MRI is required). No other outlier removal.
- **External cohort:** OASIS-3.
  - First MR session with FreeSurfer output and a clinical visit within ±365 days.
  - CN: CDR 0 and "cognitively normal".
  - MCI proxy: CDR 0.5 with AD-type diagnosis.
  - AD: CDR ≥ 1 with AD-type diagnosis.
  - Everyone else is excluded, and the exclusions are counted.
  - PET: AV45 (same tracer family as ADNI) within ±365 days. Sensitivity analysis with PIB.
- **Inputs:** 82 Desikan-Killiany regional measures per modality, plus age, sex, education, APOE ε4,
  CSF Aβ42, p-tau181 and p-tau/Aβ42. **MMSE, CDR, ADAS and FAQ are never inputs.**

## 2. Fixed hyperparameters (configs/default.yaml; no tuning on ADNI)
- Model: d=32, 2 layers, 4 heads, dropout 0.2.
- Optimiser: AdamW, lr 1e-3, weight decay 1e-4, batch 64, at most 200 epochs, patience 25.
- Modality dropout: PET 0.5, CSF 0.3, MRI 0.05.
- Loss weights: λ_kd = λ_hall = 0.5, λ_anchor = 0.1, rule learning-rate multiplier 10.
- Prior graph: k = 8.
- Model selection inside each training fold uses the mean validation BACC over observed and
  PET-withheld inputs.

## 3. Internal evaluation
- Stratified 5-fold subject-level CV, repeated with seeds 0–4; 15% of each training fold is used
  for validation. The split files are saved, and `check_leakage.py` must pass.
- Scenarios:
  - observed
  - PET withheld
  - CSF withheld
  - MRI only
  - Subsets of the observed scenario: complete-case, and truly PET-missing.
- Comparators (same folds):
  - LR, RBF-SVM and HGB on concatenated features plus missingness indicators.
  - MRI-only and PET-only graph models.
  - Ablations: no rules, no graph, no prior, concatenation fusion, mask token, no modality
    dropout, no anchoring.

## 4. Hypotheses and decision rules
| # | Hypothesis | Test | Supported if |
|---|---|---|---|
| H1 (primary) | Under observed availability, MR-XGNN is **non-inferior** to the best classical baseline in 3-class balanced accuracy | Paired bootstrap (2000 resamples) of ΔBACC | Lower 95% CI of ΔBACC > −2.0 pp. Superiority is claimed only if the lower CI is > 0 |
| H2 (primary) | With PET withheld, MR-XGNN loses less BACC than (a) its mask-token variant and (b) LR | Paired bootstrap of the difference in drops | 95% CI excludes 0 |
| H3 | Attributions are faithful | Deletion test, k = 8 | Top-8 drop ≥ 2× random-8 drop in ≥ 80% of fold×seed runs |
| H4 | Attributions are anatomically plausible | AUROC of region importance vs the a-priori AD-signature set (MCI+AD subjects) | Mean AUROC > 0.5 with the 95% CI over runs above 0.5 |
| H5 (external) | The model transfers to OASIS-3 with normative recalibration | AUC for CN vs impaired, score 1 − P(CN) | Report the AUC with 95% CI. Recalibrated vs not recalibrated compared by paired bootstrap |
| H6 | Rules add evidence without drifting from guidelines | Rule ablation Δ p(true); threshold drift | Descriptive: report every rule |

- Multiple comparisons: the ablation p-values are Holm-corrected within each scenario.
- **Every hypothesis is reported whatever the outcome.** A negative H1 or H2 result is reported as
  such; the paper's framing follows the data (see "Honest risk" in the rework brief).

## 5. Secondary and sensitivity analyses (pre-listed)
- Binary tasks: CN vs AD and CN vs MCI, with the same pipeline and `classes` set accordingly.
- Including SMC as CN.
- OASIS-3 with PIB instead of AV45.
- Calibration: ECE and reliability diagram.
- Per-site performance in ADNI (descriptive).

## 6. Things we will NOT do
- Use test folds to choose hyperparameters, epochs or thresholds.
- Drop comparators or scenarios from the paper because they look unfavourable.
- Compare our numbers directly with published numbers from different cohorts or splits.
- Report any number from synthetic data.
