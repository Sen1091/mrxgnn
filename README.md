# MR-XGNN: a pre-registered evaluation of multimodal Alzheimer's disease classification

Code for the article *"What do graph structure, amyloid PET and knowledge rules add to multimodal Alzheimer's
disease classification? A pre-registered evaluation with faithfulness-tested explanations"*
(D. Senthil Kumar, R. Pugalenthi; submitted to *Neuroinformatics*).

Pre-registration: https://doi.org/10.17605/OSF.IO/J4KG6 · Deviations: `docs/deviations.md`

MR-XGNN is a graph neural network over 82 FreeSurfer regions that fuses regional MRI, amyloid PET and clinical
data (CSF, APOE). It predicts missing PET from MRI, uses graph attention biased by an anatomical prior, and adds
guideline-anchored knowledge rules. The repository also contains the classical baselines, all ablations, the
explanation tests (deletion faithfulness, anatomical plausibility, stability, rule switch-off), and the scripts
that produce every table and figure in the paper.

**No ADNI data are included.** ADNI data must be obtained from https://adni.loni.usc.edu under its Data Use
Agreement. `split_definitions/` lists only the participant identifiers (PTID) assigned to each fold of the three-class
analysis, so that the cohort and splits can be checked; it contains no imaging, clinical or diagnostic data.

## Install
```bash
python -m venv venv            # Python 3.10-3.12
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q                      # 7 tests, synthetic data only
```

## Reproduce the paper
1. Download from ADNI (IDA → Study Files) into one folder, e.g. `../ADNI_raw`:
   `ADNIMERGE2` R package (`.tar.gz`), `UCSFFSX7_*.csv`, `UCBERKELEY_AMY_6MM_*.csv`, `UPENNBIOMK_ROCHE_ELECSYS_*.csv`.
2. Build the model inputs (prints aggregate counts only):
   ```bash
   python scripts/prepare_adni2.py --raw ../ADNI_raw --out data/adni_processed
   ```
3. Run everything (resumable; about 1-2 days on a 16-thread CPU):
   ```bash
   python run_all.py --jobs 3
   ```
   This runs the full model, three baselines, ten ablations, the binary tasks, the summaries, the figures, and
   writes `results/paper_numbers.json`. If Node.js is installed it also fills the manuscript template.
4. Supplementary material: `python manuscript/build_esm1.py --help`.

`scripts/run_external.py` and `scripts/prepare_oasis3.py` implement the registered OASIS-3 external validation,
which could not be carried out in the paper (deviation C8).

## Layout
| Path | Contents |
|---|---|
| `mrxgnn/` | model, training, data normalisation (fold-internal W-scores), metrics, explanations |
| `scripts/` | data preparation, cross-validation, baselines, ablations, statistics, figures |
| `configs/` | fixed hyperparameters (`default.yaml`) and rules (`rules.yaml`) |
| `docs/` | analysis plan and deviations log |
| `manuscript/` | manuscript and supplement builders (numbers inserted from `paper_numbers.json`) |
| `split_definitions/` | three-class participant-to-fold assignment for every seed (identifiers only) |
| `tests/` | unit and pipeline tests on synthetic data |

## Licence and citation
MIT licence (`LICENSE`). Please cite the article and the archived software (`CITATION.cff`).
