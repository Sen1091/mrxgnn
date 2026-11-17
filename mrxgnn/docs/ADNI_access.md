# Getting ADNI data (do this first; approval usually takes about 1–2 weeks)

1. Go to https://adni.loni.usc.edu → "Data & Samples" → "Access Data and Samples" → apply
   through the LONI Image & Data Archive (IDA).
2. Each author who touches the data applies separately, with an institutional email
   (not Gmail), and agrees to the ADNI Data Use Agreement.
   - The research plan should be specific: *"Region-level multimodal (MRI, PET, CSF) graph neural
     network for Alzheimer's diagnosis robust to missing modalities, with evaluated explanations;
     baseline visits of ADNI 1/GO/2/3."*
3. After approval, go to IDA → ADNI → "Download" → "Study Data" and download:
   - `ADNIMERGE` (Study Info). This has the baseline diagnosis, demographics, APOE4 and CSF.
   - A UCSF FreeSurfer cross-sectional regional table (MRI image analysis), plus its data dictionary.
   - A UC Berkeley regional PET table (FDG and/or AV45/FBB amyloid), plus its data dictionary and
     methods PDF.
   - The UPENN CSF (Elecsys) methods document, so the CSF cut-offs in `configs/rules.yaml` can be
     checked.
4. **Publication requirements:**
   - Author by-line: "… for the Alzheimer's Disease Neuroimaging Initiative*".
   - The ADNI footnote and acknowledgement text from `ADNI_Manuscript_Citations.pdf` on the ADNI site.
   - Check whether the current DUA requires sending the manuscript to the ADNI Data and Publications
     Committee before submission.
5. **Report in the paper:**
   - Download date and table versions.
   - The ADNI phases included.
   - Inclusion rule: baseline visit, MRI available.
   - A flow diagram of how many subjects had PET and CSF.

Optional external validation: AIBL (https://aibl.org.au) or OASIS-3 (https://sites.wustl.edu/oasisbrains).
Each has its own application.

## OASIS-3 (external validation) — apply in parallel
1. Create a NITRC account: https://www.nitrc.org/account/register.php
2. Submit the OASIS data request at https://sites.wustl.edu/oasisbrains/ and accept the OASIS Data Use Terms.
   The terms cover: non-commercial use, no re-identification, and secure storage. Publications that use
   Avid (AV45) data must be sent to Avid 30 days before publication.
3. Download from NITRC-IR:
   - ADRC clinical data (CDR, diagnoses)
   - demographics (sex, education, APOE)
   - FreeSurfer regional outputs
   - PUP regional PET SUVRs (AV45; PIB for the sensitivity analysis)
4. Fill in `configs/oasis3_columns.yaml` and `configs/roi_map_oasis3.csv` from the real headers
   (`scripts/prepare_oasis3.py --suggest`), then run `prepare_oasis3.py`.
5. Cite LaMontagne et al. (2019), doi:10.1101/2019.12.13.19014902, and use the OASIS-3 acknowledgement text.
