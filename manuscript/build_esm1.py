"""Build Online Resource 1 (electronic supplementary material) for the Neuroinformatics manuscript.
  python manuscript/build_esm1.py --results results --figures figures --roi-map data/adni_processed/roi_map_used.csv \
      --deviations docs/deviations.md --rules configs/rules.yaml --out ESM_1.docx
Uses only aggregate result files; no participant-level data.
"""
import argparse, json, os, re
import pandas as pd, yaml
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

ap = argparse.ArgumentParser()
for k in ("results", "figures", "roi-map", "deviations", "rules", "out", "log"):
    ap.add_argument("--" + k)
a = ap.parse_args()
R = a.results
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2)

H = lambda t, l=1: doc.add_heading(t, level=l)
def P(t, bold=False, italic=False, size=None):
    p = doc.add_paragraph(); parts = re.split(r"(\*\*.*?\*\*)", t)
    for x in parts:
        r = p.add_run(x.strip("*") if x.startswith("**") else x); r.bold = bold or x.startswith("**"); r.italic = italic
        if size: r.font.size = Pt(size)
    return p
def T(caption, header, rows, note=None):
    P(caption, size=9)
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(str(h)); r.bold = True; r.font.size = Pt(8)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(8)
    if note: P(note, italic=True, size=8)
    doc.add_paragraph()
f1 = lambda m, s: f"{100*m:.1f} ± {100*s:.1f}"
NAMES = {"full": "MR-XGNN", "lr_concat": "Logistic regression", "svm_rbf_concat": "RBF-SVM", "hgb_concat": "Gradient boosting",
         "mri_only": "MRI only (graph)", "pet_only": "PET only (graph)", "no_clin_rules": "w/o clinical data and rules",
         "no_rules": "w/o rules", "no_graph": "w/o graph attention", "no_prior": "w/o anatomical prior",
         "concat_fusion": "concatenation fusion", "no_halluc": "w/o PET prediction", "no_moddrop": "w/o modality dropout",
         "no_anchor": "w/o threshold anchoring"}

t = doc.add_paragraph(); r = t.add_run("Online Resource 1"); r.bold = True; r.font.size = Pt(14)
P("What do graph structure, amyloid PET and knowledge rules add to multimodal Alzheimer’s disease classification? A pre-registered evaluation with faithfulness-tested explanations", italic=True)
P("Journal: Neuroinformatics. Corresponding author: [corresponding author, institutional e-mail]")
P("Contents: S1 Participant selection · S2 Features and region map · S3 Rules · S4 Implementation and hyperparameters · "
  "S5 Detailed three-class results · S6 Secondary binary tasks · S7 Additional figures · S8 Deviations from the pre-registration · S9 Leakage checks")

# ---------- S1 flow
H("S1 Participant selection (ADNI)")
log = open(a.log).read() if a.log and os.path.exists(a.log) else ""
num = lambda pat: (re.search(pat, log) or [None, "?"])[1]
el = re.search(r"eligible by diagnosis: \{'MCI': (\d+), 'CN': (\d+), 'AD': (\d+)\}", log)
T("Table S1 Participant flow. Source tables: ADNIMERGE2 (ADSL, DXSUM, ARM, DATADIC), UCSFFSX7, UCBERKELEY_AMY_6MM, UPENNBIOMK_ROCHE_ELECSYS (downloaded 4 November 2026).",
  ["Step", "n remaining / excluded"],
  [["Participants in ADSL (all phases)", num(r"ADSL subjects: (\d+)")],
   ["Original protocol ADNI-1, ADNI-GO, ADNI-2 or ADNI-3", num(r"in phases \[.*?\] : (\d+)")],
   ["Excluded: not enrolled (screen failures)", num(r"not enrolled.*?: (\d+)")],
   ["Excluded: no baseline diagnosis or date", num(r"no baseline diagnosis/date: (\d+)")],
   ["Excluded: enrolled with significant memory concern (SMC)", num(r"SMC enrolment arm: (\d+)")],
   ["Eligible by diagnosis (CN / MCI / AD)", f"{int(el[2])+int(el[1])+int(el[3])} ({el[2]} / {el[1]} / {el[3]})" if el else "?"],
   ["Excluded: no FreeSurfer 7 session within 180 days passing QC", f"{int(el[2])+int(el[1])+int(el[3]) - 2236}" if el else "?"],
   ["Analysed cohort (CN / MCI / AD)", "2236 (752 / 1083 / 401)"],
   ["… with amyloid PET within 180 days (florbetapir / florbetaben)", "1226 (898 / 328)"],
   ["… with Elecsys CSF within 180 days", "1462"]],
  "Baseline diagnosis was taken from DXSUM at visit ‘bl’ (screening visit if no ‘bl’ row) and agreed with ADSL.DX in 100% of participants with both. "
  "FreeSurfer sessions with OVERALLQC ‘Fail’ or ‘Hippocampus Only’ were removed (18 of 12,515 rows); blank OVERALLQC was treated as not failed (C6). "
  "PET scans with qc_flag 0 or −2 and NAV4694 scans were removed (C7). All analysed sessions used FreeSurfer 7.2.0; 1,517 MRI sessions were 3 T and 719 were 1.5 T. Median gap between baseline and MRI was 24 days.")

# ---------- S2 features
H("S2 Features and region map")
P("Each participant is described by 82 regional MRI measures (68 Desikan–Killiany cortical thickness averages and 14 FreeSurfer subcortical volumes), "
  "82 regional amyloid-PET SUVRs for the same regions (UC Berkeley 6-mm analysis, whole-cerebellum reference), and seven clinical variables: age, sex, years of education, "
  "APOE ε4 count, CSF Aβ₄₂, p-tau₁₈₁ and p-tau₁₈₁/Aβ₄₂ (Roche Elecsys). MMSE, CDR, ADAS-Cog and FAQ were not used as inputs. Columns were mapped from the ADNIMERGE2 data dictionary; the map below is generated by the code (roi_map_used.csv).")
rm = pd.read_csv(a.roi_map)
half = (len(rm) + 1) // 2
rows = [[rm.roi[i], rm.mri_column[i], rm.pet_column[i]] for i in range(len(rm))]
T("Table S2 Region-to-column map (UCSFFSX7 code; UC Berkeley column).", ["Region", "FreeSurfer (UCSFFSX7)", "Amyloid PET (UC Berkeley)"], rows)

# ---------- S3 rules
H("S3 Rules")
rules = yaml.safe_load(open(a.rules))["rules"]
E = json.load(open(os.path.join(R, "adni3/summary/explain_summary.json")))
RA, RL = E.get("rule_ablation", {}), E.get("rules_learned", {})
T("Table S3 Rule definitions, learned thresholds (mean ± SD over 25 folds) and switch-off effects in participants in whom the rule is active (three-class model).",
  ["Rule", "Variable", "Direction", "Guideline T₁", "Learned T₁", "n active", "Δ p(true)", "BACC with / without"],
  [[r["name"], r["var"], "higher = disease" if r["higher_is_disease"] else "lower = disease", r["t1"],
    f"{RL.get(r['name'], {}).get('t1_learned', float('nan')):.4g} ± {RL.get(r['name'], {}).get('t1_sd', float('nan')):.2g}",
    f"{RA.get(r['name'], {}).get('n_active', float('nan')):.0f}", f"{RA.get(r['name'], {}).get('mean_true_prob_drop', float('nan')):+.3f}",
    f"{100*RA.get(r['name'], {}).get('bacc_with', float('nan')):.1f} / {100*RA.get(r['name'], {}).get('bacc_without', float('nan')):.1f}"] for r in rules],
  "CSF guideline thresholds: ADNI-derived Elecsys cut-offs of Hansson et al. (2018). MRI rules act on W-scores (SD units from the age- and sex-expected value in CN).")
P("**How the rules were specified.** The six rules and their guideline thresholds were specified by the authors before any model was trained: the CSF cut-offs are the ADNI-derived Elecsys values of Hansson et al. (2018), APOE ε4 carriage is coded as at least one allele, and regional atrophy uses the conventional criterion W < −1.5. The registered procedure, in which two large language models would draft candidate rules from guideline text and a neurologist would accept, edit or reject each rule, was not carried out (deviation C9, S8). The rules have therefore not been clinically reviewed, and H6 is reported descriptively for these author-specified rules.")

# ---------- S4 implementation
H("S4 Implementation and hyperparameters")
env = json.load(open(os.path.join(R, "environment.json")))
cfg = [("Embedding dimension d", "32"), ("Graph attention layers / heads", "2 / 4"), ("Dropout", "0.2"),
       ("Prior graph", "8 strongest |structural covariance| neighbours per region + homotopic edges; fitted per training fold"),
       ("Optimiser", "AdamW, learning rate 1e-3 (rules 1e-2), weight decay 1e-4, batch 64"),
       ("Early stopping", "patience 25, max 200 epochs; criterion = mean validation BACC over observed and PET-withheld inputs"),
       ("Modality dropout", "PET 0.5, CSF 0.3, MRI 0.05 (only if PET present)"), ("Loss weights", "λ_kd 0.5, λ_hall 0.5, λ_anc 0.1"),
       ("Class weights", "inverse class frequency in the training fold"), ("Integrated Gradients", "32 steps; baseline W = 0"),
       ("Cross-validation", "stratified 5-fold, 5 seeds (0–4), 15% of each training fold for validation"),
       ("Baselines", "scikit-learn LogisticRegression, SVC (RBF), HistGradientBoostingClassifier on W-scores + clinical + missingness indicators; median imputation fitted per fold"),
       ("Software", f"Python {env.get('python')}, PyTorch {env.get('torch')}, scikit-learn {env.get('sklearn')}; {env.get('os')}"),
       ("Hardware", "AMD x86-64 CPU, 16 logical cores, no GPU"),
       ("Training time", "about 4–9 h per model for 25 folds with three models in parallel; median 41 epochs (range 29–62) for the full model")]
T("Table S4 Fixed settings (all pre-specified; none tuned on test data).", ["Setting", "Value"], cfg)

# ---------- S5 three-class detail
H("S5 Detailed three-class results")
M = pd.read_csv(os.path.join(R, "adni3/summary/main_table.csv"))
O = M[(M.scenario == "observed") & (M.subset == "all")].set_index("variant")
order = [v for v in NAMES if v in O.index]
T("Table S5a Per-class sensitivity, specificity and precision (%, mean ± SD), observed availability.",
  ["Method"] + [f"{m} {c}" for m in ("Sens", "Spec", "Prec") for c in ("CN", "MCI", "AD")],
  [[NAMES[v]] + [f"{100*O.loc[v, f'{m}_{c}_mean']:.1f}" for m in ("sens", "spec", "prec") for c in ("CN", "MCI", "AD")] for v in order])
T("Table S5b All metrics (%, mean ± SD; 95% CI from participant bootstrap), observed availability.",
  ["Method", "BACC", "BACC 95% CI", "AUC", "AUC 95% CI", "Macro-F1", "Accuracy", "ECE"],
  [[NAMES[v], f1(O.loc[v, "bacc_mean"], O.loc[v, "bacc_sd"]), O.loc[v, "bacc_ci"], f1(O.loc[v, "auc_mean"], O.loc[v, "auc_sd"]), O.loc[v, "auc_ci"],
    f1(O.loc[v, "macro_f1_mean"], O.loc[v, "macro_f1_sd"]), f1(O.loc[v, "acc_mean"], O.loc[v, "acc_sd"]), f1(O.loc[v, "ece_mean"], O.loc[v, "ece_sd"])] for v in order])
sc = ["observed:all", "observed:complete_MRI+PET", "observed:PET_missing", "drop_pet:all", "drop_csf:all", "mri_only:all"]
G = M.set_index(["variant", "scenario", "subset"])
T("Table S5c Balanced accuracy (%) for every variant and availability scenario.",
  ["Method", "Observed", "Complete cases", "PET truly missing", "PET withheld", "CSF withheld", "MRI only"],
  [[NAMES[v]] + [f1(G.loc[(v, *s.split(':')), 'bacc_mean'], G.loc[(v, *s.split(':')), 'bacc_sd']) if (v, *s.split(':')) in G.index else "—" for s in sc] for v in order])
A = pd.read_csv(os.path.join(R, "adni3/summary/ablation.csv"))
T("Table S5d Paired bootstrap comparisons with the full model (BACC full − variant, points; Holm correction within scenario).",
  ["Variant", "Scenario", "Δ", "95% CI", "p", "p (Holm)"],
  [[NAMES.get(r.variant, r.variant), r.scenario, f"{100*r._d:+.1f}", f"{100*r._lo:+.1f} to {100*r._hi:+.1f}", f"{r._p:.3f}", "< 0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}"] for _, r in A.iterrows()])
H2r = pd.read_csv(os.path.join(R, "adni3/summary/h2_robustness.csv"))
T("Table S5e H2: difference in PET-withholding drops (comparator drop − MR-XGNN drop, points; positive favours MR-XGNN).",
  ["Comparator", "Δ drop", "95% CI", "p"],
  [[NAMES.get(r.comparator, r.comparator), f"{r.drop_comparator_minus_drop_ref_pp:+.2f}", f"{r.ci_lo_pp:+.2f} to {r.ci_hi_pp:+.2f}", f"{r.p:.3f}"] for _, r in H2r.iterrows()])
D = E["deletion"]
T("Table S5f Deletion test (drop in true-class probability; mean ± SD over 25 runs).", ["k", "Top-k attributed", "Random k"],
  [[k, f"{v['top_mean']:.3f} ± {v['top_sd']:.3f}", f"{v['random_mean']:.3f} ± {v['random_sd']:.3f}"] for k, v in D.items()])

# ---------- S6 binary
H("S6 Secondary binary tasks (pre-registered)")
P("The same pipeline was run for CN versus AD and CN versus MCI, restricting the cohort to the two classes, with the full model and the three classical baselines.")
for lab, title in (("adni_cn_ad", "CN versus AD"), ("adni_cn_mci", "CN versus MCI")):
    B = pd.read_csv(os.path.join(R, lab, "summary/main_table.csv")); n = int(B.n.max())
    BO = B[(B.scenario == "observed") & (B.subset == "all")].set_index("variant")
    c2 = title.split(" versus ")[1]
    T(f"Table S6{'a' if 'ad' in lab[-2:] else 'b'} {title} (n = {n}); observed availability (%, mean ± SD).",
      ["Method", "BACC", "95% CI", "AUC", f"Sens {c2}", "Spec", "Prec", "ECE"],
      [[NAMES[v], f1(BO.loc[v, "bacc_mean"], BO.loc[v, "bacc_sd"]), BO.loc[v, "bacc_ci"], f1(BO.loc[v, "auc_mean"], BO.loc[v, "auc_sd"]),
        f"{100*BO.loc[v, f'sens_{c2}_mean']:.1f}", f"{100*BO.loc[v, f'spec_{c2}_mean']:.1f}", f"{100*BO.loc[v, f'prec_{c2}_mean']:.1f}",
        f1(BO.loc[v, "ece_mean"], BO.loc[v, "ece_sd"])] for v in ("lr_concat", "svm_rbf_concat", "hgb_concat", "full")],
      "Sens/Spec/Prec refer to the impaired class.")
    BG = B.set_index(["variant", "scenario", "subset"])
    T(f"Table S6{'c' if 'ad' in lab[-2:] else 'd'} {title}: balanced accuracy (%) by availability scenario.",
      ["Method", "Observed", "Complete cases", "PET truly missing", "PET withheld", "CSF withheld", "MRI only"],
      [[NAMES[v]] + [f1(BG.loc[(v, *s.split(':')), 'bacc_mean'], BG.loc[(v, *s.split(':')), 'bacc_sd']) for s in sc] for v in ("lr_concat", "svm_rbf_concat", "hgb_concat", "full")])

# ---------- S7 figures
H("S7 Additional figures")
fc = os.path.join(a.figures, "fig_calibration.png")
if os.path.exists(fc):
    doc.add_picture(fc, width=Cm(11)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P("Fig. S1 Reliability diagram of MR-XGNN for three-class diagnosis (observed availability).", size=9)

# ---------- S8 deviations
H("S8 Deviations from the pre-registration")
P("Registration: OSF Preregistration, 4 November 2026, https://doi.org/10.17605/OSF.IO/J4KG6.")
md = open(a.deviations).read()
rows = [l.strip().strip("|").split("|") for l in md.splitlines() if l.startswith("| C")]
T("Table S8 Deviations and clarifications.", ["#", "Date", "Change", "Reason", "Data seen?", "Expected effect"],
  [[re.sub(r"\*\*", "", c.strip()) for c in r] for r in rows])
tm = re.search(r"Timing:(.*)", md)
if tm: P("Timing:" + tm.group(1))
P("No hyperparameter, model, metric or hypothesis was changed after results were seen.")

# ---------- S9 leakage
H("S9 Leakage checks")
P("(i) Splits are made at the participant level and every participant appears in exactly one test fold per seed; a script (scripts/check_leakage.py) "
  "verified that no participant appears in more than one partition within any fold. (ii) W-score models, clinical standardisation, the prior graph and baseline "
  "imputation are fitted on training partitions only. (iii) MMSE, CDR, ADAS-Cog and FAQ are not inputs. (iv) Only one (baseline) visit per participant is used.")
doc.save(a.out)
print("wrote", a.out)
