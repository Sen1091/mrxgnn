// Build the Neuroinformatics manuscript.
//   node build_ni.js out.docx [paper_numbers.json] [figures_dir]
// Without a numbers file every {{key}} stays a highlighted placeholder.
const fs = require("fs"), path = require("path");
const C = require("./content_ni.js");
const REFS = JSON.parse(fs.readFileSync(path.join(__dirname, "refs_apa.json")));
const [, , OUT = "manuscript_ni.docx", NUMF, FIGDIR] = process.argv;
const NUM = NUMF && fs.existsSync(NUMF) ? JSON.parse(fs.readFileSync(NUMF)) : {};
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, ImageRun, Footer, PageNumber, TabStopType, PageBreak } = require("docx");
const FONT = "Times New Roman";
const missing = new Set(), used = new Set();

function cite(nums) {
  const parts = nums.map(n => { used.add(n); return REFS[n - 1].cite.replace(/^\(|\)$/g, ""); });
  return "(" + parts.join("; ") + ")";
}
function expand(t, short) {
  t = t.replace(/<(\d+(?:,\d+)*)>/g, (_, s) => cite(s.split(",").map(Number)));
  return t.replace(/\{\{([^}]+)\}\}/g, (_, k) => (k in NUM ? `§§${NUM[k]}§§` : (missing.add(k), short ? "[[—]]" : `[[${k}]]`)));
}
function runs(text, base = {}, short = false) {
  text = expand(text, short);
  const out = [], re = /(\[\[.*?\]\]|§§.*?§§|\*\*.*?\*\*|\*[^*\s][^*]*?\*)/g; let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("[[")) out.push(new TextRun({ text: "[" + t.slice(2, -2) + "]", font: FONT, highlight: "yellow", ...base }));
    else if (t.startsWith("§§")) out.push(new TextRun({ text: t.slice(2, -2), font: FONT, ...base }));
    else if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, o.run || {}), spacing: { after: 120, line: 480 },
  alignment: o.align || AlignmentType.JUSTIFIED, indent: o.indent });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT, bold: true, size: 26 })], spacing: { before: 300, after: 140 } });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT, bold: true, italics: true, size: 24 })], spacing: { before: 200, after: 100 } });
const W = 9026;
const EQ = (t, n) => new Paragraph({ children: [new TextRun({ text: "\t" + t, font: "Cambria Math", italics: true }), new TextRun({ text: `\t(${n})`, font: FONT })],
  tabStops: [{ type: TabStopType.CENTER, position: W / 2 }, { type: TabStopType.RIGHT, position: W }], spacing: { before: 100, after: 160 } });

const bd = { style: BorderStyle.SINGLE, size: 4, color: "000000" }, nb = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(caption, header, rows, note, firstFrac = 0.3) {
  const n = header.length, first = Math.round(W * firstFrac), rest = Math.floor((W - first) / (n - 1));
  const widths = [first, ...Array(n - 1).fill(rest)]; widths[n - 1] = W - first - rest * (n - 2);
  const cell = (t, i, h, last) => new TableCell({ width: { size: widths[i], type: WidthType.DXA },
    borders: { top: h ? bd : nb, bottom: (h || last) ? bd : nb, left: nb, right: nb },
    shading: h ? { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" } : undefined, margins: { top: 30, bottom: 30, left: 60, right: 60 },
    children: [new Paragraph({ alignment: i ? AlignmentType.CENTER : AlignmentType.LEFT, children: runs(t, { size: 17, bold: h }, true) })] });
  return [new Paragraph({ children: runs(caption, { size: 20 }), spacing: { before: 240, after: 80 }, keepNext: true }),
    new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
        ...rows.map((r, ri) => new TableRow({ children: r.map((t, i) => cell(t, i, false, ri === rows.length - 1)) }))] }),
    ...(note ? [new Paragraph({ children: runs(note, { size: 16, italics: true }), spacing: { after: 240 } })] : [])];
}
const k = s => `{{${s}}}`;
const V = { lr_concat: "Logistic regression", svm_rbf_concat: "RBF-SVM", hgb_concat: "Gradient boosting", mri_only: "MRI graph only",
  pet_only: "PET graph only", no_clin_rules: "MRI+PET graph, no clinical/rules", full: "**MR-XGNN (ours)**",
  no_rules: "w/o knowledge rules", no_graph: "w/o graph attention (node-wise MLP)", no_prior: "w/o anatomical prior",
  concat_fusion: "concatenation instead of gating", no_halluc: "mask token instead of hallucinator",
  no_moddrop: "w/o modality dropout/distillation", no_anchor: "rules w/o threshold anchoring" };
const T1ROWS = ["Age (years)", "Female", "Education (years)", "APOE ε4 carriers", "MMSE (not a model input)", "CDR (not a model input)", "PET available", "CSF available"];
const RULES = [["csf_abeta42_low", "CSF Aβ₄₂ low (pg/mL)", "977"], ["csf_ptau_abeta_ratio_high", "CSF p-tau/Aβ₄₂ high", "0.025"],
  ["apoe4_carrier", "APOE ε4 carrier", "≥ 1 allele"], ["left_hippocampal_atrophy", "Left hippocampal atrophy (W)", "−1.5"],
  ["right_hippocampal_atrophy", "Right hippocampal atrophy (W)", "−1.5"], ["left_entorhinal_atrophy", "Left entorhinal atrophy (W)", "−1.5"]];
const TABLES = {
  t1a: () => table("**Table 1a** ADNI development cohort at baseline (mean ± SD or n (%)). Group differences: ANOVA or χ².",
    ["Characteristic", "CN", "MCI", "AD", "p"], [["N", k("t1.n.CN"), k("t1.n.MCI"), k("t1.n.AD"), ""],
      ...T1ROWS.filter(r => !r.startsWith("CDR")).map(r => [r, k(`t1.${r}.CN`), k(`t1.${r}.MCI`), k(`t1.${r}.AD`), k(`t1.${r}.p`)])]),
  t1b: () => table("**Table 1b** OASIS-3 external cohort (MCI = CDR 0.5 proxy). CSF is not available.",
    ["Characteristic", "CN", "MCI proxy", "AD", "p"], [["N", k("et1.n.CN"), k("et1.n.MCI"), k("et1.n.AD"), ""],
      ...T1ROWS.filter(r => !r.startsWith("MMSE") && !r.startsWith("CSF")).map(r => [r, k(`et1.${r}.CN`), k(`et1.${r}.MCI`), k(`et1.${r}.AD`), k(`et1.${r}.p`)])]),
  t2: () => table("**Table 2** Three-class diagnosis in ADNI with observed availability (%; mean ± SD over 5 × 5-fold cross-validation; 95% bootstrap CI for balanced accuracy).",
    ["Method", "Balanced accuracy", "95% CI", "Macro-AUC", "Macro-F1", "ECE"],
    ["lr_concat", "svm_rbf_concat", "hgb_concat", "mri_only", "pet_only", "no_clin_rules", "full"].map(v =>
      [V[v], k(`main.${v}.bacc`), k(`main.${v}.bacc_ci`), k(`main.${v}.auc`), k(`main.${v}.macro_f1`), k(`main.${v}.ece`)]), null, 0.26),
  t3: () => table("**Table 3** Balanced accuracy (%) under missing-modality scenarios in ADNI.",
    ["Method", "Observed (all)", "Complete cases", "PET truly missing", "PET withheld", "CSF withheld", "MRI only"],
    ["lr_concat", "hgb_concat", "no_moddrop", "no_halluc", "full"].map(v => [V[v],
      ...["observed.all", "observed.complete_MRI+PET", "observed.PET_missing", "drop_pet.all", "drop_csf.all", "mri_only.all"].map(s => k(`rob.${v}.${s}`))]), null, 0.24),
  t4: () => table("**Table 4** Ablation: full model minus variant, paired bootstrap difference in balanced accuracy (points; positive = full model better) with Holm-adjusted p.",
    ["Variant", "Δ observed", "95% CI", "p", "Δ PET withheld", "95% CI", "p"],
    ["no_rules", "no_graph", "no_prior", "concat_fusion", "no_halluc", "no_moddrop", "no_anchor"].map(v => [V[v],
      k(`abl.${v}.observed.d`), k(`abl.${v}.observed.ci`), k(`abl.${v}.observed.p`), k(`abl.${v}.drop_pet.d`), k(`abl.${v}.drop_pet.ci`), k(`abl.${v}.drop_pet.p`)]), null, 0.26),
  t5: () => table("**Table 5** Evaluation of the explanations (mean ± SD over folds × seeds).",
    ["Criterion", "Metric", "MR-XGNN", "Reference"],
    [["Faithfulness", "Δ p(true), top-8 regions reset", k("expl.del.8.top"), "random 8: " + k("expl.del.8.rand")],
     ["Faithfulness", "Δ p(true), top-16 regions reset", k("expl.del.16.top"), "random 16: " + k("expl.del.16.rand")],
     ["Faithfulness", "Runs with top-8 ≥ 2 × random-8 (%)", k("h3.frac"), "≥ 80 (H3)"],
     ["Plausibility", "Precision@10, AD-signature (%)", k("expl.prec10"), "chance " + k("expl.chance")],
     ["Plausibility", "AUROC vs AD-signature", k("expl.auroc"), "0.50 (H4)"],
     ["Stability", "Spearman ρ across folds/seeds", k("expl.stab"), "—"]]),
  t6: () => table("**Table 6** Neurologist-vetted rules: guideline versus learned threshold and effect of switching the rule off.",
    ["Rule", "Guideline T₁", "Learned T₁", "n active", "Δ p(true)"],
    RULES.map(([id, name, g]) => [name, g, k(`rule.${id}.t1_learned`), k(`rule.${id}.n_active`), k(`rule.${id}.dprob`)])),
  t7: () => table("**Table 7** External validation in OASIS-3 (trained on ADNI). AUC (%) for CN vs impaired, scored as 1 − P(CN); mean ± SD over seeds, 95% CI in brackets.",
    ["Method", "Observed", "PET withheld", "MRI only", "3-class BACC (observed)"],
    [["MR-XGNN, recalibrated", "MR-XGNN", "recalibrated"], ["Logistic regression, recalibrated", "LogReg", "recalibrated"],
     ["MR-XGNN, no recalibration", "MR-XGNN", "no_recalibration"]].map(([n, m, r]) => [n,
      `${k(`ext.${m}.${r}.observed.auc`)} [${k(`ext.${m}.${r}.observed.auc_ci`)}]`, k(`ext.${m}.${r}.drop_pet.auc`), k(`ext.${m}.${r}.mri_only.auc`), k(`ext.${m}.${r}.observed.bacc`)])),
};
const FIG = {
  fig1: ["fig1_architecture.png", "**Fig. 1** Overview of MR-XGNN. Regional MRI and PET W-scores are nodes of an 82-region graph. Missing PET node embeddings are predicted from the MRI graph. Modalities are fused by region-level gates and processed by graph attention biased by an anatomical prior. Clinically anchored rules add evidence from observed biomarkers."],
};
const RESULT_FIGS = [
  ["fig_robustness.png", "**Fig. 2** Balanced accuracy in ADNI under each missing-modality scenario (mean ± SD over repeats)."],
  ["fig_confusion.png", "**Fig. 3** Confusion matrix of MR-XGNN (observed availability, seed-averaged predictions)."],
  ["fig_deletion.png", "**Fig. 4** Deletion test: drop in true-class probability when the top-k attributed regions or k random regions are reset to normative values."],
  ["fig_top_rois.png", "**Fig. 5** The fifteen most-attributed regions for AD participants; blue marks a-priori AD-signature regions."],
  ["fig_gates.png", "**Fig. 6** Region-level PET reliance (1 − gate) by diagnostic group in participants with PET."],
  ["fig_calibration.png", "**Fig. 7** Reliability diagram of MR-XGNN (observed availability)."],
];
function figure(file, caption, width = 600) {
  const out = [];
  const cands = [FIGDIR && path.join(FIGDIR, file), path.join(__dirname, "../figures", file)].filter(Boolean);
  const f = cands.find(p => fs.existsSync(p));
  if (f) {
    const buf = fs.readFileSync(f), w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
    out.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, children: [new ImageRun({ type: "png", data: buf, transformation: { width, height: Math.round(width * h / w) } })] }));
  } else out.push(P(`[[insert figures/${file}]]`, { align: AlignmentType.CENTER }));
  out.push(new Paragraph({ children: runs(caption, { size: 20 }), spacing: { before: 60, after: 240 } }));
  return out;
}

const kids = [];
kids.push(new Paragraph({ spacing: { after: 240 }, children: runs(C.title, { bold: true, size: 30 }) }));
kids.push(P(C.authors, { align: AlignmentType.LEFT, run: { size: 22 } }));
C.affils.forEach(a => kids.push(P(a, { align: AlignmentType.LEFT, run: { size: 18 } })));
kids.push(P("**Running title:** " + C.running, { align: AlignmentType.LEFT, run: { size: 20 } }));
kids.push(H1("Abstract")); kids.push(P(C.abstract));
kids.push(P("**Keywords** " + C.keywords, { align: AlignmentType.LEFT }));
for (const s of C.sections) {
  kids.push(H1(s.h));
  for (const b of s.body) {
    if (typeof b === "string") {
      if (b.startsWith("- ")) kids.push(P("•  " + b.slice(2), { indent: { left: 360, hanging: 240 } }));
      else kids.push(P(b));
    }
    else if (b.h2) kids.push(H2(b.h2));
    else if (b.eq) kids.push(EQ(b.eq, b.n));
    else if (b.table) kids.push(...TABLES[b.table]());
    else if (b.figure) kids.push(...figure(...FIG[b.figure]));
  }
  if (s.h === "Results") RESULT_FIGS.forEach(([f, c]) => kids.push(...figure(f, c, 480)));
}
kids.push(H1("Information Sharing Statement")); kids.push(P(C.iss));
for (const [h, t] of C.declarations) { kids.push(H2(h)); kids.push(P(t)); }
kids.push(H1("References"));
const cited = [...used].map(n => REFS[n - 1]).sort((a, b) => a.apa.localeCompare(b.apa, "en", { sensitivity: "base" }));
cited.forEach(r => kids.push(new Paragraph({ children: runs(r.apa.replace(/\[\[/g, "[[").replace(/<|>/g, ""), { size: 20 }), spacing: { after: 100, line: 360 }, indent: { left: 480, hanging: 480 } })));
const doc = new Document({ styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
    lineNumbers: { countBy: 1, restart: "continuous" } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children: kids }] });
Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(OUT, b);
  const unused = REFS.map((r, i) => i + 1).filter(n => !used.has(n));
  console.log(`wrote ${OUT}; numbers filled: ${Object.keys(NUM).length ? "yes" : "no"}; unfilled keys: ${missing.size}; refs cited ${used.size}/${REFS.length}; uncited: ${unused.join(",") || "none"}`);
  if (missing.size && Object.keys(NUM).length) console.log("MISSING KEYS:", [...missing].join(" | "));
});
