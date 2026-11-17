const fs = require("fs");
const { Document, Packer, Paragraph, TextRun } = require("docx");
const F = "Times New Roman";
const p = (t, o = {}) => new Paragraph({ spacing: { after: 160, line: 300 }, children: [].concat(t).map(x => typeof x === "string" ? new TextRun({ text: x, font: F, size: 22 }) : x), ...o });
const b = t => new TextRun({ text: t, font: F, size: 22, bold: true });
const y = t => new TextRun({ text: t, font: F, size: 22, highlight: "yellow" });
const doc = new Document({ sections: [{ children: [
  p([y("[Date]")]),
  p("The Editor-in-Chief\nComputer Methods and Programs in Biomedicine"),
  p("Dear Editor,"),
  p(["We submit the original research article ", b("“Missing-modality-robust region-graph fusion of MRI, amyloid PET and clinical data with evaluated rule-based explanations for Alzheimer’s disease: a pre-registered study with external validation”"), " for consideration in Computer Methods and Programs in Biomedicine."]),
  p([b("Why this journal. "), "The paper contributes a computational method together with openly released, documented software (preprocessing adapters, model, evaluation and figure scripts, configurations and split definitions), which matches the journal’s aim of publishing computing methodology and demonstrable software for biomedical research. It also extends a line of multimodal Alzheimer’s disease work the journal has published, including MACFNet (Comput Methods Programs Biomed 2024;254:108259)."]),
  p([b("What is known. "), "Multimodal MRI–PET classifiers for Alzheimer’s disease are usually trained and tested on participants who have every modality, validated only within one cohort, and explained with saliency or attention maps whose faithfulness is not tested. Reported accuracies vary widely with evaluation choices such as split level and the use of MMSE as an input."]),
  p([b("What this study adds. "), "(1) A region-graph model that remains usable when amyloid PET or CSF is missing, by predicting PET node embeddings from the MRI graph and training with modality dropout; (2) guideline-anchored rules that switch off when their input is absent and whose thresholds and effects are reported; (3) quantitative evaluation of explanations (deletion faithfulness, anatomical plausibility, stability); and (4) a protocol pre-registered before data access (https://doi.org/10.17605/OSF.IO/J4KG6), with participant-level repeated cross-validation in ADNI (n = ", y("[N]"), "), no cognitive scores as inputs, and external validation in OASIS-3."]),
  p([b("Key results. "), y("[Two sentences from paper_numbers.json: three-class balanced accuracy with CI, change when PET is withheld, OASIS-3 AUC.]")]),
  p("This manuscript has not been published and is not under consideration elsewhere, and all authors have approved its submission. Data were obtained from ADNI and OASIS-3 under their data use agreements; the authors have no competing interests. The use of generative AI tools is declared in the manuscript."),
  p(["Suggested reviewers (no conflicts of interest): ", y("[3–5 names, affiliations, institutional e-mails — independent researchers in multimodal AD machine learning]")]),
  p("Yours sincerely,"),
  p([y("[Corresponding author name, department, institution, institutional e-mail, ORCID]")]),
] }] });
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(process.argv[2], buf); console.log("ok"); });
