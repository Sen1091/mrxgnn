const fs = require("fs");
const { Document, Packer, Paragraph, TextRun } = require("docx");
const F = "Times New Roman";
const p = t => new Paragraph({ spacing: { after: 160, line: 300 }, children: [].concat(t).map(x => typeof x === "string" ? new TextRun({ text: x, font: F, size: 22 }) : x) });
const b = t => new TextRun({ text: t, font: F, size: 22, bold: true });
const y = t => new TextRun({ text: t, font: F, size: 22, highlight: "yellow" });
const doc = new Document({ sections: [{ children: [
  p([y("[Date]")]), p("The Editors-in-Chief, Neuroinformatics"), p("Dear Editors,"),
  p(["We submit the Original Article ", b("“What do graph structure, amyloid PET and knowledge rules add to multimodal Alzheimer’s disease classification? A pre-registered evaluation with faithfulness-tested explanations”"), " for consideration in Neuroinformatics."]),
  p("Multimodal deep-learning models for Alzheimer’s disease frequently report near-perfect accuracy, but they are rarely compared with strong simple baselines under leakage-controlled protocols, and their explanations are seldom tested. We pre-registered (https://doi.org/10.17605/OSF.IO/J4KG6) an evaluation of a graph neural network that contains the components recent work proposes: region-level fusion of MRI and amyloid PET, prediction of missing PET from MRI, an anatomical prior, and guideline-anchored rules."),
  p(["In 2,236 ADNI participants, with repeated participant-level cross-validation and no cognitive scores as inputs, the model matched but did not beat logistic regression (three-class balanced accuracy 63.0% vs 62.6%). None of the architectural components added accuracy, and amyloid PET added under one point once MRI, CSF and APOE were known. Its explanations, however, were faithful in every run and anatomically plausible. We report the negative findings in full, including a pre-registered hypothesis that was not supported."]),
  p("We believe the paper suits Neuroinformatics because its contribution is methodological and reproducible: a pre-registered protocol, a released, tested software pipeline for ADNI’s current ADNIMERGE2 tables, split definitions, and explanation-evaluation tools that others can apply to their own models."),
  p("The manuscript has not been published and is not under consideration elsewhere, and all authors approved its submission. Data were used under the ADNI Data Use Agreement. The authors declare no competing interests, and the use of generative AI is declared in the manuscript."),
  p(["Suggested reviewers: ", y("[3–5 independent researchers in neuroimaging machine learning, with institutional e-mails]")]),
  p("Yours sincerely,"), p([y("[Corresponding author name, department, institution, institutional e-mail, ORCID]")]),
] }] });
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(process.argv[2], buf); console.log("ok"); });
