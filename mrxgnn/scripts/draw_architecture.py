"""Fig. 1 - MR-XGNN architecture schematic (vector PDF + 300-dpi PNG).
  python scripts/draw_architecture.py --out figures/fig1_architecture
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figures/fig1_architecture"); a = ap.parse_args()
BLUE, ORANGE, AQUA, YEL, VIO, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 57); ax.axis("off")


def box(x, y, w, h, title, sub="", col=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=col + "1f", ec=col, lw=1.1))
    nt = title.count("\n") + 1
    if sub:
        ax.text(x + w / 2, y + h - 1.0, title, ha="center", va="top", fontsize=6.6, weight="bold", color=INK, linespacing=1.1)
        ax.text(x + w / 2, y + h - 1.6 - 2.6 * nt, sub, ha="center", va="top", fontsize=4.9, color=INK2, linespacing=1.2)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=6.6, weight="bold", color=INK)


def arrow(x1, y1, x2, y2, dashed=False, col=INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.9, color=col,
                                 linestyle=(0, (3, 2)) if dashed else "-"))


# inputs
box(1, 43, 15, 9, "T1 MRI", "82 ROI W-scores\n(FreeSurfer DK)", BLUE)
box(1, 29, 15, 9, "PET", "82 ROI SUVR W-scores\n(may be missing)", ORANGE)
box(1, 5, 15, 13, "Clinical", "age, sex, education,\nAPOE4, CSF A\u03b242, p-tau\n+ availability masks\n(no MMSE / CDR)", AQUA)
# node encoders
box(20, 43, 14, 9, "MRI node enc.", "h_m,r + ROI emb.", BLUE)
box(20, 29, 14, 9, "PET node enc.", "h_p,r + ROI emb.", ORANGE)
box(20, 16, 14, 8.5, "Hallucinator", "PET nodes predicted\nfrom MRI node graph\n(used if PET missing)", VIO)
arrow(16, 47.5, 20, 47.5); arrow(16, 33.5, 20, 33.5)
arrow(31, 24.5, 31, 29, dashed=True, col=VIO)
# gated fusion
box(37.5, 32, 19, 14, "Region-level\ngated fusion", "g_r = \u03c3(W[h_m; h_p; m])\nh_r = g_r h_m + (1\u2212g_r) h_p", INK2)
arrow(34, 47.5, 37.5, 42); arrow(34, 33.5, 37.5, 36)
# graph attention
box(59, 32, 18, 14, "Graph attention \u00d7 L", "attention + \u03bb log A_prior\nA_prior: structural covariance\n+ homotopic edges", BLUE)
arrow(56.5, 39, 59, 39)
box(81, 32, 17, 14, "Attention readout", "node weights w_r\n(ROI importance)", BLUE)
arrow(77, 39, 81, 39)
box(20, 3, 14, 8.5, "Clinical enc.", "MLP[x\u2299m ; m]", AQUA)
arrow(16, 8, 20, 7.5, col=AQUA)
# rules
box(38, 7, 39, 14.5, "Clinically anchored differentiable rules",
    "\u03b4_k = a_k \u03c3((x \u2212 T1_k)/\u03c4_k) + b_k ReLU(x \u2212 T2_k)\nactive only if x observed; T_k = T_k(guideline) + \u0394T_k\n"
    "penalty \u03bb||\u0394T||\u00b2 keeps thresholds anchored\nLLM-drafted, clinician-vetted", YEL)
arrow(16, 14, 38, 14, col=AQUA)
# head
box(81, 12, 17, 13, "Classifier", "logits + \u03a3_k \u03b4_k w_k\n\u2192 CN / MCI / AD", INK2)
arrow(89.5, 32, 89.5, 25)
arrow(77, 16, 81, 18, col=YEL)
ax.plot([34, 93], [4.5, 4.5], color=AQUA, lw=0.9); arrow(93, 4.5, 93, 12, col=AQUA)
# training / explanations
box(38, 48.5, 60, 7, "Training and evaluation", "modality dropout (PET p=0.5, CSF p=0.3) + full\u2192dropped KL distillation + hallucination loss;\nevaluated explanations: IG ROI maps, deletion fidelity, rule ablation, plausibility, stability", VIO)
fig.savefig(a.out + ".png", bbox_inches="tight"); fig.savefig(a.out + ".pdf", bbox_inches="tight")
print("saved", a.out)
