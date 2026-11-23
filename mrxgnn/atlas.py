"""Region-of-interest definitions.

Default atlas: FreeSurfer Desikan-Killiany (34 cortical ROIs per hemisphere) + 7 subcortical
ROIs per hemisphere = 82 nodes. This matches the regional tables ADNI distributes
(UCSF FreeSurfer; UC Berkeley amyloid-PET regional SUVRs), so no image processing is needed.

AD_SIGNATURE lists regions with well-established early AD involvement (medial temporal
atrophy, temporoparietal/posterior-cingulate hypometabolism). It is used ONLY to score the
anatomical plausibility of explanations, never as a model input.
Refs: Jack et al. 2015 (AD-signature cortical thickness), Landau et al. 2011 (FDG meta-ROI),
Braak & Braak 1991.
"""

DK_CORTICAL = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus", "entorhinal",
    "fusiform", "inferiorparietal", "inferiortemporal", "isthmuscingulate", "lateraloccipital",
    "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal",
    "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate",
    "rostralmiddlefrontal", "superiorfrontal", "superiorparietal", "superiortemporal",
    "supramarginal", "frontalpole", "temporalpole", "transversetemporal", "insula",
]
SUBCORTICAL = ["hippocampus", "amygdala", "thalamus", "caudate", "putamen", "pallidum", "accumbens"]

ROIS = [f"{h}_{r}" for h in ("lh", "rh") for r in DK_CORTICAL + SUBCORTICAL]

_SIG = {"entorhinal", "parahippocampal", "hippocampus", "amygdala", "inferiorparietal",
        "precuneus", "posteriorcingulate", "isthmuscingulate", "middletemporal",
        "inferiortemporal", "fusiform", "temporalpole"}
AD_SIGNATURE = [r for r in ROIS if r.split("_", 1)[1] in _SIG]


def hemisphere_pairs(rois=ROIS):
    """Index pairs (lh_x, rh_x) — used as homotopic edges in the anatomical prior."""
    idx = {r: i for i, r in enumerate(rois)}
    return [(idx[r], idx["rh_" + r[3:]]) for r in rois if r.startswith("lh_") and "rh_" + r[3:] in idx]
