"""Checks that no subject appears in more than one partition of any fold, and that every subject
is tested exactly once per seed. Put its output in the supplementary material.
  python scripts/check_leakage.py results/adni/full/splits.csv
"""
import sys
import pandas as pd

S = pd.read_csv(sys.argv[1])
bad = 0
for (seed, fold), g in S.groupby(["seed", "fold"]):
    sets = {p: set(x.subject_id) for p, x in g.groupby("part")}
    for p1, p2 in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = sets.get(p1, set()) & sets.get(p2, set())
        if inter:
            bad += 1
            print(f"LEAK seed={seed} fold={fold} {p1}&{p2}: {len(inter)} subjects, e.g. {list(inter)[:3]}")
for seed, g in S[S.part == "test"].groupby("seed"):
    c = g.subject_id.value_counts()
    if (c != 1).any():
        bad += 1
        print(f"seed={seed}: {int((c != 1).sum())} subjects tested more than once")
print("OK: no subject overlap between partitions; each subject tested once per seed" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)
