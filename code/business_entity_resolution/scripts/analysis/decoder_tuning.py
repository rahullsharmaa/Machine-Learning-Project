"""Offline decoder tuning on the held-out predictions (work/valid_preds.npz).

Compares the production decoder (owner constraint + expected-F0.5 prefix) with probability
re-calibration variants, tuned 2-fold over the fold-0 entities so every number is out-of-sample.
Light on memory (~1.5 GB): safe to run while a training step is alive, unlike matrix-level experiments.

Usage: decoder_tuning.py <valid_preds.npz> <train_ground_truth.tsv> <train_source1.parquet>
"""
import sys, time, zlib
import numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "src"))
from decode import argmax_filter, expected_f05_decode          # noqa: E402
from evaluate import macro_f05, truth_from_frame                 # noqa: E402
from data import read_ground_truth                               # noqa: E402
from pipeline import entity_fold                                 # noqa: E402

npz, gt_path, s1_path = sys.argv[1:4]
z = np.load(npz)
p2, y, g1, g23, s1, s23 = z["p2"].astype(np.float64), z["y"], z["g1"], z["g23"], z["s1"], z["s23"]
truth = truth_from_frame(read_ground_truth(gt_path))
all_ids = pd.read_parquet(s1_path, columns=["entity_id"])["entity_id"].to_numpy(dtype=object)
val_entities = all_ids[entity_fold(all_ids) == 0]
half = np.array([zlib.crc32(("h" + e).encode()) % 2 for e in val_entities], np.int8)
ent_half = dict(zip(val_entities, half))
row_half = np.array([ent_half[e] for e in s1], np.int8)


def score(keep, ents):
    lists = {}
    for a, b in zip(s1[keep], s23[keep]):
        lists.setdefault(a, set()).add(b)
    return macro_f05(lists, truth, ents)


def decode(p, min_p=0.02):
    pp = np.where(argmax_filter(p, g23), p, 0)
    return expected_f05_decode(pp, g1, min_p=min_p)


t = time.time()
base = decode(p2)
print(f"baseline: {score(base, val_entities):.5f}  (halves {score(base, val_entities[half == 0]):.5f} / {score(base, val_entities[half == 1]):.5f})  {time.time()-t:.0f}s", flush=True)

# ---- power calibration p^a, tuned per half on the other half
alphas = [0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5]
tab = {}
for a in alphas:
    k = decode(np.clip(p2, 0, 1) ** a)
    tab[a] = (score(k, val_entities[half == 0]), score(k, val_entities[half == 1]))
    print(f"  alpha={a:<5} halves {tab[a][0]:.5f} / {tab[a][1]:.5f}", flush=True)
best0 = max(alphas, key=lambda a: tab[a][1])   # tuned on half 1, applied to half 0
best1 = max(alphas, key=lambda a: tab[a][0])
print(f"power calibration (cross-tuned): {0.5 * (tab[best0][0] + tab[best1][1]):.5f}  alphas {best0}/{best1}")

# ---- isotonic calibration, fitted on one half of the pairs, applied to the other
from sklearn.isotonic import IsotonicRegression
pc = p2.copy()
for h in (0, 1):
    fit_m = row_half != h
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p2[fit_m], y[fit_m])
    pc[~fit_m] = iso.predict(p2[~fit_m])
k = decode(pc)
print(f"isotonic calibration (2-fold): {score(k, val_entities):.5f}")

# ---- min_p sensitivity
for mp in (0.05, 0.1):
    print(f"min_p={mp}: {score(decode(p2, min_p=mp), val_entities):.5f}")

# ---- owner-constraint margin: let a record be claimed by a second S1 within `m` of the best
for m in (0.05, 0.15):
    pp = np.where(argmax_filter(p2, g23, margin=m), p2, 0)
    print(f"argmax margin={m}: {score(expected_f05_decode(pp, g1), val_entities):.5f}")
