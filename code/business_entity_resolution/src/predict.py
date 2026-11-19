"""Inference: produce matching_results.tsv and candidate_pairs.tsv for a split.

Runs the same sharded, memory-bounded stages as training (blocking -> stage-0 filter ->
pruning -> pair features), then streams the feature shards through the stage-1 and
stage-2 models and decodes one match list per Source 1 entity.
"""
from __future__ import annotations

import gc
import os
import time
from typing import Dict, Set

import lightgbm as lgb
import numpy as np
import pandas as pd

from blocking import BlockingConfig
from decode import argmax_filter, build_lists, expected_f05_decode, write_submission
from features import KEY_COLS, STAGE2_COLS, add_stage2_features, record_keys
from pipeline import ModelBundle, PruneConfig, q16
from runner import Runner, log


def decode(p: np.ndarray, i1: np.ndarray, i23: np.ndarray, strategy: str) -> np.ndarray:
    """strategy examples: 'thr=0.5 argmax=True', 'expectedF05 argmax=True'."""
    parts = dict(kv.split("=") for kv in strategy.split() if "=" in kv)
    am = parts.get("argmax", "False") == "True"
    if strategy.startswith("expectedF05"):
        pp = np.where(argmax_filter(p, i23), p, 0) if am else p
        return expected_f05_decode(pp, i1)
    thr = float(parts["thr"])
    keep = p >= thr
    if am:
        keep &= argmax_filter(p, i23)
    return keep


def run(data_dir: str, split: str, cache_dir: str, model_dir: str, lexicon: str, out_dir: str, work_dir: str,
        n_jobs: int = 6, strategy: str | None = None, save_scores: bool = False, shard_rows: int = 2_200_000):
    os.makedirs(out_dir, exist_ok=True)
    mb = ModelBundle.load(model_dir)
    best = mb.decode.get("best", "p2 thr=0.5 argmax=True")
    strategy = strategy or best
    use_p1 = strategy.startswith("p1 ")
    strategy = strategy.split(" ", 1)[1] if strategy.startswith(("p1 ", "p2 ")) else strategy
    log(f"decode strategy: {strategy} (stage {'1' if use_p1 else '2'} probabilities)")

    r = Runner(data_dir, cache_dir, work_dir, model_dir, lexicon, split=split, n_jobs=n_jobs, shard_rows=shard_rows)
    if mb.blocking:
        r.bcfg = BlockingConfig(**mb.blocking)
    if mb.prune:
        r.pcfg = PruneConfig(**mb.prune)
    countries = r.countries()                      # open set: whatever labels occur in this split
    log(f"countries in {split}: {countries}")
    r.step_block()
    r.step_score0()
    r.step_feats()

    all_s1 = r.all_s1_ids()
    matches: Dict[str, Set[str]] = {}
    cands: Dict[str, Set[str]] = {}
    scores = []
    cols = mb.feature_cols
    for c in countries:
        t = time.time()
        n1 = len(r.s1_ids(c))
        meta = r.read_meta(c, ["i1", "i23", "p0"])
        i1 = meta["i1"].to_numpy(np.int64)
        i23 = meta["i23"].to_numpy(np.int64)
        p0 = meta["p0"].to_numpy(np.float32)
        del meta
        # stage 1 (streamed)
        p1 = np.concatenate([mb.stage1.predict(q16(df[cols].to_numpy(np.float32)), num_threads=12).astype(np.float32)
                             for df in r.iter_feat_batches(c, cols)])
        # stage 2: competition features from the whole country, then streamed scoring
        keys = record_keys(*r.partition(c, columns=KEY_COLS))
        s2 = add_stage2_features(pd.DataFrame(index=range(len(i1))), p1, i1, i23, keys)
        del keys
        p2 = np.empty(len(p1), np.float32)
        # use the stage-2 inputs the saved model was trained with (older bundles carry fewer columns)
        s2cols = [x for x in mb.stage2.feature_name() if x not in cols] or STAGE2_COLS
        off = 0
        for df in r.iter_feat_batches(c, cols):
            n = len(df)
            for col in s2cols:
                df[col] = s2[col].to_numpy(np.float32)[off:off + n]
            p2[off:off + n] = mb.stage2.predict(q16(df[cols + s2cols].to_numpy(np.float32)), num_threads=12)
            off += n
        del s2
        p = p1 if use_p1 else p2
        keep = decode(p, i1, i23, strategy)
        s1_ids = r.s1_ids(c)
        s23_ids = r.s23_ids(c)
        matches.update(build_lists(keep, s1_ids, s23_ids, i1, i23))
        cands.update(build_lists(np.ones(len(i1), bool), s1_ids, s23_ids, i1, i23))
        log(f"  {c}: {len(i1):,} candidates ({len(i1)/n1:.2f} per S1); {keep.sum():,} matches for {len(set(i1[keep])):,} entities; "
            f"mean matches/S1 {keep.sum()/n1:.3f}; scored in {time.time()-t:.0f}s")
        if save_scores:
            scores.append(pd.DataFrame({"s1": s1_ids[i1], "s23": s23_ids[i23], "p1": p1, "p2": p2, "p0": p0}))
        del i1, i23, p0, p1, p2, keep
        gc.collect()
    write_submission(os.path.join(out_dir, "matching_results.tsv"), all_s1, matches, "matched_entity_ids")
    write_submission(os.path.join(out_dir, "candidate_pairs.tsv"), all_s1, cands, "candidate_entity_ids")
    n_c = sum(len(v) for v in cands.values())
    n_m = sum(len(v) for v in matches.values())
    log(f"wrote {out_dir}: {n_m:,} matches, {n_c:,} candidates (density {n_c/len(all_s1):.2f} per S1), "
        f"singletons predicted {sum(1 for e in all_s1 if not matches.get(e))/len(all_s1):.4f}")
    if save_scores and scores:
        pd.concat(scores, ignore_index=True).to_parquet(os.path.join(out_dir, "pair_scores.parquet"), index=False)
    if r.labelled:
        from evaluate import macro_f05, truth_from_frame
        truth = truth_from_frame(r.gt)
        log(f"  {split} macro-F0.5 over all {len(all_s1):,} Source 1 entities: {macro_f05(matches, truth, all_s1):.5f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None, help="scratch directory for the per-shard intermediate files")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--shard-rows", type=int, default=2_200_000)
    ap.add_argument("--save-scores", action="store_true")
    a = ap.parse_args()
    run(a.data, a.split, a.cache, a.models, a.lexicon, a.out, a.work or os.path.join(a.out, "work"),
        strategy=a.strategy, save_scores=a.save_scores, shard_rows=a.shard_rows)
