"""End-to-end matching pipeline: blocking -> cascade pruning -> pairwise model -> decoding.

Stages (all country-agnostic; run per country partition):
  0. blocking (blocking.py) gives candidate pairs with TF-IDF cosines / ranks.
  1. a light LightGBM "stage-0" model on blocking-only features prunes the candidate
     set (keeps ~99.9% of true pairs) so that the expensive string features are computed
     for a bounded number of pairs.
  2. full pairwise features (features.py) + LightGBM stage-1 model.
  3. stage-2 LightGBM re-scores each pair with context features derived from stage-1
     probabilities of the competing pairs (same S1 entity / same S2-S3 record).
  4. decoding (decode.py) turns probabilities into the per-entity match lists.
"""
from __future__ import annotations

import gc
import json
import os
import time
import zlib
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from blocking import RANK_COLS, BlockingConfig, block_partition
from features import (STAGE2_COLS, _group_count, _group_max, _group_second_max, add_stage2_features,
                      compute_features)

STAGE0_BASE = ["cos_name", "cos_addr", "cos_blend"] + RANK_COLS
_COS3 = ["cos_name", "cos_addr", "cos_blend"]


def rank_within_sorted(keys: np.ndarray, scores: np.ndarray, n_groups: int):
    """Rank of every row inside its key group by decreasing score (0 = best) plus the group's
    best and second-best score. Uses one float64 argsort on a composite key (scores in [0, 1]).
    Returns (rank int16, best[n_groups] float32, second[n_groups] float32).
    Memory-lean: only the composite key and its argsort are 8-byte arrays; everything else is
    <= 4 bytes/row so a 250M-row country fits comfortably next to the candidate arrays."""
    n = len(keys)
    comp = keys.astype(np.float64)
    comp *= 2.0
    comp -= scores  # scores in [0, 1] -> composite sorts by key asc, score desc
    order = np.argsort(comp, kind="stable")
    del comp
    k = keys[order]
    brk = np.flatnonzero(k[1:] != k[:-1]) + 1                     # first row of every group but the first
    starts = np.concatenate([[0], brk]).astype(np.int64)
    sizes = np.diff(np.concatenate([starts, [n]]))
    first_keys = k[starts] if n else k
    del k, brk
    pos = np.arange(n, dtype=np.int32)
    pos -= np.repeat(starts.astype(np.int32), sizes)
    rank = np.empty(n, np.int16)
    rank[order] = np.minimum(pos, 32000).astype(np.int16)
    best = np.zeros(n_groups, np.float32)
    second = np.zeros(n_groups, np.float32)
    if n:
        best[first_keys] = scores[order[starts]]
        has2 = sizes >= 2
        second[first_keys[has2]] = scores[order[starts[has2] + 1]]
    return rank, best, second


class Stage0Table:
    """Cheap post-blocking features for every candidate pair of one partition.

    All group statistics (per S1 entity / per S2-S3 record) are precomputed as small
    per-group arrays; the row-level feature matrix is materialised lazily with
    :meth:`rows` so that partitions with >100M candidate pairs fit in memory.
    """

    def __init__(self, cand: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame):
        self.i1 = cand["i1"].to_numpy()
        self.i23 = cand["i23"].to_numpy()
        n1, n23 = len(s1), len(s23)
        self.base = {c: cand[c].to_numpy() for c in STAGE0_BASE}
        i1 = self.i1.astype(np.int64)
        i23 = self.i23.astype(np.int64)
        self.cnt_a = np.bincount(i1, minlength=n1).astype(np.float32)
        self.cnt_b = np.bincount(i23, minlength=n23).astype(np.float32)
        self.rank_a: Dict[str, np.ndarray] = {}
        self.rank_b: Dict[str, np.ndarray] = {}
        self.best_a: Dict[str, np.ndarray] = {}
        self.best_b: Dict[str, np.ndarray] = {}
        self.sec_a: Dict[str, np.ndarray] = {}
        self.sec_b: Dict[str, np.ndarray] = {}
        for c in _COS3:
            v = self.base[c]
            self.rank_a[c], self.best_a[c], self.sec_a[c] = rank_within_sorted(i1, v, n1)
            self.rank_b[c], self.best_b[c], self.sec_b[c] = rank_within_sorted(i23, v, n23)
        del i1, i23
        # reference-side ambiguity and record shape
        self.side1 = {
            "dup_name1": _value_counts(s1["name_nospace"]),
            "dup_addr1": _value_counts(s1["addr"], zero_empty=True),
            "dup_street1": _value_counts(s1["addr_street"], zero_empty=True),
            "ntok_name1": _ntok(s1["name_core"]),
            "ntok_addr1": _ntok(s1["addr"]),
        }
        self.side23 = {
            "empty_addr23": (s23["addr"].to_numpy(dtype=object) == "").astype(np.float32),
            "ntok_name23": _ntok(s23["name_core"]),
            "ntok_addr23": _ntok(s23["addr"]),
            "nonlatin23": s23["is_nonlatin"].to_numpy(np.float32) if "is_nonlatin" in s23 else np.zeros(n23, np.float32),
            "domain23": s23["is_domain"].to_numpy(np.float32) if "is_domain" in s23 else np.zeros(n23, np.float32),
        }
        self.n = len(self.i1)
        self.cols = list(self.rows(slice(0, 1)).columns)

    def rows(self, idx) -> pd.DataFrame:
        i1 = self.i1[idx].astype(np.int64)
        i23 = self.i23[idx].astype(np.int64)
        F: Dict[str, np.ndarray] = {c: self.base[c][idx].astype(np.float32) for c in STAGE0_BASE}
        F["n_nets"] = sum((F[c] < 99).astype(np.float32) for c in RANK_COLS)
        F["cnt_a"] = self.cnt_a[i1]
        F["cnt_b"] = self.cnt_b[i23]
        for c in _COS3:
            v = F[c]
            F[f"{c}_gap_a"] = self.best_a[c][i1] - v
            F[f"{c}_other_a"] = self.sec_a[c][i1]
            F[f"{c}_gap_b"] = self.best_b[c][i23] - v
            F[f"{c}_other_b"] = self.sec_b[c][i23]
            F[f"{c}_rk_a"] = self.rank_a[c][idx].astype(np.float32)
            F[f"{c}_rk_b"] = self.rank_b[c][idx].astype(np.float32)
        for k, arr in self.side1.items():
            F[k] = arr[i1]
        for k, arr in self.side23.items():
            F[k] = arr[i23]
        return pd.DataFrame(F)

    def predict(self, booster: lgb.Booster, cols: List[str], step: int = 5_000_000, num_threads: int = 12) -> np.ndarray:
        p = np.empty(self.n, np.float32)
        for s in range(0, self.n, step):
            p[s:s + step] = booster.predict(self.rows(slice(s, s + step))[cols], num_threads=num_threads)
        return p


def _value_counts(col: pd.Series, zero_empty: bool = False) -> np.ndarray:
    vals = col.to_numpy(dtype=object)
    codes, uniq = pd.factorize(vals, sort=False)
    cnt = np.bincount(codes).astype(np.float32)[codes]
    if zero_empty:
        cnt[vals == ""] = 0
    return cnt


def _ntok(col: pd.Series) -> np.ndarray:
    return col.str.count(" ").to_numpy(np.float32) + (col.to_numpy(dtype=object) != "")


def stage0_features(cand: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame:
    """Materialised stage-0 feature frame (small partitions / tests only)."""
    return Stage0Table(cand, s1, s23).rows(slice(0, len(cand)))


def entity_fold(ids: np.ndarray, n_folds: int = 5) -> np.ndarray:
    return np.array([zlib.crc32(e.encode()) % n_folds for e in ids], dtype=np.int8)


def labels_for(cand: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame, owner: Dict[str, str]) -> np.ndarray:
    a = s1["entity_id"].to_numpy(dtype=object)[cand["i1"].to_numpy()]
    b = s23["entity_id"].to_numpy(dtype=object)[cand["i23"].to_numpy()]
    return np.fromiter((1 if owner.get(y) == x else 0 for x, y in zip(a, b)), dtype=np.int8, count=len(a))


def owner_map(gt: pd.DataFrame) -> Dict[str, str]:
    owner: Dict[str, str] = {}
    for e, m in zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].values):
        if m:
            for x in m.split(","):
                if x:
                    owner[x] = e
    return owner


@dataclass
class PruneConfig:
    # Operating point chosen on the full training set (entity-held-out sweep): overall density
    # 18.6 candidates per S1 (India 20.8, US 17.1) while keeping 99.99% of the recalled true pairs.
    p0_min: float = 0.0002      # keep pairs with stage-0 prob >= p0_min
    top_b: int = 3              # always keep the best top_b S1 candidates of every S2/S3 record
    top_a: int = 3              # always keep the best top_a S2/S3 candidates of every S1 entity
    max_per_a: int = 60         # hard cap per S1 entity (by stage-0 prob)


def prune_candidates(cand: pd.DataFrame, p0: np.ndarray, cfg: PruneConfig, ranks=None) -> np.ndarray:
    """Boolean keep-mask. ``ranks`` = (rank_a, rank_b) by stage-0 probability can be passed
    to evaluate several configurations without re-sorting."""
    if ranks is None:
        ranks = prune_ranks(cand, p0)
    rank_a, rank_b = ranks
    keep = p0 >= cfg.p0_min
    keep |= rank_b < cfg.top_b
    keep |= rank_a < cfg.top_a
    keep &= rank_a < cfg.max_per_a
    return keep


def prune_ranks(cand: pd.DataFrame, p0: np.ndarray):
    i1 = cand["i1"].to_numpy(np.int32)
    i23 = cand["i23"].to_numpy(np.int32)
    p = np.clip(p0, 0.0, 1.0).astype(np.float32, copy=False)
    rank_a, _, _ = rank_within_sorted(i1, p, int(i1.max()) + 1 if len(i1) else 0)
    gc.collect()
    rank_b, _, _ = rank_within_sorted(i23, p, int(i23.max()) + 1 if len(i23) else 0)
    gc.collect()
    return rank_a, rank_b


LGB_PARAMS_S0 = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                     bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8)
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, max_bin=255)


def q16(a) -> np.ndarray:
    """float32 values rounded to float16 precision. The training matrix is stored as float16 (half the
    memory); applying the same rounding at inference keeps the bin assignment identical."""
    return np.asarray(a, dtype=np.float32).astype(np.float16).astype(np.float32)


def predict_rows(booster: lgb.Booster, X: np.ndarray, num_threads: int = 8, batch: int = 500_000) -> np.ndarray:
    """Batched prediction over a (float16) matrix without a full float32 copy."""
    out = np.empty(len(X), np.float32)
    for s in range(0, len(X), batch):
        out[s:s + batch] = booster.predict(X[s:s + batch].astype(np.float32), num_threads=num_threads)
    return out


class ColumnSlice(lgb.Sequence):
    """Row-batch view of the first ``n_cols`` columns of a float16/float32 matrix. Lets LightGBM bin a
    column subset without materialising a contiguous (multi-GB) float64 copy of the sub-matrix."""

    def __init__(self, X: np.ndarray, n_cols: int, batch_size: int = 250_000):
        self.X, self.n_cols, self.batch_size = X, n_cols, batch_size

    def __getitem__(self, idx):
        return self.X[idx, :self.n_cols].astype(np.float64)   # LightGBM's Sequence protocol wants doubles

    def __len__(self):
        return len(self.X)


def make_dataset(X: np.ndarray, y: np.ndarray, feature_name, n_cols=None) -> lgb.Dataset:
    """Constructed (binned) LightGBM dataset of ``X[:, :n_cols]``; raw data is released after binning."""
    n_cols = X.shape[1] if n_cols is None else n_cols
    data = ColumnSlice(X, n_cols)
    ds = lgb.Dataset(data, label=y, feature_name=list(feature_name), free_raw_data=True,
                     params={"max_bin": 255, "verbose": -1})
    ds.construct()
    gc.collect()
    return ds


def fit_lgb(X, y: np.ndarray, params: dict, rounds: int, X_val=None, y_val=None, log=print,
            feature_name=None) -> lgb.Booster:
    """``X`` may be a DataFrame, a float32 matrix (then pass ``feature_name``) or an already
    constructed ``lgb.Dataset`` (``y`` is then only used for the row count in the log)."""
    if isinstance(X, lgb.Dataset):
        dtrain = X
    else:
        kw = {"feature_name": list(feature_name)} if feature_name is not None else {}
        dtrain = lgb.Dataset(X, label=y, free_raw_data=True, **kw)
    valid_sets = [dtrain]
    names = ["train"]
    if X_val is not None:
        valid_sets.append(lgb.Dataset(X_val, label=y_val, reference=dtrain))
        names.append("valid")
    t = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=valid_sets, valid_names=names,
                        callbacks=[lgb.log_evaluation(100)])
    del dtrain
    gc.collect()
    log(f"    trained {rounds} rounds on {len(y):,} rows in {time.time()-t:.0f}s")
    return booster


@dataclass
class ModelBundle:
    stage0: Optional[lgb.Booster] = None
    stage1: Optional[lgb.Booster] = None
    stage2: Optional[lgb.Booster] = None
    feature_cols: List[str] = field(default_factory=list)
    stage0_cols: List[str] = field(default_factory=list)
    decode: Dict = field(default_factory=dict)
    blocking: Dict = field(default_factory=dict)
    prune: Dict = field(default_factory=dict)

    def save(self, d: str) -> None:
        os.makedirs(d, exist_ok=True)
        for name in ["stage0", "stage1", "stage2"]:
            b = getattr(self, name)
            if b is not None:
                b.save_model(os.path.join(d, f"{name}.txt"))
        meta = dict(feature_cols=self.feature_cols, stage0_cols=self.stage0_cols, decode=self.decode,
                    blocking=self.blocking, prune=self.prune)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)

    @classmethod
    def load(cls, d: str) -> "ModelBundle":
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        mb = cls(**meta)
        for name in ["stage0", "stage1", "stage2"]:
            p = os.path.join(d, f"{name}.txt")
            if os.path.exists(p):
                setattr(mb, name, lgb.Booster(model_file=p))
        return mb


def predict_pairs(mb: ModelBundle, feat: pd.DataFrame, i1: np.ndarray, i23: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (p1, p2) for a feature table."""
    p1 = mb.stage1.predict(feat[mb.feature_cols], num_threads=8).astype(np.float32)
    if mb.stage2 is None:
        return p1, p1
    f2 = add_stage2_features(feat, p1, i1, i23)
    p2 = mb.stage2.predict(f2[mb.feature_cols + STAGE2_COLS], num_threads=8).astype(np.float32)
    return p1, p2
