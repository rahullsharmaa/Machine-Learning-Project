"""Turn pair probabilities into per-Source-1 match lists.

Two ideas are used:
  1. Each Source 2/3 record belongs to at most one Source 1 entity, so for every S2/S3
     record only its highest-probability Source 1 candidate may be kept.
  2. The metric is a per-entity F_0.5 that includes singletons.  Given calibrated
     probabilities, the subset maximising the *expected* F_0.5 of an entity is a prefix
     of its candidates sorted by probability; the expectation is computed exactly under
     independence (Poisson-binomial convolution).  A plain global threshold decoder is
     kept as the baseline.
"""
from __future__ import annotations

from typing import Dict, Set

import numpy as np


def argmax_filter(p: np.ndarray, i23: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Boolean mask keeping, per S2/S3 record, only the best-scoring S1 candidate
    (candidates within ``margin`` of the best are also kept)."""
    order = np.lexsort((-p, i23))
    k = i23[order]
    starts = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(k)])
    best = np.repeat(p[order][starts], sizes)
    keep_sorted = p[order] >= best - margin
    keep = np.empty(len(p), bool)
    keep[order] = keep_sorted
    return keep


def threshold_decode(p: np.ndarray, thr: float) -> np.ndarray:
    return p >= thr


def _pb_pmf(probs: np.ndarray) -> np.ndarray:
    d = np.array([1.0])
    for q in probs:
        nd = np.zeros(len(d) + 1)
        nd[:-1] += d * (1 - q)
        nd[1:] += d * q
        d = nd
    return d


def _expected_f05_prefixes(probs: np.ndarray) -> np.ndarray:
    """probs sorted descending. Expected F_0.5 for predicting the top-m, m = 0..n.

    F_0.5 with tp true positives in a prediction of size m and fn misses equals
    1.25*tp / (0.25*(tp+fn) + m); for m = 0 it is 1 iff there are no true matches.
    """
    n = len(probs)
    out = np.zeros(n + 1)
    # suffix pmfs (true matches outside the prefix)
    suffix = [None] * (n + 1)
    d = np.array([1.0])
    suffix[n] = d
    for m in range(n - 1, -1, -1):
        q = probs[m]
        nd = np.zeros(len(d) + 1)
        nd[:-1] += d * (1 - q)
        nd[1:] += d * q
        suffix[m] = nd
        d = nd
    out[0] = suffix[0][0]
    d_in = np.array([1.0])
    for m in range(1, n + 1):
        q = probs[m - 1]
        nd = np.zeros(len(d_in) + 1)
        nd[:-1] += d_in * (1 - q)
        nd[1:] += d_in * q
        d_in = nd
        tp = np.arange(m + 1)[:, None].astype(float)
        fn = np.arange(len(suffix[m]))[None, :].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            F = np.where(tp > 0, 1.25 * tp / (0.25 * (tp + fn) + m), 0.0)
        out[m] = float((d_in[:, None] * suffix[m][None, :] * F).sum())
    return out


def expected_f05_decode(p: np.ndarray, i1: np.ndarray, min_p: float = 0.02, max_n: int = 30) -> np.ndarray:
    """Per S1 entity choose the probability-sorted prefix maximising expected F_0.5."""
    keep = np.zeros(len(p), bool)
    order = np.lexsort((-p, i1))
    k = i1[order]
    starts = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
    ends = np.r_[starts[1:], len(k)]
    ps = p[order]
    for s, e in zip(starts, ends):
        seg = ps[s:e][:max_n]
        m_valid = int((seg >= min_p).sum())
        if m_valid == 0:
            continue
        seg = seg[:m_valid]
        if seg[-1] >= 0.99:
            keep[order[s:s + m_valid]] = True
            continue
        ef = _expected_f05_prefixes(seg)
        m_best = int(np.argmax(ef))
        if m_best > 0:
            keep[order[s:s + m_best]] = True
    return keep


def build_lists(keep: np.ndarray, s1_ids: np.ndarray, s23_ids: np.ndarray, i1: np.ndarray, i23: np.ndarray) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for a, b in zip(s1_ids[i1[keep]], s23_ids[i23[keep]]):
        out.setdefault(a, set()).add(b)
    return out


def write_submission(path: str, s1_ids, lists: Dict[str, Set[str]], col: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for e in s1_ids:
            ids = lists.get(e)
            f.write(f"{e}\t{','.join(sorted(ids)) if ids else ''}\n")
