"""Fast character 3-gram TF-IDF (pure NumPy; ~50x faster than scikit-learn's analyzer).

Texts are ASCII after normalisation.  Each text is padded with a space on both sides
and every 3-byte window becomes a term id (c0*65536 + c1*256 + c2); windows crossing a
document boundary are discarded.  Term frequencies use the sub-linear 1+log(tf)
weighting, idf = log((1+n)/(1+df)) + 1, rows are L2-normalised.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

_SEP = 1  # byte used between documents (never appears in normalised text)


def _grams_chunk(texts: List[str]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (doc_index, term_id) for all 3-grams of the chunk."""
    joined = ("\x01 " + " \x01 ".join(texts) + " \x01").encode("ascii", "replace")
    buf = np.frombuffer(joined, dtype=np.uint8)
    n = len(buf)
    if n < 3:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), len(texts)
    c0 = buf[:-2].astype(np.int64)
    c1 = buf[1:-1].astype(np.int64)
    c2 = buf[2:].astype(np.int64)
    valid = (c0 != _SEP) & (c1 != _SEP) & (c2 != _SEP)
    term = (c0 << 16) | (c1 << 8) | c2
    # document index of every position = number of separators seen before it
    doc = np.cumsum(buf == _SEP)[:-2] - 1
    return doc[valid], term[valid], len(texts)


def char3_counts(texts: List[str], chunk: int = 500_000) -> sp.csr_matrix:
    """Sparse (n_docs x 2^24) matrix of 3-gram counts."""
    mats = []
    for s in range(0, len(texts), chunk):
        doc, term, nd = _grams_chunk(texts[s:s + chunk])
        m = sp.coo_matrix((np.ones(len(doc), np.float32), (doc, term)), shape=(nd, 1 << 24)).tocsr()
        m.sum_duplicates()
        mats.append(m)
    if not mats:
        return sp.csr_matrix((0, 1 << 24), dtype=np.float32)
    return sp.vstack(mats).tocsr() if len(mats) > 1 else mats[0]


class Char3Tfidf:
    def __init__(self, min_df: int = 2, max_df: float = 0.05):
        self.min_df = min_df
        self.max_df = max_df
        self.idf: np.ndarray | None = None
        self.col_map: np.ndarray | None = None

    def fit(self, counts: sp.csr_matrix) -> "Char3Tfidf":
        n = counts.shape[0]
        df = np.bincount(counts.indices, minlength=counts.shape[1])
        keep = (df >= self.min_df) & (df <= max(self.max_df * n, self.min_df))
        cols = np.flatnonzero(keep)
        self.col_map = np.full(counts.shape[1], -1, np.int64)
        self.col_map[cols] = np.arange(len(cols))
        self.idf = (np.log((1 + n) / (1 + df[cols])) + 1.0).astype(np.float32)
        return self

    def transform(self, counts: sp.csr_matrix) -> sp.csr_matrix:
        assert self.idf is not None
        new_cols = self.col_map[counts.indices]
        mask = new_cols >= 0
        data = counts.data.astype(np.float32)
        data = 1.0 + np.log(data)
        # rebuild csr with only kept columns
        row = np.repeat(np.arange(counts.shape[0], dtype=np.int64), np.diff(counts.indptr))
        m = sp.csr_matrix((data[mask] * self.idf[new_cols[mask]], (row[mask], new_cols[mask])),
                          shape=(counts.shape[0], len(self.idf)), dtype=np.float32)
        norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        m = sp.diags((1.0 / norms).astype(np.float32)) @ m
        return m.tocsr()


def fit_transform_pair(texts_a: List[str], texts_b: List[str], min_df: int = 2, max_df: float = 0.05):
    """TF-IDF fitted on a+b, returning (A, B) as L2-normalised CSR matrices."""
    ca = char3_counts(texts_a)
    cb = char3_counts(texts_b)
    vec = Char3Tfidf(min_df, max_df).fit(sp.vstack([ca, cb]).tocsr())
    return vec.transform(ca), vec.transform(cb)


def topq_per_row(M: sp.csr_matrix, q: int) -> sp.csr_matrix:
    """Keep only the ``q`` largest entries of every row (query-side pruning for retrieval)."""
    if q <= 0 or M.nnz == 0:
        return M
    row = np.repeat(np.arange(M.shape[0], dtype=np.int64), np.diff(M.indptr))
    order = np.lexsort((-M.data, row))
    r = row[order]
    starts = np.searchsorted(r, r, side="left")
    rank = np.arange(len(r)) - starts
    keep = np.zeros(len(r), bool)
    keep[order[rank < q]] = True
    out = sp.csr_matrix((M.data[keep], M.indices[keep], np.r_[0, np.cumsum(np.add.reduceat(keep, M.indptr[:-1], dtype=np.int64) * (np.diff(M.indptr) > 0))]),
                        shape=M.shape)
    return out
