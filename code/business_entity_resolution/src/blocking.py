"""Candidate generation (blocking).

Runs independently inside every country partition (the country label is an open set:
whatever values occur in the data are used).  Every Source 2/3 record is matched
against the Source 1 reference of the same country with several complementary,
cheap "nets".  All nets work on IDF-weighted *word* vectors:

  name space    - tokens of the cleaned core name, plus typo-tolerant keys per token
                  (sorted-letters key for words of >= 5 letters, 5-letter prefix key for
                  words of >= 7 letters) so that scrambled / truncated words still
                  share a term with the original;
  address space - tokens of the cleaned address (house number, street, locality, city).

Retrieval nets (sparse top-k dot products, both sides pruned to their rarest terms):
  * name net     top-k S1 rows by name cosine
  * address net  top-k S1 rows by address cosine
  * blended net  top-k by w*name + (1-w)*address cosine
  * reverse net  top-k S2/S3 rows for every S1 row on the blended space (so that S1
                 entities crowded out by popular names still receive candidates)
Exact-key nets (hash joins, no pruning):
  * same core name (space-insensitive), ranked by address cosine  - handles the very
    common case of a name shared by dozens of entities with different addresses
  * same cleaned address (all members; groups are tiny)
  * same street component, ranked by name cosine

The union is the candidate set.  Every candidate carries the exact cosines and its rank
in each net; a light learned filter (pipeline.stage0) then reduces the union to the
final candidate list handed to the matcher.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn


@dataclass
class BlockingConfig:
    k_name: int = 20
    k_addr: int = 20
    k_blend: int = 30
    k_reverse: int = 6
    g_name: int = 12           # same-name group: keep the g_name members with the most similar address
    g_name_max: int = 400      # ignore same-name groups larger than this (score-based nets still apply)
    g_street: int = 5          # same-street group: keep the g_street members with the most similar name
    g_street_max: int = 300
    min_name_cos: float = 0.15
    min_addr_cos: float = 0.15
    min_blend_cos: float = 0.12
    blend_name_weight: float = 0.5
    df_cap_frac: float = 0.003  # address terms occurring in more than this fraction of rows are not used for retrieval
    df_cap_min: int = 2000
    df_cap_frac_name: float = 0.01
    q_name: int = 10            # rarest terms kept per row for retrieval (name space)
    q_addr: int = 12            # ... address space
    name_bigrams: bool = True
    name_whole: bool = True
    addr_bigrams: bool = True
    addr_keys: bool = False     # scrambled-letter keys for address words (no measurable recall gain; off)
    n_threads: int = 8
    # safety cap only (pathological entities); the learned stage-0 filter does the real pruning.
    # A hard cap of 150 by blended cosine lost 58% of all blocking misses on India.
    max_cands_per_s1: int = 600
    chunk: int = 500_000


# ----------------------------------------------------------------------------
# token spaces
# ----------------------------------------------------------------------------
_SOUNDEX = str.maketrans("abcdefghijklmnopqrstuvwxyz", "01230120022455012623010202")


def soundex(w: str) -> str:
    """Classic Soundex code (jai/jay -> J000, laxmi/lakshmi -> L250, shree/sri -> S600)."""
    if not w or not w.isalpha():
        return ""
    codes = w.translate(_SOUNDEX)
    out = w[0]
    last = codes[0]
    for ch, c in zip(w[1:], codes[1:]):
        if c != "0" and c != last:
            out += c
        if ch not in "hw":
            last = c
    return (out + "000")[:4]


class Vocab:
    """Incremental string -> int mapping (memory-safe token encoding in chunks)."""

    def __init__(self):
        self.map: Dict[str, int] = {}

    def encode_docs(self, texts: List[str], enrich: bool, bigrams: bool, whole: bool, addr_keys: bool = False,
                    chunk: int = 200_000) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indptr, term_ids) of the documents (duplicates within a doc kept; removed later).

        enrich  - add typo-tolerant per-word keys (sorted letters, 4-letter prefix / suffix)
        bigrams - add adjacent word pairs ("delhi technology" -> "delhi_technology"), which keep
                  very frequent words useful for retrieval through their rarer combinations
        whole   - add the whole text as one key (exact-name term)
        """
        m = self.map
        indptr = np.zeros(len(texts) + 1, dtype=np.int64)
        ids_parts: List[np.ndarray] = []
        enrich_cache: Dict[str, List[str]] = {}
        pos = 0
        for s in range(0, len(texts), chunk):
            buf: List[int] = []
            for k, t in enumerate(texts[s:s + chunk]):
                toks = t.split()
                out: List[str] = list(toks)
                if bigrams and len(toks) >= 2:
                    out.extend(a + "_" + b for a, b in zip(toks[:-1], toks[1:]))
                if whole and len(toks) >= 2:
                    out.append("=" + "_".join(toks))
                if enrich:
                    for w in toks:
                        e = enrich_cache.get(w)
                        if e is None:
                            e = []
                            if w.isalpha():
                                if len(w) >= 3:
                                    e.append("*" + soundex(w))           # phonetic variants (jai/jay, laxmi/lakshmi)
                                if len(w) >= 5:
                                    e.append("~" + "".join(sorted(w)))   # scrambled letters
                                    e.append("^" + w[:4])                # truncation / suffix typo
                                    if len(w) >= 6:
                                        e.append("$" + w[-4:])           # prefix typo
                            enrich_cache[w] = e
                        out.extend(e)
                elif addr_keys:
                    for w in toks:
                        if len(w) >= 5 and w.isalpha():
                            out.append("~" + "".join(sorted(w)))         # blcok/block, suoth/south
                for w in out:
                    i = m.get(w)
                    if i is None:
                        i = len(m)
                        m[w] = i
                    buf.append(i)
                pos += len(out)
                indptr[s + k + 1] = pos
            ids_parts.append(np.asarray(buf, dtype=np.int32))
        ids = np.concatenate(ids_parts) if ids_parts else np.zeros(0, np.int32)
        return indptr, ids


class TokenSpace:
    """IDF-weighted, L2-normalised binary token vectors for (query rows, index rows)."""

    def __init__(self, texts_q: List[str], texts_i: List[str], enrich: bool, bigrams: bool, whole: bool, df_cap: int, q: int,
                 addr_keys: bool = False):
        voc = Vocab()
        ip_q, id_q = voc.encode_docs(texts_q, enrich, bigrams, whole, addr_keys)
        ip_i, id_i = voc.encode_docs(texts_i, enrich, bigrams, whole, addr_keys)
        V = len(voc.map)
        del voc
        nq, ni = len(texts_q), len(texts_i)
        X = sp.csr_matrix((np.ones(len(id_q) + len(id_i), np.float32), np.concatenate([id_q, id_i]),
                           np.concatenate([ip_q, ip_i[1:] + ip_q[-1]])), shape=(nq + ni, V))
        X.sum_duplicates()
        X.data[:] = 1.0
        n = nq + ni
        df = np.bincount(X.indices, minlength=V)
        self.idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
        W = X.multiply(self.idf).tocsr()
        del X
        norms = np.sqrt(np.asarray(W.multiply(W).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        W = (sp.diags((1.0 / norms).astype(np.float32)) @ W).tocsr()
        self.Q: sp.csr_matrix = W[:nq]
        self.I: sp.csr_matrix = W[nq:]
        rare = (df <= df_cap).astype(np.float32)
        self.Qp = topq_per_row(_mask_cols(self.Q, rare), q)
        self.Ip = topq_per_row(_mask_cols(self.I, rare), q)
        self.nnz_per_row = W.nnz / max(n, 1)
        self.V = V
        del W

    def drop_pruned(self):
        """Release the pruned (net) matrices once all nets have been computed."""
        self.Qp = None
        self.Ip = None


def _mask_cols(M: sp.csr_matrix, keep: np.ndarray) -> sp.csr_matrix:
    out = M.multiply(keep[None, :]).tocsr()
    out.eliminate_zeros()
    return out


def topq_per_row(M: sp.csr_matrix, q: int) -> sp.csr_matrix:
    """Keep only the ``q`` largest entries of every row."""
    if q <= 0 or M.nnz == 0:
        return M
    M = M.tocsr()
    row = np.repeat(np.arange(M.shape[0], dtype=np.int64), np.diff(M.indptr))
    order = np.lexsort((-M.data, row))
    r = row[order]
    starts = np.searchsorted(r, r, side="left")
    rank = np.arange(len(r)) - starts
    keep = np.zeros(len(r), bool)
    keep[order[rank < q]] = True
    counts = np.bincount(row[keep], minlength=M.shape[0])
    return sp.csr_matrix((M.data[keep], M.indices[keep], np.r_[0, np.cumsum(counts)]), shape=M.shape)


def row_dot(A: sp.csr_matrix, B: sp.csr_matrix, ra: np.ndarray, rb: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    """Exact cosine (dot of L2 rows) for arbitrary (ra[i], rb[i]) pairs."""
    out = np.empty(len(ra), dtype=np.float32)
    for s in range(0, len(ra), chunk):
        e = min(s + chunk, len(ra))
        out[s:e] = np.asarray(A[ra[s:e]].multiply(B[rb[s:e]]).sum(axis=1)).ravel()
    return out


# ----------------------------------------------------------------------------
# retrieval helpers
# ----------------------------------------------------------------------------
def _topn(A: sp.csr_matrix, BT: sp.csr_matrix, k: int, thr: float, n_threads: int, chunk: int) -> sp.csr_matrix:
    if A.shape[0] == 0 or BT.shape[1] == 0:
        return sp.csr_matrix((A.shape[0], BT.shape[1]), dtype=np.float32)
    outs = []
    for s in range(0, A.shape[0], chunk):
        C = sp_matmul_topn(A[s:s + chunk], BT, top_n=k, threshold=thr, sort=True, n_threads=n_threads)
        outs.append(C.tocsr())
    return sp.vstack(outs).tocsr() if len(outs) > 1 else outs[0]


def _csr_to_pairs(C: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, cols, rank_within_row) of a sorted top-n CSR."""
    rows = np.repeat(np.arange(C.shape[0], dtype=np.int64), np.diff(C.indptr))
    cols = C.indices.astype(np.int64)
    vals = C.data
    order = np.lexsort((-vals, rows))
    rows, cols = rows[order], cols[order]
    starts = np.searchsorted(rows, rows, side="left")
    rank = (np.arange(len(rows)) - starts).astype(np.int16)
    return rows, cols, rank


def _group_pairs(key_q: np.ndarray, key_i: np.ndarray, max_group: int) -> Tuple[np.ndarray, np.ndarray]:
    """All (q, i) pairs whose keys are equal and non-empty; groups larger than max_group are skipped."""
    codes, uniq = pd.factorize(np.concatenate([key_q, key_i]), sort=False)
    cq, ci = codes[: len(key_q)], codes[len(key_q):]
    empty = np.flatnonzero(uniq == "")
    order = np.argsort(ci, kind="stable")
    ci_sorted = ci[order]
    starts = np.searchsorted(ci_sorted, cq, side="left")
    ends = np.searchsorted(ci_sorted, cq, side="right")
    sizes = ends - starts
    ok = (sizes > 0) & (sizes <= max_group)
    if len(empty):
        ok &= cq != empty[0]
    qs = np.flatnonzero(ok)
    sizes = sizes[qs]
    tot = int(sizes.sum())
    if tot == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rq = np.repeat(qs, sizes).astype(np.int64)
    offs = np.arange(tot) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    ri = order[np.repeat(starts[qs], sizes) + offs].astype(np.int64)
    return rq, ri


def _sorted_tokens(col: pd.Series) -> np.ndarray:
    """Order-insensitive exact key: tokens sorted (address order varies between sources)."""
    return np.array([" ".join(sorted(a.split())) for a in col.to_numpy(dtype=object)], dtype=object)


def _rank_within(keys: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = np.lexsort((-scores, keys))
    k = keys[order]
    starts = np.searchsorted(k, k, side="left")
    rank = np.empty(len(keys), np.int64)
    rank[order] = np.arange(len(keys)) - starts
    return rank


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
RANK_COLS = ["r_name", "r_addr", "r_blend", "r_rev", "g_name", "g_addr", "g_street"]


def block_partition(s1: pd.DataFrame, s23: pd.DataFrame, cfg: BlockingConfig, log=print,
                    max_rows: int = 4_500_000) -> pd.DataFrame:
    """Generate candidates for one country partition.

    ``s1`` and ``s23`` are normalised frames (see normalize.normalize_frame). Returns a
    DataFrame with integer positions ``i1`` (row in s1) and ``i23`` (row in s23), the
    rank of the pair in every net (99 = not retrieved by that net) and exact cosines.

    Very large partitions are processed in contiguous S2/S3 shards of at most ``max_rows``
    rows (every per-record net is independent of the other records; only the IDF weights
    and the per-entity reverse net see one shard at a time), which bounds peak memory.
    """
    n23 = len(s23)
    if n23 <= max_rows:
        return _block_partition_once(s1, s23, cfg, log)
    n_shards = int(np.ceil(n23 / max_rows))
    bounds = np.linspace(0, n23, n_shards + 1).astype(int)
    parts = []
    for si in range(n_shards):
        lo, hi = bounds[si], bounds[si + 1]
        log(f"    shard {si + 1}/{n_shards}: S2/S3 rows {lo:,}-{hi:,}")
        part = _block_partition_once(s1, s23.iloc[lo:hi].reset_index(drop=True), cfg, log)
        part["i23"] = (part["i23"].to_numpy(np.int64) + lo).astype(np.int32)
        parts.append(part)
    cand = pd.concat(parts, ignore_index=True)
    del parts
    log(f"    sharded candidates: {len(cand):,} ({len(cand)/max(len(s1),1):.1f} per S1)")
    return cand


def _block_partition_once(s1: pd.DataFrame, s23: pd.DataFrame, cfg: BlockingConfig, log=print) -> pd.DataFrame:
    t0 = time.time()
    n1, n23 = len(s1), len(s23)
    if n1 == 0 or n23 == 0:
        return pd.DataFrame({"i1": np.zeros(0, np.int64), "i23": np.zeros(0, np.int64)})
    n_all = n1 + n23
    df_cap = max(cfg.df_cap_min, int(cfg.df_cap_frac * n_all))
    df_cap_name = max(cfg.df_cap_min, int(cfg.df_cap_frac_name * n_all))
    name_sp = TokenSpace(s23["name_core"].tolist(), s1["name_core"].tolist(), True, cfg.name_bigrams, cfg.name_whole, df_cap_name, cfg.q_name)
    addr_sp = TokenSpace(s23["addr"].tolist(), s1["addr"].tolist(), False, cfg.addr_bigrams, False, df_cap, cfg.q_addr, addr_keys=cfg.addr_keys)
    log(f"    token spaces in {time.time()-t0:.0f}s (name V={name_sp.V:,} nnz/row={name_sp.nnz_per_row:.1f}; "
        f"addr V={addr_sp.V:,} nnz/row={addr_sp.nnz_per_row:.1f}; df caps {df_cap_name:,}/{df_cap:,})")

    w = np.float32(np.sqrt(cfg.blend_name_weight))
    v = np.float32(np.sqrt(1.0 - cfg.blend_name_weight))
    Qb = sp.hstack([name_sp.Qp * w, addr_sp.Qp * v]).tocsr()
    Ib = sp.hstack([name_sp.Ip * w, addr_sp.Ip * v]).tocsr()

    keys: List[np.ndarray] = []
    ranks: Dict[str, np.ndarray] = {}
    spans: List[Tuple[str, int, int]] = []

    def add(col: str, i23_arr: np.ndarray, i1_arr: np.ndarray, rk: np.ndarray):
        start = sum(len(k) for k in keys)
        keys.append(i1_arr * np.int64(n23) + i23_arr)
        ranks[col] = rk.astype(np.int16)
        spans.append((col, start, start + len(i1_arr)))

    nets = [
        ("name", name_sp.Qp, name_sp.Ip, cfg.k_name, cfg.min_name_cos, "r_name", False),
        ("addr", addr_sp.Qp, addr_sp.Ip, cfg.k_addr, cfg.min_addr_cos, "r_addr", False),
        ("blend", Qb, Ib, cfg.k_blend, cfg.min_blend_cos, "r_blend", False),
        ("reverse", Ib, Qb, cfg.k_reverse, cfg.min_blend_cos, "r_rev", True),
    ]
    for label, A, B, k, thr, col, rev in nets:
        if k <= 0:
            continue
        t1 = time.time()
        BT = B.T.tocsr()
        C = _topn(A, BT, k, thr, cfg.n_threads, cfg.chunk)
        del BT
        r, c, rk = _csr_to_pairs(C)
        del C
        if rev:
            add(col, c, r, rk)        # r = i1, c = i23
        else:
            add(col, r, c, rk)        # r = i23, c = i1
        log(f"    {label} net: {len(r):,} pairs in {time.time()-t1:.0f}s")
    del Qb, Ib
    name_sp.drop_pruned()
    addr_sp.drop_pruned()

    # exact-key nets -------------------------------------------------------
    t1 = time.time()
    rq, ri = _group_pairs(s23["name_nospace"].to_numpy(dtype=object), s1["name_nospace"].to_numpy(dtype=object), cfg.g_name_max)
    if len(rq):
        sc = row_dot(addr_sp.Q, addr_sp.I, rq, ri)
        rk = _rank_within(rq, sc)
        keep = rk < cfg.g_name
        add("g_name", rq[keep], ri[keep], rk[keep])
        log(f"    same-name groups: {len(rq):,} member pairs -> {keep.sum():,} kept in {time.time()-t1:.0f}s")
    t1 = time.time()
    rq, ri = _group_pairs(_sorted_tokens(s23["addr"]), _sorted_tokens(s1["addr"]), 50)
    if len(rq):
        add("g_addr", rq, ri, np.zeros(len(rq), np.int16))
        log(f"    same-address groups: {len(rq):,} pairs in {time.time()-t1:.0f}s")
    t1 = time.time()
    rq, ri = _group_pairs(s23["addr_street"].to_numpy(dtype=object), s1["addr_street"].to_numpy(dtype=object), cfg.g_street_max)
    if len(rq):
        sc = row_dot(name_sp.Q, name_sp.I, rq, ri)
        rk = _rank_within(rq, sc)
        keep = rk < cfg.g_street
        add("g_street", rq[keep], ri[keep], rk[keep])
        log(f"    same-street groups: {len(rq):,} member pairs -> {keep.sum():,} kept in {time.time()-t1:.0f}s")
    del rq, ri, sc, rk, keep

    # union (memory-lean: sorted unique keys + searchsorted, int32 positions) ----------
    t1 = time.time()
    allk = np.concatenate(keys)
    del keys
    uk, inv = np.unique(allk, return_inverse=True)
    del allk
    n_u = len(uk)
    cols: Dict[str, np.ndarray] = {
        "i1": (uk // n23).astype(np.int32),
        "i23": (uk % n23).astype(np.int32),
    }
    del uk
    for col, s, e in spans:
        arr = np.full(n_u, 99, np.int16)
        arr[inv[s:e]] = ranks[col]
        cols[col] = arr
    del ranks, inv
    for col in RANK_COLS:
        if col not in cols:
            cols[col] = np.full(n_u, 99, np.int16)
    i1 = cols["i1"].astype(np.int64)
    i23 = cols["i23"].astype(np.int64)
    cols["cos_name"] = row_dot(name_sp.Q, name_sp.I, i23, i1)
    cols["cos_addr"] = row_dot(addr_sp.Q, addr_sp.I, i23, i1)
    cols["cos_blend"] = (cfg.blend_name_weight * cols["cos_name"] + (1 - cfg.blend_name_weight) * cols["cos_addr"]).astype(np.float32)
    del name_sp, addr_sp, i23
    log(f"    union: {n_u:,} pairs in {time.time()-t1:.0f}s")

    if cfg.max_cands_per_s1 and n_u:
        # rows are sorted by i1 (key order); rank within each S1 entity by blended cosine
        rank_a = _rank_within(i1, cols["cos_blend"])
        keep = rank_a < cfg.max_cands_per_s1
        del rank_a
        if not keep.all():
            cols = {c: a[keep] for c, a in cols.items()}
        del keep
    del i1
    cand = pd.DataFrame(cols)
    del cols
    log(f"    candidates: {len(cand):,} ({len(cand)/max(n1,1):.1f} per S1) in {time.time()-t0:.0f}s total")
    return cand
