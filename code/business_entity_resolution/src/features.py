"""Pairwise feature engineering for candidate pairs.

All features are computed vectorised (RapidFuzz ``cpdist`` on several cores, sparse token
algebra for IDF-weighted overlaps, NumPy for the rest).  Nothing depends on the country
label itself, so the model transfers to countries unseen in training.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import CountVectorizer

from blocking import RANK_COLS
from normalize import LEGAL_TOKENS

N_WORKERS = 8


def _cp(a, b, scorer) -> np.ndarray:
    if len(a) == 0:
        return np.zeros(0, np.float32)
    return process.cpdist(a, b, scorer=scorer, workers=N_WORKERS, dtype=np.float32)


def _row_dot(A: sp.csr_matrix, B: sp.csr_matrix, ra: np.ndarray, rb: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(len(ra), dtype=np.float32)
    for s in range(0, len(ra), chunk):
        e = min(s + chunk, len(ra))
        out[s:e] = np.asarray(A[ra[s:e]].multiply(B[rb[s:e]]).sum(axis=1)).ravel()
    return out


class TokenSpace:
    """Binary token matrix + IDF for one text column of one partition (S1 rows, S23 rows)."""

    def __init__(self, texts_s1: List[str], texts_s23: List[str]):
        vec = CountVectorizer(analyzer=str.split, binary=True, dtype=np.float32)
        X = vec.fit_transform(texts_s1 + texts_s23).tocsr()
        n = X.shape[0]
        df = np.asarray(X.sum(axis=0)).ravel()
        self.idf = (np.log((n + 1) / (df + 1)) + 1.0).astype(np.float32)
        self.X1 = X[: len(texts_s1)]
        self.X23 = X[len(texts_s1):]
        self.W1 = self.X1.multiply(self.idf).tocsr()
        self.sum1 = np.asarray(self.W1.sum(axis=1)).ravel()
        self.sum23 = np.asarray(self.X23.multiply(self.idf).sum(axis=1)).ravel()
        self.cnt1 = np.asarray(self.X1.sum(axis=1)).ravel()
        self.cnt23 = np.asarray(self.X23.sum(axis=1)).ravel()
        thr = np.quantile(self.idf, 0.6) if len(self.idf) else 0.0
        rare = (self.idf >= thr).astype(np.float32)
        self.R1 = self.X1.multiply(rare).tocsr()

    def pair_feats(self, i1: np.ndarray, i23: np.ndarray, prefix: str) -> Dict[str, np.ndarray]:
        shared_idf = _row_dot(self.W1, self.X23, i1, i23)
        shared_cnt = _row_dot(self.X1, self.X23, i1, i23)
        shared_rare = _row_dot(self.R1, self.X23, i1, i23)
        s1, s23 = self.sum1[i1], self.sum23[i23]
        c1, c23 = self.cnt1[i1], self.cnt23[i23]
        union_idf = s1 + s23 - shared_idf
        mn = np.minimum(s1, s23)
        return {
            f"{prefix}_idf_jacc": np.where(union_idf > 0, shared_idf / np.maximum(union_idf, 1e-6), 0).astype(np.float32),
            f"{prefix}_idf_cont": np.where(mn > 0, shared_idf / np.maximum(mn, 1e-6), 0).astype(np.float32),
            f"{prefix}_tok_jacc": np.where(c1 + c23 - shared_cnt > 0, shared_cnt / np.maximum(c1 + c23 - shared_cnt, 1), 0).astype(np.float32),
            f"{prefix}_tok_cont": np.where(np.minimum(c1, c23) > 0, shared_cnt / np.maximum(np.minimum(c1, c23), 1), 0).astype(np.float32),
            f"{prefix}_shared_rare": shared_rare.astype(np.float32),
            f"{prefix}_shared_idf": shared_idf.astype(np.float32),
            f"{prefix}_ntok1": c1.astype(np.float32),
            f"{prefix}_ntok23": c23.astype(np.float32),
            # asymmetric differences: decoys *add* a word to the entity name, true variants tend to drop one
            f"{prefix}_extra23": (c23 - shared_cnt).astype(np.float32),
            f"{prefix}_extra1": (c1 - shared_cnt).astype(np.float32),
            f"{prefix}_extra_idf23": (s23 - shared_idf).astype(np.float32),
        }


def _suffix_set(name: str) -> frozenset:
    return frozenset(t for t in name.split() if t in LEGAL_TOKENS)


def _first_tok(arr) -> np.ndarray:
    return np.array([s.split(" ", 1)[0] if s else "" for s in arr], dtype=object)


_LEET = str.maketrans("0123456789", "olzeasgtbg")


def leet_name(s: str) -> str:
    """Undo digit-for-letter substitutions ('Behavi0ral', 'O5X', '8RANDS') inside alphanumeric tokens;
    pure numbers are left alone."""
    if not any(ch.isdigit() for ch in s):
        return s
    out = []
    for t in s.split():
        if any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
            out.append(t.translate(_LEET))
        else:
            out.append(t)
    return " ".join(out)


def ensure_derived_columns(df: pd.DataFrame) -> None:
    """Per-record helper columns derived from the normalised text (cached on the frame)."""
    if "name_leet" not in df.columns:
        df["name_leet"] = [leet_name(s) for s in df["name_core"].to_numpy(dtype=object)]


def _house_edit(h1: np.ndarray, h23: np.ndarray) -> np.ndarray:
    """Levenshtein distance between house numbers (capped at 9); -1 when either side is empty."""
    from rapidfuzz.distance import Levenshtein
    out = np.full(len(h1), -1, np.int8)
    m = (h1 != "") & (h23 != "")
    if m.any():
        d = process.cpdist(h1[m], h23[m], scorer=Levenshtein.distance, workers=N_WORKERS, dtype=np.int32)
        out[m] = np.minimum(d, 9).astype(np.int8)
    return out


def _house_prefix(h1: np.ndarray, h23: np.ndarray) -> np.ndarray:
    """1 when one house number is a strict prefix of the other (dropped / extra trailing digit),
    0 otherwise, -1 when either side is empty."""
    out = np.full(len(h1), -1, np.int8)
    for i, (a, b) in enumerate(zip(h1, h23)):
        if a and b:
            out[i] = 1 if (a != b and (a.startswith(b) or b.startswith(a))) else 0
    return out


def _base_features(s1: pd.DataFrame, s23: pd.DataFrame, i1: np.ndarray, i23: np.ndarray) -> Dict[str, np.ndarray]:
    """String-similarity features for one chunk of pairs."""
    n = len(i1)
    F: Dict[str, np.ndarray] = {}

    def col(df, c):
        return df[c].to_numpy(dtype=object)

    # ---------------- names
    nm1, nm23 = col(s1, "name")[i1], col(s23, "name")[i23]
    co1, co23 = col(s1, "name_core")[i1], col(s23, "name_core")[i23]
    ns1, ns23 = col(s1, "name_nospace")[i1], col(s23, "name_nospace")[i23]
    al23 = col(s23, "name_alias")[i23]

    F["n_full_tsr"] = _cp(nm1, nm23, fuzz.token_set_ratio)
    F["n_core_tsr"] = _cp(co1, co23, fuzz.token_set_ratio)
    F["n_core_tsort"] = _cp(co1, co23, fuzz.token_sort_ratio)
    F["n_core_ratio"] = _cp(co1, co23, fuzz.ratio)
    F["n_core_partial"] = _cp(co1, co23, fuzz.partial_ratio)
    F["n_nospace_ratio"] = _cp(ns1, ns23, fuzz.ratio)
    has_alias = col(s23, "has_alias")[i23].astype(np.int8)
    alias_tsr = np.zeros(n, np.float32)
    if has_alias.any():
        m = has_alias.astype(bool)
        alias_tsr[m] = _cp(co1[m], al23[m], fuzz.token_set_ratio)
    F["n_alias_tsr"] = alias_tsr
    F["n_best_tsr"] = np.maximum(F["n_core_tsr"], alias_tsr)
    F["n_core_exact"] = (co1 == co23).astype(np.int8)
    le1, le23 = col(s1, "name_leet")[i1], col(s23, "name_leet")[i23]
    F["n_leet_ratio"] = _cp(le1, le23, fuzz.ratio)
    F["n_leet_exact"] = ((le1 == le23) & (co1 != co23)).astype(np.int8)     # equal only after undoing digit substitutions
    del le1, le23
    ft1, ft23 = _first_tok(co1), _first_tok(co23)
    F["n_first_eq"] = (ft1 == ft23).astype(np.int8)
    F["n_first_ratio"] = _cp(ft1, ft23, fuzz.ratio)
    F["n_len1"] = np.array([len(s) for s in co1], np.float32)
    F["n_len23"] = np.array([len(s) for s in co23], np.float32)
    F["n_len_ratio"] = (np.minimum(F["n_len1"], F["n_len23"]) / np.maximum(np.maximum(F["n_len1"], F["n_len23"]), 1)).astype(np.float32)
    sf1 = [_suffix_set(s) for s in nm1]
    sf23 = [_suffix_set(s) for s in nm23]
    F["n_suffix_agree"] = np.array([1 if (a and b and a == b) else (0 if (a and b) else -1) for a, b in zip(sf1, sf23)], np.int8)
    F["b_is_domain"] = col(s23, "is_domain")[i23].astype(np.int8)
    F["b_is_nonlatin"] = col(s23, "is_nonlatin")[i23].astype(np.int8)
    F["b_has_alias"] = has_alias
    F["b_single_tok"] = np.array([1 if len(s.split()) == 1 else 0 for s in co23], np.int8)
    F["b_is_s3"] = np.array([1 if e.startswith("S3") else 0 for e in col(s23, "entity_id")[i23]], np.int8)
    del nm1, nm23, co1, co23, ns1, ns23, al23, ft1, ft23, sf1, sf23

    # ---------------- addresses
    ad1, ad23 = col(s1, "addr")[i1], col(s23, "addr")[i23]
    rs1, rs23 = col(s1, "addr_rest")[i1], col(s23, "addr_rest")[i23]
    st1, st23 = col(s1, "addr_street")[i1], col(s23, "addr_street")[i23]
    F["a_tsr"] = _cp(ad1, ad23, fuzz.token_set_ratio)
    F["a_tsort"] = _cp(ad1, ad23, fuzz.token_sort_ratio)
    F["a_ratio"] = _cp(ad1, ad23, fuzz.ratio)
    F["a_partial"] = _cp(ad1, ad23, fuzz.partial_ratio)
    F["a_rest_tsr"] = _cp(rs1, rs23, fuzz.token_set_ratio)
    F["a_street_ratio"] = _cp(st1, st23, fuzz.ratio)
    F["a_exact"] = ((ad1 == ad23) & (ad1 != "")).astype(np.int8)
    F["a_street_exact"] = ((st1 == st23) & (st1 != "")).astype(np.int8)
    F["a_len1"] = np.array([len(s) for s in ad1], np.float32)
    F["a_len23"] = np.array([len(s) for s in ad23], np.float32)
    del ad1, ad23, rs1, rs23, st1, st23
    h1, h23 = col(s1, "house_no")[i1], col(s23, "house_no")[i23]
    F["house_eq"] = np.where((h1 == "") | (h23 == ""), -1, np.where(h1 == h23, 1, 0)).astype(np.int8)
    F["house_edit"] = _house_edit(h1, h23)
    F["house_prefix"] = _house_prefix(h1, h23)
    r1, r23 = col(s1, "addr_region")[i1], col(s23, "addr_region")[i23]
    F["region_eq"] = np.where((r1 == "") | (r23 == ""), -1, np.where(r1 == r23, 1, 0)).astype(np.int8)
    nu1, nu23 = col(s1, "addr_nums")[i1], col(s23, "addr_nums")[i23]
    F["nums_tsr"] = _cp(nu1, nu23, fuzz.token_set_ratio)
    F["nums_eq"] = np.where((nu1 == "") | (nu23 == ""), -1, np.where(nu1 == nu23, 1, 0)).astype(np.int8)
    F["a_empty23"] = col(s23, "addr_empty")[i23].astype(np.int8)
    return F


def compute_features(s1: pd.DataFrame, s23: pd.DataFrame, cand: pd.DataFrame, log=print, chunk: int = 2_000_000) -> pd.DataFrame:
    """Feature matrix for candidate pairs of one partition.

    ``cand`` must contain i1, i23 (positions in s1 / s23) and the blocking columns
    cos_name, cos_addr, cos_blend, r_name, r_addr, r_blend, r_rev.  String features are
    computed in chunks straight into preallocated arrays (bounded transient memory);
    token-overlap and competition features are computed on the whole table.
    """
    t0 = time.time()
    ensure_derived_columns(s1)
    ensure_derived_columns(s23)
    i1 = cand["i1"].to_numpy(np.int64)
    i23 = cand["i23"].to_numpy(np.int64)
    n = len(cand)
    F: Dict[str, np.ndarray] = {}
    for s in range(0, n, chunk):
        part = _base_features(s1, s23, i1[s:s + chunk], i23[s:s + chunk])
        if not F:
            F = {k: np.empty(n, dtype=v.dtype) for k, v in part.items()}
        for k, v in part.items():
            F[k][s:s + len(v)] = v
        del part
    log(f"    string features in {time.time()-t0:.0f}s")

    # ---------------- anagram-tolerant name similarity (sorted letters of the space-less core name)
    srt1 = np.array(["".join(sorted(s)) for s in s1["name_nospace"].to_numpy(dtype=object)], dtype=object)
    srt23 = np.array(["".join(sorted(s)) for s in s23["name_nospace"].to_numpy(dtype=object)], dtype=object)
    F["n_sorted_ratio"] = np.empty(n, np.float32)
    for s in range(0, n, chunk):
        F["n_sorted_ratio"][s:s + chunk] = _cp(srt1[i1[s:s + chunk]], srt23[i23[s:s + chunk]], fuzz.ratio)
    del srt1, srt23

    # ---------------- ambiguity of the reference side: how many S1 entities share this name / address / street
    def dup_counts(col: str, arr1, arr23):
        vc = s1[col].value_counts()
        c1 = vc.reindex(s1[col].to_numpy()).to_numpy(np.float32)[arr1]
        c23 = vc.reindex(s23[col].to_numpy()).fillna(0).to_numpy(np.float32)[arr23]
        return c1, c23
    for col, tag in [("name_nospace", "name"), ("addr", "addr"), ("addr_street", "street")]:
        c1, c23 = dup_counts(col, i1, i23)
        F[f"dup_{tag}1"] = c1
        F[f"dup_{tag}23"] = c23          # number of S1 entities carrying exactly the S2/S3 value (0 = none)
    empty_street = s1["addr_street"].to_numpy(dtype=object)[i1] == ""
    F["dup_street1"][empty_street] = 0
    F["dup_street23"][s23["addr_street"].to_numpy(dtype=object)[i23] == ""] = 0
    F["dup_addr1"][s1["addr"].to_numpy(dtype=object)[i1] == ""] = 0
    F["dup_addr23"][s23["addr"].to_numpy(dtype=object)[i23] == ""] = 0

    # ---------------- IDF-weighted token overlaps
    ts = TokenSpace(s1["name_core"].tolist(), s23["name_core"].tolist())
    F.update(ts.pair_feats(i1, i23, "nt"))
    ts = TokenSpace(s1["addr"].tolist(), s23["addr"].tolist())
    F.update(ts.pair_feats(i1, i23, "at"))
    ts = TokenSpace(s1["addr_rest"].tolist(), s23["addr_rest"].tolist())
    F.update(ts.pair_feats(i1, i23, "rt"))
    del ts

    # ---------------- blocking scores
    for c in ["cos_name", "cos_addr", "cos_blend"]:
        F[c] = cand[c].to_numpy(np.float32)
    for c in RANK_COLS:
        F[c] = cand[c].to_numpy(np.int16)
    F["n_nets"] = sum((F[c] < 99).astype(np.int8) for c in RANK_COLS).astype(np.int8)

    feat = pd.DataFrame(F)
    del F
    feat = add_competition_features(feat, i1, i23)
    log(f"    features: {n:,} pairs x {feat.shape[1]} cols in {time.time()-t0:.0f}s")
    return feat


# ----------------------------------------------------------------------------
# group helpers (competition features)
# ----------------------------------------------------------------------------
def _group_max(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    v = values[order]
    starts = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
    mx = np.maximum.reduceat(v, starts)
    sizes = np.diff(np.r_[starts, len(k)])
    out = np.empty_like(values)
    out[order] = np.repeat(mx, sizes)
    return out


def _group_second_max(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Max over the *other* members of the group (excluding self); -1 when alone."""
    order = np.lexsort((-values, keys))
    k = keys[order]
    v = values[order]
    starts = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
    ends = np.r_[starts[1:], len(k)]
    sizes = ends - starts
    first = np.repeat(v[starts], sizes)
    second = np.repeat(v[np.minimum(starts + 1, ends - 1)], sizes)
    pos = np.arange(len(k)) - np.repeat(starts, sizes)
    alone = np.repeat(sizes, sizes) == 1
    other = np.where(pos == 0, np.where(alone, -1.0, second), first)
    out = np.empty_like(values)
    out[order] = other
    return out


def _group_sum(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    v = values[order]
    starts = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
    sm = np.add.reduceat(v, starts)
    sizes = np.diff(np.r_[starts, len(k)])
    out = np.empty_like(values)
    out[order] = np.repeat(sm, sizes)
    return out


def _group_count(keys: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    return cnt[inv]


COMPETITION_COLS = ("cos_blend", "cos_name", "cos_addr", "n_best_tsr", "a_tsr", "nt_idf_jacc")


def add_competition_features(feat: pd.DataFrame, i1: np.ndarray, i23: np.ndarray) -> pd.DataFrame:
    feat["cnt_b"] = _group_count(i23).astype(np.float32)
    feat["cnt_a"] = _group_count(i1).astype(np.float32)
    for c in COMPETITION_COLS:
        v = feat[c].to_numpy(np.float32)
        feat[f"{c}_gap_b"] = (_group_max(v, i23) - v).astype(np.float32)
        feat[f"{c}_other_b"] = _group_second_max(v, i23).astype(np.float32)
        feat[f"{c}_gap_a"] = (_group_max(v, i1) - v).astype(np.float32)
    return feat


def _group_max_other_sub(values: np.ndarray, keys: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """Max of ``values`` over the rows that share the row's key but carry a *different* ``sub`` label;
    -1 when there is none."""
    order = np.lexsort((-values, keys))
    k, v, s = keys[order], values[order], sub[order]
    n = len(k)
    starts = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
    sizes = np.diff(np.r_[starts, n])
    gid = np.repeat(np.arange(len(starts)), sizes)
    first_sub = s[starts][gid]
    first_val = v[starts][gid]
    idx = np.where(s != first_sub, np.arange(n), n)               # first row (best value) with another sub
    fd = np.minimum.reduceat(idx, starts)
    fd_val = np.where(fd < n, v[np.minimum(fd, n - 1)], -1.0).astype(values.dtype)[gid]
    other = np.where(s == first_sub, fd_val, first_val)
    out = np.empty_like(values)
    out[order] = other
    return out


# Record attributes whose exact agreement between candidate records of the same Source 1 entity
# (or across entities) carries collective evidence: records of one entity are perturbations of the
# same canonical record, so they tend to repeat each other's address / house number / spelling,
# whereas decoy records carry a unique, freshly perturbed house number.
SIB_KEYS = (("addr", "addr"), ("house_no", "house"), ("addr_nums", "nums"), ("addr_street", "street"),
            ("name", "name"), ("name_core", "core"))
SIB_TAGS = tuple(t for _, t in SIB_KEYS)
OTH_TAGS = ("addr", "name", "core")
KEY_COLS = ["entity_id", "country"] + [c for c, _ in SIB_KEYS]

STAGE2_COLS = (["p1", "p1_max_b", "p1_gap_b", "p1_other_b", "p1_max_a", "p1_gap_a", "p1_other_a", "p1_sum_a", "p1_conf_a"]
               + [f"sib_{t}_n" for t in SIB_TAGS] + [f"sib_{t}_p" for t in SIB_TAGS] + [f"oth_{t}_p" for t in OTH_TAGS])


def record_keys(s1: pd.DataFrame, s23: pd.DataFrame) -> Dict[str, tuple]:
    """Integer codes of the SIB_KEYS columns, shared between the S1 and S23 rows of one country
    (-1 = empty value)."""
    keys = {}
    for col, tag in SIB_KEYS:
        a = s1[col].to_numpy(dtype=object)
        b = s23[col].to_numpy(dtype=object)
        allv = np.concatenate([a, b])
        codes, _ = pd.factorize(allv)
        codes = codes.astype(np.int64)
        codes[allv == ""] = -1
        keys[tag] = (codes[:len(a)], codes[len(a):])
    return keys


def add_stage2_features(feat: pd.DataFrame, p: np.ndarray, i1: np.ndarray, i23: np.ndarray,
                        keys: Dict[str, tuple]) -> pd.DataFrame:
    """Second-stage features derived from first-stage probabilities (in place): competition for the
    record / for the entity, plus sibling agreement (same S1, other records with the same attribute)
    and cross-entity claims (other S1 entities whose candidates carry the same attribute)."""
    p = p.astype(np.float32)
    i1 = i1.astype(np.int64)
    i23 = i23.astype(np.int64)
    feat["p1"] = p
    feat["p1_max_b"] = _group_max(p, i23)
    feat["p1_gap_b"] = feat["p1_max_b"] - p
    feat["p1_other_b"] = _group_second_max(p, i23)
    feat["p1_max_a"] = _group_max(p, i1)
    feat["p1_gap_a"] = feat["p1_max_a"] - p
    feat["p1_other_a"] = _group_second_max(p, i1)
    feat["p1_sum_a"] = _group_sum(p, i1) - p
    conf = (p > 0.5).astype(np.float32)
    feat["p1_conf_a"] = _group_sum(conf, i1) - conf
    for tag in SIB_TAGS:
        _, c23 = keys[tag]
        k = c23[i23]
        valid = k >= 0
        mult = int(c23.max()) + 2 if len(c23) else 2
        g = np.where(valid, i1 * mult + k, -1)
        cnt = (_group_count(g) - 1).astype(np.float32)
        cnt[~valid] = 0
        feat[f"sib_{tag}_n"] = cnt
        sp = _group_second_max(p, g)
        sp[~valid] = -1
        feat[f"sib_{tag}_p"] = sp.astype(np.float32)
        if tag in OTH_TAGS:
            op = _group_max_other_sub(p, np.where(valid, k, -1), i1)
            op[~valid] = -1
            feat[f"oth_{tag}_p"] = op.astype(np.float32)
        del g, k, valid, cnt, sp
    return feat
