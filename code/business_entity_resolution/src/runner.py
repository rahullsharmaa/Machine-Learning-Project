"""Shared, memory-bounded driver for the candidate stages of both training and inference.

Every country partition is processed in contiguous S2/S3 *shards* of at most ``shard_rows``
records (positions ``i23`` stay global within the country).  Each stage writes one file
per (split, country, shard) into ``work_dir`` and later stages stream those files, so the
peak memory footprint is bounded by one shard (~80M raw candidate pairs) rather than by
the whole country:

  block   -> work/<split>_cand_<c>_s<k>.parquet   raw blocking candidates (+ labels for train)
  stage0  -> work/<split>_p0_<c>_s<k>.npy         stage-0 filter probabilities
             work/<split>_keep_<c>_s<k>.npy       pruning mask (computed per country)
  feats   -> work/<split>_feat_<c>_s<k>.parquet   pruned candidates + full pair features
"""
from __future__ import annotations

import gc
import json
import os
import time
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from blocking import BlockingConfig, _block_partition_once
from data import ensure_normalized, list_countries, load_partition, read_ground_truth, read_parquet_lean
from features import compute_features
from pipeline import PruneConfig, Stage0Table, entity_fold, labels_for, owner_map, prune_candidates, prune_ranks

ID_COLS = ["entity_id", "country"]
BLOCK_COLS = ["entity_id", "country", "name_core", "name_nospace", "addr", "addr_street"]
STAGE0_COLS = BLOCK_COLS + ["is_nonlatin", "is_domain"]
NON_FEATURE = {"i1", "i23", "y", "fold", "s1_id", "s23_id", "country", "g1", "g23", "sample"}


def release_memory():
    """gc + glibc malloc_trim so freed whole-country arrays go back to the OS between countries
    (otherwise RSS keeps the high-water mark and the next country's arrays stack on top of it)."""
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Runner:
    def __init__(self, data_dir: str, cache_dir: str, work_dir: str, model_dir: str, lexicon: str, split: str = "train",
                 n_jobs: int = 6, train_sample: float = 0.4, shard_rows: int = 2_200_000):
        self.data_dir, self.cache_dir, self.work_dir, self.model_dir, self.lexicon = data_dir, cache_dir, work_dir, model_dir, lexicon
        self.split = split
        self.n_jobs = n_jobs
        self.train_sample = train_sample
        self.shard_rows = shard_rows
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self.labelled = split == "train"
        if self.labelled:
            self.gt = read_ground_truth(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
            self.owner = owner_map(self.gt)
        else:
            self.gt, self.owner = None, {}
        self.bcfg = BlockingConfig()
        self.pcfg = PruneConfig()
        self._countries = None
        self._paths = None
        self._part_cache: Dict[tuple, tuple] = {}

    # ------------------------------------------------------------------ data
    def paths(self):
        if self._paths is None:
            self._paths = ensure_normalized(self.data_dir, self.split, self.cache_dir, self.lexicon, n_jobs=self.n_jobs, log=log)
        return self._paths

    def countries(self) -> List[str]:
        if self._countries is None:
            self._countries = [c for c in list_countries(self.paths()) if len(self.s1_ids(c)) > 0 and len(self.s23_ids(c)) > 0]
        return self._countries

    def partition(self, country: str, ids_only: bool = False, columns=None):
        """(s1, s23) frames of one country; only one full partition is kept in memory at a time."""
        key = (country, ids_only, tuple(columns) if columns else None)
        if key in self._part_cache:
            return self._part_cache[key]
        if not ids_only:
            self._part_cache = {k: v for k, v in self._part_cache.items() if k[1]}   # drop other full partitions
            gc.collect()
        s1c, s23c = load_partition(self.paths(), country, columns=ID_COLS if ids_only else columns)
        self._part_cache[key] = (s1c, s23c)
        return s1c, s23c

    def s1_ids(self, country: str) -> np.ndarray:
        return self.partition(country, ids_only=True)[0]["entity_id"].to_numpy(dtype=object)

    def s23_ids(self, country: str) -> np.ndarray:
        return self.partition(country, ids_only=True)[1]["entity_id"].to_numpy(dtype=object)

    def all_s1_ids(self) -> np.ndarray:
        return pd.read_parquet(self.paths()[0], columns=["entity_id"])["entity_id"].to_numpy(dtype=object)

    # ------------------------------------------------------------------ shards
    def shards(self, country: str) -> List[Tuple[int, int]]:
        """Contiguous S2/S3 row ranges of one country. The layout is persisted in the work directory
        the first time it is computed so that every later step (and re-run) sees the same shards."""
        if not hasattr(self, "_shards"):
            self._shards: Dict[str, List[Tuple[int, int]]] = {}
        if country in self._shards:
            return self._shards[country]
        path = os.path.join(self.work_dir, f"{self.split}_shards_{country}.json")
        if os.path.exists(path):
            with open(path) as f:
                sh = [tuple(x) for x in json.load(f)]
        else:
            n23 = len(self.s23_ids(country))
            n = max(1, int(np.ceil(n23 / self.shard_rows)))
            b = np.linspace(0, n23, n + 1).astype(int)
            sh = [(int(b[i]), int(b[i + 1])) for i in range(n)]
            with open(path, "w") as f:
                json.dump(sh, f)
        self._shards[country] = sh
        return sh

    def fpath(self, kind: str, country: str, k: int, ext: str = "parquet") -> str:
        return os.path.join(self.work_dir, f"{self.split}_{kind}_{country}_s{k}.{ext}")

    def shard_files(self, kind: str, country: str, ext: str = "parquet") -> List[str]:
        return [self.fpath(kind, country, k, ext) for k in range(len(self.shards(country)))]

    # ------------------------------------------------------------------ step: block
    def step_block(self, countries=None, sample: float = 1.0):
        """Blocking per country shard. ``sample`` < 1 blocks a random subset of the S2/S3 rows of each
        country against the full S1 (fast recall / density diagnostics; nothing is cached)."""
        for c in countries or self.countries():
            s1c, s23c = self.partition(c, columns=BLOCK_COLS)
            if sample < 1.0:
                rng = np.random.default_rng(0)
                sub = s23c.iloc[np.sort(rng.choice(len(s23c), int(sample * len(s23c)), replace=False))].reset_index(drop=True)
                log(f"blocking {c} (sample {sample}): S1={len(s1c):,} S23={len(sub):,}")
                cand = _block_partition_once(s1c, sub, self.bcfg, log=log)
                if self.labelled:
                    cand["y"] = labels_for(cand, s1c, sub, self.owner)
                    n_true = sum(1 for e in sub["entity_id"].values if e in self.owner)
                    log(f"  {c}: blocking recall {cand['y'].sum()/max(n_true,1):.5f} ({cand['y'].sum():,}/{n_true:,}), density {len(cand)/len(s1c):.1f}")
                    self.blocking_report(cand, s1c, sub)
                del cand, sub
                gc.collect()
                continue
            found = n_true = n_pairs = 0
            for k, (lo, hi) in enumerate(self.shards(c)):
                out = self.fpath("cand", c, k)
                if os.path.exists(out):
                    continue
                shard = s23c.iloc[lo:hi].reset_index(drop=True)
                log(f"blocking {c} shard {k + 1}/{len(self.shards(c))}: S1={len(s1c):,} S23 rows {lo:,}-{hi:,}")
                cand = _block_partition_once(s1c, shard, self.bcfg, log=log)
                if self.labelled:
                    cand["y"] = labels_for(cand, s1c, shard, self.owner)
                    nt = sum(1 for e in shard["entity_id"].values if e in self.owner)
                    log(f"  {c} s{k}: blocking recall {cand['y'].sum()/max(nt,1):.5f} ({cand['y'].sum():,}/{nt:,}), density {len(cand)/len(s1c):.1f}")
                    self.blocking_report(cand, s1c, shard)
                    found += int(cand["y"].sum()); n_true += nt
                n_pairs += len(cand)
                cand["i23"] = (cand["i23"].to_numpy(np.int64) + lo).astype(np.int32)
                cand.to_parquet(out, index=False, row_group_size=2_000_000)
                del cand, shard
                gc.collect()
            if n_pairs:
                msg = f"  {c}: total candidates {n_pairs:,} ({n_pairs/len(s1c):.1f} per S1)"
                if self.labelled and n_true:
                    msg += f", blocking recall {found/n_true:.5f} ({found:,}/{n_true:,})"
                log(msg)
            del s1c, s23c
            self._part_cache = {k: v for k, v in self._part_cache.items() if k[1]}
            gc.collect()

    def blocking_report(self, cand: pd.DataFrame, s1c: pd.DataFrame, s23c: pd.DataFrame, n_show: int = 15):
        """Per-net recall and a sample of the true pairs that blocking missed (train only)."""
        from blocking import RANK_COLS
        y = cand["y"].to_numpy() == 1
        for col in RANK_COLS:
            hit = cand[col].to_numpy() < 99
            log(f"    net {col:9s}: pairs {hit.sum():>12,}  true pairs found {(hit & y).sum():>9,}  share of found positives {(hit & y).sum()/max(y.sum(),1):.4f}")
        owner = self.owner
        s1_pos = {e: i for i, e in enumerate(s1c["entity_id"].to_numpy(dtype=object))}
        found = set(zip(cand.loc[y, "i1"].to_numpy(), cand.loc[y, "i23"].to_numpy()))
        missed = []
        for j, e in enumerate(s23c["entity_id"].to_numpy(dtype=object)):
            o = owner.get(e)
            if o is None:
                continue
            i = s1_pos.get(o)
            if i is not None and (i, j) not in found:
                missed.append((i, j))
        log(f"    missed true pairs: {len(missed):,}")
        if missed:
            mi = np.array([m[0] for m in missed]); mj = np.array([m[1] for m in missed])
            a23 = s23c["addr"].to_numpy(dtype=object)[mj]
            n23 = s23c["name_nospace"].to_numpy(dtype=object)[mj]
            n1 = s1c["name_nospace"].to_numpy(dtype=object)[mi]
            amb = s1c["name_nospace"].value_counts().reindex(n1).to_numpy() >= 2
            empty = np.array([a == "" for a in a23])
            same = n23 == n1
            log(f"      misses with empty S23 address: {empty.mean():.3f}; same core name: {same.mean():.3f}; "
                f"S1 name ambiguous (>=2 S1 share it): {amb.mean():.3f}; empty addr & different name: {(empty & ~same).mean():.3f}; "
                f"empty addr & same name & ambiguous: {(empty & same & amb).mean():.3f}; non-empty addr & different name: {(~empty & ~same).mean():.3f}")
            rng = np.random.default_rng(1)
            for i, j in [missed[k] for k in rng.choice(len(missed), min(n_show, len(missed)), replace=False)]:
                log(f"      MISS  S23 [{s23c['name_core'].iat[j]!r} | {s23c['addr'].iat[j]!r}]  vs  S1 [{s1c['name_core'].iat[i]!r} | {s1c['addr'].iat[i]!r}]")

    # ------------------------------------------------------------------ stage 0 (filter) scoring + pruning
    def stage0_table(self, c: str, k: int):
        """Lazy stage-0 feature table of one shard (+ labels and folds when available)."""
        cand = read_parquet_lean(self.fpath("cand", c, k))
        s1c, s23c = self.partition(c, columns=STAGE0_COLS)
        y = cand["y"].to_numpy(np.int8) if "y" in cand else None
        fold = entity_fold(s1c["entity_id"].to_numpy(dtype=object))[cand["i1"].to_numpy()]
        t = time.time()
        tbl = Stage0Table(cand, s1c, s23c)
        log(f"  stage0 table {c} s{k}: {tbl.n:,} pairs, {len(tbl.cols)} features in {time.time()-t:.0f}s")
        del cand
        gc.collect()
        return tbl, y, fold

    def load_stage0_model(self):
        import lightgbm as lgb
        m0 = lgb.Booster(model_file=os.path.join(self.model_dir, "stage0.txt"))
        with open(os.path.join(self.model_dir, "stage0_cols.json")) as f:
            cols = json.load(f)
        return m0, cols

    def step_score0(self, countries=None, sweep: bool = False):
        """Score every raw candidate with the stage-0 filter and compute the per-country pruning masks."""
        m0, cols = self.load_stage0_model()
        for c in countries or self.countries():
            ks = range(len(self.shards(c)))
            if all(os.path.exists(self.fpath("keep", c, k, "npy")) for k in ks):
                continue
            for k in ks:
                out = self.fpath("p0", c, k, "npy")
                if os.path.exists(out):
                    continue
                tbl, _, _ = self.stage0_table(c, k)
                t = time.time()
                p0 = tbl.predict(m0, cols)
                log(f"  stage0 scored {c} s{k} in {time.time()-t:.0f}s")
                np.save(out, p0)
                del tbl, p0
                gc.collect()
            self.prune_country(c, sweep=sweep)

    # The challenge requires an average candidate density strictly inside [15, 30] per S1 entity.  The
    # configured operating point lands well inside that band on the training countries; on an unseen
    # country (France is test-only) the stage-0 score distribution can shift, so the probability floor
    # is re-tuned per country by bisection whenever the resulting density leaves the safety band below.
    DENSITY_BAND = (16.0, 28.0)
    DENSITY_TARGET = 20.0

    def _density_guard(self, c: str, cand_pos: pd.DataFrame, p0: np.ndarray, ranks, keep: np.ndarray, n1: int):
        dens = keep.sum() / max(n1, 1)
        lo_b, hi_b = self.DENSITY_BAND
        if lo_b <= dens <= hi_b:
            return keep, self.pcfg
        log(f"  density guard {c}: {dens:.2f} per S1 with p0_min={self.pcfg.p0_min:g} is outside [{lo_b}, {hi_b}]; re-tuning p0_min")
        lo, hi = 1e-7, 0.5
        best, best_cfg, best_gap = keep, self.pcfg, abs(dens - self.DENSITY_TARGET)
        for _ in range(20):
            mid = float(np.sqrt(lo * hi))
            cfg = PruneConfig(**{**vars(self.pcfg), "p0_min": mid})
            k2 = prune_candidates(cand_pos, p0, cfg, ranks)
            d2 = k2.sum() / max(n1, 1)
            gap = abs(d2 - self.DENSITY_TARGET)
            if gap < best_gap:
                best, best_cfg, best_gap = k2, cfg, gap
            if d2 > self.DENSITY_TARGET:
                lo = mid  # too dense -> raise the probability floor
            else:
                hi = mid
            if gap < 0.25:
                break
        log(f"  density guard {c}: p0_min {self.pcfg.p0_min:g} -> {best_cfg.p0_min:g}, density {best.sum()/max(n1,1):.2f} per S1")
        return best, best_cfg

    def prune_country(self, c: str, sweep: bool = False):
        """Global (whole-country) pruning from the per-shard stage-0 scores; writes one keep mask per shard.
        Only (i1, i23, p0[, y]) are held for the whole country: <= 13 bytes/row."""
        ks = range(len(self.shards(c)))
        lens, i1, i23, y = [], [], [], []
        for k in ks:
            m = pq.read_table(self.fpath("cand", c, k), columns=["i1", "i23"] + (["y"] if self.labelled else []))
            lens.append(m.num_rows)
            i1.append(m.column("i1").to_numpy().astype(np.int32))
            i23.append(m.column("i23").to_numpy().astype(np.int32))
            if self.labelled:
                y.append(m.column("y").to_numpy().astype(np.int8))
            del m
        i1 = np.concatenate(i1)
        i23 = np.concatenate(i23)
        y = np.concatenate(y) if self.labelled else None
        p0 = np.concatenate([np.load(self.fpath("p0", c, k, "npy")).astype(np.float32, copy=False) for k in ks])
        cand_pos = pd.DataFrame({"i1": i1, "i23": i23}, copy=False)
        del i1, i23
        gc.collect()
        n1 = len(self.s1_ids(c))
        ranks = prune_ranks(cand_pos, p0)
        keep = prune_candidates(cand_pos, p0, self.pcfg, ranks)
        keep, used_cfg = self._density_guard(c, cand_pos, p0, ranks, keep, n1)
        msg = f"  prune {c}: keep {keep.mean():.3f} of {len(keep):,} pairs -> density {keep.sum()/n1:.2f} per S1 (p0_min={used_cfg.p0_min:g})"
        if y is not None:
            fold = entity_fold(self.s1_ids(c))[cand_pos["i1"].to_numpy()]
            mv = fold == 0
            msg += (f"; positive recall fold0 {(keep[mv] & (y[mv]==1)).sum()/max((y[mv]==1).sum(),1):.5f}"
                    f" (all {(keep & (y==1)).sum()/max((y==1).sum(),1):.5f})")
        log(msg)
        if sweep and y is not None:
            pos = y == 1
            for pmin in [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]:
                for mpa in [40, 60]:
                    cfg = PruneConfig(**{**vars(self.pcfg), "p0_min": pmin, "max_per_a": mpa})
                    k2 = prune_candidates(cand_pos, p0, cfg, ranks)
                    log(f"      p0_min={pmin} max_per_a={mpa}: density {k2.sum()/n1:.2f}, positive recall {(k2 & pos).sum()/max(pos.sum(),1):.5f}")
                    del k2
            del pos
        off = 0
        for k, n in zip(ks, lens):
            np.save(self.fpath("keep", c, k, "npy"), keep[off:off + n])
            off += n
        del cand_pos, p0, ranks, keep, y
        gc.collect()
        release_memory()

    # ------------------------------------------------------------------ step: feats
    def step_feats(self, countries=None):
        for c in countries or self.countries():
            for k in range(len(self.shards(c))):
                out = self.fpath("feat", c, k)
                if os.path.exists(out):
                    continue
                cand = read_parquet_lean(self.fpath("cand", c, k))
                p0 = np.load(self.fpath("p0", c, k, "npy"))
                keep = np.load(self.fpath("keep", c, k, "npy"))
                cand = cand[keep].reset_index(drop=True)
                p0 = p0[keep]
                del keep
                gc.collect()
                s1c, s23c = self.partition(c)
                log(f"features {c} s{k}: {len(cand):,} pruned candidates")
                feat = compute_features(s1c, s23c, cand, log=log)
                feat["p0"] = p0
                i1 = cand["i1"].to_numpy()
                feat["i1"] = i1
                feat["i23"] = cand["i23"].to_numpy()
                if self.labelled:
                    ids = s1c["entity_id"].to_numpy(dtype=object)
                    feat["y"] = cand["y"].to_numpy()
                    feat["fold"] = entity_fold(ids)[i1]
                    h = np.array([zlib.crc32((e + "#s").encode()) % 1000 for e in ids], dtype=np.int16)
                    feat["sample"] = (h[i1] < int(self.train_sample * 1000)).astype(np.int8)   # entity-level subsample flag
                feat.to_parquet(out, index=False, row_group_size=1_000_000)
                del feat, cand, p0
                gc.collect()

    # ------------------------------------------------------------------ streaming helpers over feature shards
    def feature_cols(self) -> List[str]:
        c0 = self.countries()[0]
        schema = pq.read_schema(self.fpath("feat", c0, 0))
        return [n for n in schema.names if n not in NON_FEATURE]

    def iter_feat_batches(self, c: str, columns: List[str], batch_size: int = 1_000_000):
        for path in self.shard_files("feat", c):
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
                yield batch.to_pandas()

    def read_meta(self, c: str, columns: List[str]) -> pd.DataFrame:
        parts = [pq.read_table(p, columns=columns).to_pandas() for p in self.shard_files("feat", c)]
        return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
