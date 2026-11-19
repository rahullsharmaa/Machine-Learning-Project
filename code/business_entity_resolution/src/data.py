"""Data loading and memory-safe, cached normalisation.

Normalisation of the ~12M records per split is done by worker processes that each read
their own slice of the TSV (``spawn`` context, so nothing large is inherited from the
parent) and write a parquet part.  Peak memory therefore stays at a few GB.
"""
from __future__ import annotations

import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import pandas as pd

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str, **kw) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str, quoting=3, **kw)
    for c in SRC_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df


def read_ground_truth(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str, quoting=3)


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return max(n - 1, 0)


def _normalize_chunk(args):
    path, skip, nrows, lexicon_path, seg_path, out_path = args
    from normalize import Transliterator, load_lexicon, normalize_frame
    from segment import Segmenter
    tr = Transliterator(load_lexicon(lexicon_path))
    seg = Segmenter.load(seg_path) if seg_path and os.path.exists(seg_path) else None
    df = pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str, quoting=3, header=0,
                     skiprows=range(1, skip + 1), nrows=nrows)
    for c in SRC_COLS:
        df[c] = df[c].fillna("").astype(str)
    out = normalize_frame(df, tr, seg)
    out.to_parquet(out_path, index=False)
    return out_path, len(out)


def build_segmenter(s1_path: str, out_path: str, log=print) -> str:
    """Estimate the word-segmentation unigram model from the Source 1 names of a split."""
    if os.path.exists(out_path):
        return out_path
    from segment import Segmenter
    t = time.time()
    names = pd.read_csv(s1_path, sep="\t", keep_default_na=False, dtype=str, quoting=3, usecols=["business_name"])["business_name"]
    seg = Segmenter.from_names(names.tolist())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seg.save(out_path)
    log(f"  segmenter vocabulary: {len(seg.logp):,} tokens in {time.time()-t:.0f}s")
    return out_path


def normalize_file(path: str, lexicon_path: str, out_dir: str, seg_path: str = "", n_jobs: int = 6, chunk_rows: int = 200_000, log=print) -> str:
    """Normalise one TSV into ``out_dir/<name>.parquet`` (parts are merged). Returns the parquet path."""
    name = os.path.splitext(os.path.basename(path))[0]
    final = os.path.join(out_dir, f"{name}.parquet")
    if os.path.exists(final):
        return final
    os.makedirs(out_dir, exist_ok=True)
    total = count_lines(path)
    tasks = []
    part_dir = os.path.join(out_dir, f"_{name}_parts")
    os.makedirs(part_dir, exist_ok=True)
    for k, skip in enumerate(range(0, total, chunk_rows)):
        tasks.append((path, skip, min(chunk_rows, total - skip), lexicon_path, seg_path, os.path.join(part_dir, f"part_{k:04d}.parquet")))
    t = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
        for _ in ex.map(_normalize_chunk, tasks):
            pass
    parts = sorted(glob.glob(os.path.join(part_dir, "part_*.parquet")))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(final, index=False)
    for p in parts:
        os.remove(p)
    os.rmdir(part_dir)
    log(f"  normalised {name}: {len(df):,} rows in {time.time()-t:.0f}s")
    return final


def split_paths(data_dir: str, split: str):
    return [os.path.join(data_dir, split, f"{split}_source{i}.tsv") for i in (1, 2, 3)]


def ensure_normalized(data_dir: str, split: str, cache_dir: str, lexicon_path: str, n_jobs: int = 6, log=print):
    """Normalise (if needed) and return the three parquet paths of a split."""
    paths = split_paths(data_dir, split)
    seg_path = build_segmenter(paths[0], os.path.join(cache_dir, f"{split}_segmenter.json"), log=log)
    return [normalize_file(p, lexicon_path, cache_dir, seg_path=seg_path, n_jobs=n_jobs, log=log) for p in paths]


def load_normalized(data_dir: str, split: str, cache_dir: str, lexicon_path: str, n_jobs: int = 6, log=print):
    """Return normalised (s1, s23) frames for a split, computing and caching them if needed."""
    outs = ensure_normalized(data_dir, split, cache_dir, lexicon_path, n_jobs=n_jobs, log=log)
    s1 = pd.read_parquet(outs[0])
    s23 = pd.concat([pd.read_parquet(outs[1]), pd.read_parquet(outs[2])], ignore_index=True)
    return s1, s23


def list_countries(paths) -> list:
    """Sorted union of the country labels that occur in the given parquet files (open set)."""
    cs = set()
    for p in paths:
        cs |= set(pd.read_parquet(p, columns=["country"])["country"].unique().tolist())
    return sorted(cs)


def load_partition(paths, country: str, columns=None):
    """Load (s1, s23) rows of one country only (memory-safe: other countries are never materialised)."""
    flt = [("country", "==", country)]
    s1 = pd.read_parquet(paths[0], columns=columns, filters=flt).reset_index(drop=True)
    s23 = pd.concat([pd.read_parquet(paths[1], columns=columns, filters=flt),
                     pd.read_parquet(paths[2], columns=columns, filters=flt)], ignore_index=True)
    return s1, s23


def read_parquet_lean(path: str, columns=None) -> "pd.DataFrame":
    """Parquet -> pandas without keeping a second (Arrow) copy of a very large table alive."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=columns)
    df = t.to_pandas(self_destruct=True, split_blocks=True)
    del t
    return df
