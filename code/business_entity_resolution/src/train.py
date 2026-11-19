"""Training / validation driver (see runner.py for the shared, sharded candidate stages).

  block    -> raw blocking candidates + labels        (Runner.step_block)
  stage0   -> models/stage0.txt, p0 scores, keep masks (fit here, scoring in Runner.step_score0)
  feats    -> pruned candidates + features + labels    (Runner.step_feats)
  fit      -> models/stage1.txt, models/stage2.txt, validation report

Validation is entity-held-out: Source 1 entities are hashed into 5 folds; fold 0 is
never used for fitting and the reported macro-F0.5 is computed over *all* fold-0
entities (including those for which blocking produced no candidate).
"""
from __future__ import annotations

import gc
import time
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from decode import argmax_filter, build_lists, expected_f05_decode
from evaluate import macro_f05, truth_from_frame
from features import KEY_COLS, STAGE2_COLS, add_stage2_features, record_keys
from pipeline import LGB_PARAMS, LGB_PARAMS_S0, ModelBundle, entity_fold, fit_lgb, make_dataset, predict_rows, q16
from runner import Runner, log, release_memory


class Trainer(Runner):
    def __init__(self, *a, **kw):
        kw.setdefault("split", "train")
        super().__init__(*a, **kw)

    # ------------------------------------------------------------------ step: stage0 (fit + score + prune)
    def step_stage0(self, rounds: int = 400, neg_rate: float = 0.04):
        """Light candidate filter. Trained on every positive plus a random share of the negatives of
        the non-validation folds; then every candidate of every shard is scored and pruned per country."""
        path0 = os.path.join(self.model_dir, "stage0.txt")
        if not os.path.exists(path0):
            xs, ys = [], []
            cols = None
            rng = np.random.default_rng(0)
            for c in self.countries():
                for k in range(len(self.shards(c))):
                    tbl, y, fold = self.stage0_table(c, k)
                    cols = tbl.cols
                    sel = np.flatnonzero((fold != 0) & ((y == 1) | (rng.random(len(y)) < neg_rate)))
                    xs.append(tbl.rows(sel)[cols].to_numpy(np.float32))     # compact float32 blocks, no DataFrame concat
                    ys.append(y[sel])
                    del tbl, y, fold
                    gc.collect()
            X = np.concatenate(xs)
            y = np.concatenate(ys)
            del xs, ys
            gc.collect()
            log(f"stage0 training rows {len(y):,} (pos {int(y.sum()):,}); matrix {X.nbytes/2**30:.1f} GB")
            m0 = fit_lgb(X, y, LGB_PARAMS_S0, rounds, log=log, feature_name=cols)
            m0.save_model(path0)
            with open(os.path.join(self.model_dir, "stage0_cols.json"), "w") as f:
                json.dump(cols, f)
            imp = sorted(zip(m0.feature_importance("gain"), cols), reverse=True)[:12]
            log("  stage0 top features: " + ", ".join(f"{n}={g:.0f}" for g, n in imp))
            del X, y, m0
            gc.collect()
        self.step_score0(sweep=True)

    # ------------------------------------------------------------------ step: fit
    @staticmethod
    def _row_mask(fold: np.ndarray, smp: np.ndarray, which: str) -> np.ndarray:
        """which: 'train' -> sample rows outside fold 0; 'valid' -> all fold-0 rows; 'all' -> sample rows of every fold."""
        if which == "train":
            return (fold != 0) & smp
        if which == "valid":
            return fold == 0
        return smp

    def _load_matrix(self, cols: List[str], which: str, extra: int):
        """Selected rows of every feature shard as ONE pre-allocated float32 matrix with ``extra`` spare
        columns (filled later with the stage-2 features) plus a small meta frame. Avoids the 2-3x
        transient copies of DataFrame concatenation / column selection on 10M+ rows."""
        n = 0
        for c in self.countries():
            meta = self.read_meta(c, ["fold", "sample"])
            n += int(self._row_mask(meta["fold"].to_numpy(), meta["sample"].to_numpy() == 1, which).sum())
        X = np.empty((n, len(cols) + extra), np.float16)      # half precision: 255-bin GBM, half the memory
        meta_arrs = {k: np.empty(n, np.int64) for k in ("i1", "i23", "g1", "g23")}
        meta_arrs["y"] = np.empty(n, np.int8)
        meta_arrs["fold"] = np.empty(n, np.int8)
        meta_arrs["country"] = np.empty(n, np.int16)
        off = 0
        for ci, c in enumerate(self.countries()):
            for df in self.iter_feat_batches(c, cols + ["y", "fold", "sample", "i1", "i23"]):
                m = self._row_mask(df["fold"].to_numpy(), df["sample"].to_numpy() == 1, which)
                k = int(m.sum())
                if k == 0:
                    continue
                sub = df.loc[m]
                X[off:off + k, :len(cols)] = sub[cols].to_numpy(np.float32)
                i1 = sub["i1"].to_numpy(np.int64)
                i23 = sub["i23"].to_numpy(np.int64)
                meta_arrs["i1"][off:off + k] = i1
                meta_arrs["i23"][off:off + k] = i23
                meta_arrs["g1"][off:off + k] = i1 * 10 + ci
                meta_arrs["g23"][off:off + k] = i23 * 10 + ci
                meta_arrs["y"][off:off + k] = sub["y"].to_numpy(np.int8)
                meta_arrs["fold"][off:off + k] = sub["fold"].to_numpy(np.int8)
                meta_arrs["country"][off:off + k] = ci
                off += k
                del sub, df
        assert off == n, (off, n)
        release_memory()
        return X, pd.DataFrame(meta_arrs)

    def _predict_all_p1(self, models: Dict[str, "lgb.Booster"], cols: List[str]) -> Dict[str, np.ndarray]:
        """Out-of-fold stage-1 probability for every pruned candidate of every country."""
        out = {}
        for c in self.countries():
            preds = []
            for df in self.iter_feat_batches(c, cols + ["fold"]):
                fold = df["fold"].to_numpy()
                X = q16(df[cols].to_numpy(np.float32))
                p = np.empty(len(df), np.float32)
                mA = np.isin(fold, models["A_folds"])
                mB = np.isin(fold, models["B_folds"])
                rest = ~(mA | mB)
                if mA.any():
                    p[mA] = models["B"].predict(X[mA], num_threads=8)   # A-fold rows scored by the model that did not see them
                if mB.any():
                    p[mB] = models["A"].predict(X[mB], num_threads=8)
                if rest.any():
                    p[rest] = models["full"].predict(X[rest], num_threads=8)
                preds.append(p)
            out[c] = np.concatenate(preds)
            log(f"  stage-1 OOF predictions for {c}: {len(out[c]):,}")
        return out

    def step_fit(self, rounds1: int = 800, rounds2: int = 400, final: bool = False):
        cols = self.feature_cols()
        cols2 = cols + STAGE2_COLS
        n1c = len(cols)
        which = "all" if final else "train"
        X, meta = self._load_matrix(cols, which, extra=len(STAGE2_COLS))
        y = meta["y"].to_numpy()
        fold = meta["fold"].to_numpy()
        log(f"fit: training rows {len(y):,} ({len(cols)} features, positives {y.sum():,}); matrix {X.nbytes/2**30:.1f} GB (float16)")
        # ---- stage 1: two half models (for OOF stage-2 features) + full model.  One binned dataset of
        # the stage-1 columns is built through a row-batch view (no contiguous copy of the sub-matrix);
        # the half models train on row subsets of it, so peak memory stays ~ X + 1 byte/cell.
        A_folds, B_folds = ([1, 2], [3, 4]) if not final else ([0, 1, 2], [3, 4])
        t = time.time()
        dfull = make_dataset(X, y, cols, n_cols=n1c)
        release_memory()
        log(f"  stage-1 dataset binned in {time.time()-t:.0f}s")
        half = {}
        for name, folds in (("A", A_folds), ("B", B_folds)):
            idx = np.flatnonzero(np.isin(fold, folds))
            half[name] = fit_lgb(dfull.subset(idx), y[idx], LGB_PARAMS, rounds1, log=log)
            gc.collect()
        m1 = fit_lgb(dfull, y, LGB_PARAMS, rounds1, log=log)
        del dfull
        gc.collect()
        m1.save_model(os.path.join(self.model_dir, "stage1.txt"))
        models = {"A": half["A"], "B": half["B"], "full": m1, "A_folds": A_folds, "B_folds": B_folds}
        p1_all = self._predict_all_p1(models, cols)
        del half, models
        release_memory()
        # ---- stage-2 features from OOF p1 on the complete per-country tables
        s2_all: Dict[str, pd.DataFrame] = {}
        for c in self.countries():
            m = self.read_meta(c, ["i1", "i23"])
            keys = record_keys(*self.partition(c, columns=KEY_COLS))
            s2_all[c] = add_stage2_features(pd.DataFrame(index=range(len(m))), p1_all[c],
                                            m["i1"].to_numpy(np.int64), m["i23"].to_numpy(np.int64), keys)
            del keys, m
        del p1_all
        release_memory()

        def attach(Xm: np.ndarray, which: str) -> None:
            """Fill the spare columns of ``Xm`` with the stage-2 features of the selected rows.
            Row order inside each country == shard-file order, so the masks line up with s2_all."""
            off = 0
            for c in self.countries():
                mt = self.read_meta(c, ["fold", "sample"])
                msk = self._row_mask(mt["fold"].to_numpy(), mt["sample"].to_numpy() == 1, which)
                k = int(msk.sum())
                for j, col in enumerate(STAGE2_COLS):
                    Xm[off:off + k, n1c + j] = s2_all[c][col].to_numpy(np.float32)[msk]
                off += k
            assert off == len(Xm), (off, len(Xm))

        attach(X, which)
        m2 = fit_lgb(make_dataset(X, y, cols2), y, LGB_PARAMS, rounds2, log=log)
        m2.save_model(os.path.join(self.model_dir, "stage2.txt"))
        mb = ModelBundle(stage1=m1, stage2=m2, feature_cols=cols, blocking=vars(self.bcfg), prune=vars(self.pcfg))
        with open(os.path.join(self.model_dir, "stage0_cols.json")) as f:
            mb.stage0_cols = json.load(f)
        prev = os.path.join(self.model_dir, "meta.json")
        if final and os.path.exists(prev):          # keep the decode strategy selected on the held-out fold
            with open(prev) as f:
                mb.decode = json.load(f).get("decode", {})
        mb.save(self.model_dir)
        del X, meta, y, fold
        gc.collect()
        if final:
            return
        Xv, va = self._load_matrix(cols, "valid", extra=len(STAGE2_COLS))
        attach(Xv, "valid")
        p2 = predict_rows(m2, Xv)
        p1 = Xv[:, n1c + STAGE2_COLS.index("p1")].astype(np.float32)
        del Xv
        gc.collect()
        self.report(va, p1, p2, mb)

    # ------------------------------------------------------------------ validation report
    def report(self, fv: pd.DataFrame, p1: np.ndarray, p2: np.ndarray, mb: ModelBundle):
        truth = truth_from_frame(self.gt)
        all_ids = self.all_s1_ids()
        val_entities = all_ids[entity_fold(all_ids) == 0]
        s1_ids = np.empty(len(fv), dtype=object)
        s23_ids = np.empty(len(fv), dtype=object)
        cn = fv["country"].to_numpy()
        for ci, c in enumerate(self.countries()):
            m = cn == ci
            s1_ids[m] = self.s1_ids(c)[fv["i1"].to_numpy()[m]]
            s23_ids[m] = self.s23_ids(c)[fv["i23"].to_numpy()[m]]
        g1 = fv["g1"].to_numpy()
        g23 = fv["g23"].to_numpy()
        y = fv["y"].to_numpy()
        from sklearn.metrics import average_precision_score
        log(f"  valid AP stage1 {average_precision_score(y, p1):.5f}  stage2 {average_precision_score(y, p2):.5f}")
        idx = np.arange(len(fv))
        best = None
        results = {}
        for name, p in [("p1", p1), ("p2", p2)]:
            for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
                for am in [False, True]:
                    keep = p >= thr
                    if am:
                        keep &= argmax_filter(p, g23)
                    lists = build_lists(keep, s1_ids, s23_ids, idx, idx)
                    sc = macro_f05(lists, truth, val_entities)
                    results[f"{name} thr={thr} argmax={am}"] = sc
                    if best is None or sc > best[1]:
                        best = (f"{name} thr={thr} argmax={am}", sc)
            for am in [False, True]:
                pp = p.copy()
                if am:
                    pp = np.where(argmax_filter(p, g23), pp, 0)
                keep = expected_f05_decode(pp, g1)
                lists = build_lists(keep, s1_ids, s23_ids, idx, idx)
                sc = macro_f05(lists, truth, val_entities)
                results[f"{name} expectedF05 argmax={am}"] = sc
                if sc > best[1]:
                    best = (f"{name} expectedF05 argmax={am}", sc)
        for k, v in results.items():
            log(f"  {k}: macro-F0.5 = {v:.5f}")
        log(f"BEST {best[0]} -> {best[1]:.5f}")
        mb.decode = {"best": best[0], "results": results}
        mb.save(self.model_dir)
        keep = y == 1
        lists = build_lists(keep, s1_ids, s23_ids, idx, idx)
        log(f"  oracle (perfect decisions on candidates): {macro_f05(lists, truth, val_entities):.5f}")
        # persist validation predictions for offline decoder tuning
        np.savez_compressed(os.path.join(self.work_dir, "valid_preds.npz"), p1=p1, p2=p2, y=y, g1=g1, g23=g23,
                            s1=s1_ids.astype(str), s23=s23_ids.astype(str))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["block", "stage0", "feats", "fit", "all"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--countries", default=None)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--rounds1", type=int, default=800)
    ap.add_argument("--rounds2", type=int, default=400)
    ap.add_argument("--sample", type=float, default=0.4)
    ap.add_argument("--shard-rows", type=int, default=2_200_000)
    ap.add_argument("--block-sample", type=float, default=1.0, help="fraction of S2/S3 rows to block (diagnostics only)")
    ap.add_argument("--bcfg", default="", help="blocking overrides, e.g. k_blend=20,min_blend_cos=0.1")
    ap.add_argument("--pcfg", default="", help="pruning overrides, e.g. p0_min=0.005,max_per_a=30")
    a = ap.parse_args()
    t = Trainer(a.data, a.cache, a.work, a.models, a.lexicon, train_sample=a.sample, shard_rows=a.shard_rows)
    for cfg, spec in ((t.bcfg, a.bcfg), (t.pcfg, a.pcfg)):
        for kv in filter(None, spec.split(",")):
            k, v = kv.split("=")
            cur = getattr(cfg, k)
            setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true"))
    log(f"blocking config: {vars(t.bcfg)}")
    log(f"pruning config: {vars(t.pcfg)}")
    cs = a.countries.split(",") if a.countries else None
    if a.step in ("block", "all"):
        t.step_block(cs, sample=a.block_sample)
    if a.step in ("stage0", "all"):
        t.step_stage0()
    if a.step in ("feats", "all"):
        t.step_feats(cs)
    if a.step in ("fit", "all"):
        t.step_fit(a.rounds1, a.rounds2, final=a.final)
