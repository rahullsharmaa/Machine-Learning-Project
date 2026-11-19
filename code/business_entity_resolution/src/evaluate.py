"""Official metric: macro-averaged F_0.5 over Source 1 entities (singletons included)."""
from __future__ import annotations

from typing import Dict, Iterable, Set

import numpy as np
import pandas as pd


def f05(pred: Set[str], truth: Set[str]) -> float:
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def parse_id_list(s: str) -> Set[str]:
    return set(x for x in s.split(",") if x) if s else set()


def macro_f05(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]], entities: Iterable[str] | None = None) -> float:
    ents = list(entities) if entities is not None else list(truth.keys())
    scores = [f05(pred.get(e, set()), truth.get(e, set())) for e in ents]
    return float(np.mean(scores)) if scores else 0.0


def truth_from_frame(gt: pd.DataFrame) -> Dict[str, Set[str]]:
    return {a: parse_id_list(m) for a, m in zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].values)}


def per_entity_scores(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]], entities: Iterable[str]) -> pd.DataFrame:
    rows = []
    for e in entities:
        p, t = pred.get(e, set()), truth.get(e, set())
        rows.append((e, len(p), len(t), len(p & t), f05(p, t)))
    return pd.DataFrame(rows, columns=["entity", "n_pred", "n_true", "tp", "f05"])
