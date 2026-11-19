"""Mine an Indic-script -> Latin token lexicon from the training ground truth.

For every matched (Source 1, Source 2/3) pair where the S2/S3 name is written in an
Indic script, the Indic tokens are aligned positionally with the Latin tokens of the
Source 1 name (after removing legal-form tokens on both sides when the lengths differ).
Co-occurrence counts are accumulated and, per Indic token, the most frequent Latin
token wins.  Only training data is used; the resulting JSON is a pipeline artifact.
"""
from __future__ import annotations

import collections
import json
import re
from typing import Dict, Iterable, Tuple

from normalize import LEGAL_TOKENS, _INDIC_TOKEN_RE, _INDIC_RE, fold

_LAT_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _latin_tokens(name: str):
    s = fold(name).replace("&", " and ").replace("+", " and ")
    s = re.sub(r"(?<=\w)'s\b", "s", s).replace("'", "")
    return _LAT_TOKEN_RE.findall(s)


def _indic_tokens(name: str):
    return _INDIC_TOKEN_RE.findall(name)


def mine_lexicon(pairs: Iterable[Tuple[str, str]], min_count: int = 2) -> Dict[str, str]:
    """pairs: iterable of (s1_name_latin, s23_name_indic)."""
    co: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for lat_name, ind_name in pairs:
        if not _INDIC_RE.search(ind_name):
            continue
        it = _indic_tokens(ind_name)
        lt = _latin_tokens(lat_name)
        if not it or not lt:
            continue
        if len(it) != len(lt):
            lt2 = [t for t in lt if t not in LEGAL_TOKENS]
            if len(lt2) == len(it):
                lt = lt2
            else:
                # try aligning legal-suffix-free prefix only
                m = min(len(it), len(lt))
                if m == 0:
                    continue
                for a, b in zip(it[:m], lt[:m]):
                    co[a][b] += 0.5
                continue
        for a, b in zip(it, lt):
            co[a][b] += 1
    lex = {}
    for a, cnt in co.items():
        b, n = cnt.most_common(1)[0]
        if n >= min_count:
            lex[a] = b
    return lex


def save_lexicon(lex: Dict[str, str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lex, f, ensure_ascii=False, indent=0, sort_keys=True)


def load_lexicon(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
