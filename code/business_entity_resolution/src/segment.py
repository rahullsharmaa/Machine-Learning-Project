"""Word segmentation for concatenated names (web-domain style records such as
``hospitalunitedprivate.com`` -> ``hospital united private``).

A unigram language model over name tokens is estimated from the Source 1 business
names of the split being processed (no external data), and the most probable
segmentation is found with dynamic programming (Viterbi over token boundaries).
"""
from __future__ import annotations

import collections
import math
import re
from typing import Dict, Iterable, List

_TOKEN_RE = re.compile(r"[a-z]+")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g"})


def leet_fix(tok: str) -> str:
    """Undo digit-for-letter substitutions inside alphabetic tokens (``ko1hapur`` -> ``kolhapur``).

    Only tokens that mix letters and digits, where letters dominate, are changed; pure
    numbers and codes such as ``b12`` stay untouched.
    """
    if tok.isalpha() or tok.isdigit():
        return tok
    n_alpha = sum(c.isalpha() for c in tok)
    n_digit = len(tok) - n_alpha
    if n_alpha >= 3 and n_alpha > n_digit:
        return tok.translate(_LEET)
    return tok


class Segmenter:
    def __init__(self, counts: Dict[str, int], max_len: int = 20):
        total = float(sum(counts.values())) or 1.0
        self.logp = {w: math.log(c / total) for w, c in counts.items()}
        self.max_len = max_len
        # cost of an unknown fragment of length L (per character), worse than any known token
        self.unk = math.log(1.0 / total) - 2.0
        self.cache: Dict[str, str] = {}

    @classmethod
    def from_names(cls, names: Iterable[str], min_count: int = 3) -> "Segmenter":
        from normalize import fold
        cnt: collections.Counter = collections.Counter()
        for nm in names:
            cnt.update(_TOKEN_RE.findall(fold(nm)))
        return cls({w: c for w, c in cnt.items() if c >= min_count and len(w) >= 2})

    def segment(self, s: str) -> str:
        """Split an alphabetic string into the most probable sequence of known tokens."""
        v = self.cache.get(s)
        if v is not None:
            return v
        n = len(s)
        if n == 0:
            return s
        best = [(-math.inf, 0)] * (n + 1)
        best[0] = (0.0, 0)
        for i in range(1, n + 1):
            lo = max(0, i - self.max_len)
            cand = (-math.inf, 0)
            for j in range(lo, i):
                prev = best[j][0]
                if prev == -math.inf:
                    continue
                w = s[j:i]
                lp = self.logp.get(w)
                if lp is None:
                    if i - j > 1:
                        continue
                    lp = self.unk                      # single unknown character
                sc = prev + lp
                if sc > cand[0]:
                    cand = (sc, j)
            best[i] = cand
        if best[n][0] == -math.inf:
            self.cache[s] = s
            return s
        out: List[str] = []
        i = n
        while i > 0:
            j = best[i][1]
            out.append(s[j:i])
            i = j
        out.reverse()
        # merge runs of single unknown characters back together
        merged: List[str] = []
        for t in out:
            if merged and len(t) == 1 and t not in self.logp and len(merged[-1]) == 1 and merged[-1] not in self.logp:
                merged[-1] += t
            else:
                merged.append(t)
        res = " ".join(merged)
        self.cache[s] = res
        return res

    def save(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"logp": self.logp, "max_len": self.max_len}, f)

    @classmethod
    def load(cls, path: str) -> "Segmenter":
        import json
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        seg = cls({}, d["max_len"])
        seg.logp = d["logp"]
        seg.unk = min(seg.logp.values(), default=-20.0) - 2.0
        return seg
