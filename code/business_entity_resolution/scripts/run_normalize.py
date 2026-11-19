"""Normalise every split and cache the result (safe with the 'spawn' start method)."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data import load_normalized  # noqa: E402

if __name__ == "__main__":
    data_dir = sys.argv[1]
    cache_dir = sys.argv[2]
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    splits = sys.argv[4].split(",") if len(sys.argv) > 4 else ["train", "test"]
    lex = os.path.join(ROOT, "artifacts", "translit_lexicon.json")
    for split in splits:
        t = time.time()
        s1, s23 = load_normalized(data_dir, split, cache_dir, lex, n_jobs=n_jobs)
        print(split, "done", len(s1), len(s23), round(time.time() - t, 1), flush=True)
