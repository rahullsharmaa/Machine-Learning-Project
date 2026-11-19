"""Mine the Indic->Latin transliteration lexicon from the training ground truth."""
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data import read_ground_truth, read_tsv  # noqa: E402
from lexicon import mine_lexicon, save_lexicon  # noqa: E402
from normalize import _INDIC_RE  # noqa: E402

if __name__ == "__main__":
    data_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "artifacts", "translit_lexicon.json")
    t = time.time()
    s1 = read_tsv(os.path.join(data_dir, "train", "train_source1.tsv"), usecols=["entity_id", "business_name"])
    gt = read_ground_truth(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    name1 = dict(zip(s1["entity_id"], s1["business_name"]))
    del s1
    owner = {}
    for e, m in zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].values):
        for x in m.split(","):
            if x:
                owner[x] = e
    pairs = []
    for i in (2, 3):
        d = read_tsv(os.path.join(data_dir, "train", f"train_source{i}.tsv"), usecols=["entity_id", "business_name"])
        for eid, nm in zip(d["entity_id"].values, d["business_name"].values):
            if _INDIC_RE.search(nm):
                o = owner.get(eid)
                if o is not None:
                    pairs.append((name1[o], nm))
        del d
    lex = mine_lexicon(pairs)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_lexicon(lex, out)
    print("pairs", len(pairs), "lexicon", len(lex), round(time.time() - t, 1))
