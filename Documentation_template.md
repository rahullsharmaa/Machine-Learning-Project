# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** noname
**Team Members:** 
Rahul Sharma
Joydip Sen
Krishanu Mondal
Paramjit Jena


**Submission Date:** 02/10/2026

---

## 1. Executive Summary

We resolve every Source 1 business against Sources 2/3 with a *blocking + learned cascade*.
Country-agnostic normalisation (script transliteration, legal-form canonicalisation, alias
and web-domain segmentation, address component parsing) feeds an IDF-weighted sparse
retrieval layer that unions seven complementary "nets" (name / address / blended cosine
top-k in both directions plus exact same-name, same-address and same-street groups),
recalling {{BLOCK_RECALL}} of the true pairs at ~170 raw pairs per entity. A light
LightGBM *filter* on retrieval-only signals trims this to {{DENSITY}} candidates per
entity (the shipped `candidate_pairs.tsv`) while keeping {{FILTER_RECALL}} of the true
pairs; a 97-feature LightGBM *matcher* followed by a second LightGBM stage that sees the
competition among candidates (per entity and per record) produces calibrated match
probabilities, and a per-entity **expected-F0.5 decoder** with a one-owner-per-record
constraint turns them into match lists — including deliberately empty lists for
singletons. Entity-held-out validation macro-F0.5: **{{VAL_F05}}**.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the training split (2.2M Source 1 entities, 10.3M Source 2/3 records,
3.46 true matches per entity on average, 5.6% singletons):

* **Names are heavily perturbed but rarely replaced.** The dominant noise types are
  dropped / added legal forms ("Pvt Ltd", "LLC", "Inc"), reordered words, spacing and
  punctuation changes, character typos, abbreviations, and — for India — Devanagari,
  Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam and Gurmukhi *script* versions of the
  Latin name. ~7% of true matches have an *unrelated* name (rebranding / holding-company
  style records) and are only recoverable through the address.
* **Addresses are the second key and are often missing or reduced.** ~40% of Source 2/3
  records carry an address that is a subset (street only, city only, PIN/ZIP only) of the
  Source 1 address; a large share carry no address at all. Word order is unstable
  ("Sector 12, Noida" vs "Noida Sector-12") and administrative components are optional.
* **Ambiguity is structural, not incidental.** Many Source 1 entities share the exact same
  normalised name ("Sharma General Store", "Subway"); such records can only be resolved
  via the address, and when the record has no address the pair is genuinely undecidable.
  Correct handling of this ambiguity (abstaining rather than guessing) matters because
  the metric is precision-heavy and singletons earn full credit for an empty list.
* **Country is only a partition key.** All matches are within-country; nothing else about
  a country is stable enough to hard-code, and the test set contains an unseen country
  (France). Every component is therefore learned or estimated *per partition at run time*
  (IDF weights, exact-key groups), never from a country-specific rule set.
* **Web-domain / handle style names** ("sharmageneralstore.in", "@sharma_store") appear
  as Source 2/3 variants and need word segmentation before any token similarity works.

### 2.2 Solution Strategy

**Approach Type:** Blocking + multi-stage classifier with a metric-aware decoder  
**Core Innovation:** (i) a *union of complementary sparse nets* with per-net rank features,
which makes the recall of blocking measurable and tunable per net; (ii) a learned
stage-0 filter that reaches the required 15–30 candidate density **without** losing
recall, so the matcher only sees hard negatives; (iii) *competition-aware* second-stage
features and an *expected-F0.5-optimal per-entity decoder* that directly optimises the
challenge metric including the singleton case.

```text
TSV  ->  normalise  ->  per-country sharded blocking (7 nets, union)  ->  stage-0 filter
     ->  candidate_pairs.tsv  ->  97 pair features  ->  stage-1 LightGBM
     ->  competition features  ->  stage-2 LightGBM  ->  owner constraint + expected-F0.5
     ->  matching_results.tsv
```

Everything runs per (country, shard of <= 2.2M Source 2/3 records) so peak memory stays
below ~12 GB on the full data.

---

## 3. Candidate Generation (Blocking)

* **Normalised representations** (`src/normalize.py`): Unicode NFKC + transliteration
  (mined Indic->Latin lexicon with Unidecode fall-back), lower-casing, punctuation
  folding, legal-form canonicalisation to a single token (`pvtltd`, `llc`, ...), alias
  extraction ("A (formerly B)"), web-domain / handle segmentation (Viterbi word
  segmentation over Source 1 unigram statistics), address parsing into street / locality /
  city / region / postal code with order-insensitive keys.
* **Blocking keys / nets** (`src/blocking.py`), all computed within one country:
  1. *name net*: TF-IDF cosine over name words + word bigrams + whole-name token +
     Soundex codes; top-20 Source 1 entities per record (`sparse_dot_topn`);
  2. *address net*: TF-IDF cosine over address tokens + sorted-token keys; top-20;
  3. *blended net*: concatenated name+address space; top-30;
  4. *reverse net*: top-6 records per Source 1 entity in the blended space (so entities
     with rare names are never starved);
  5. *same-name groups*: exact normalised-name equality (up to 12 members each, best by
     address cosine, hard cap 400);
  6. *same-address groups*: exact order-insensitive address equality;
  7. *same-street groups*: exact street key (up to 5, best by name cosine).
  Every pair keeps its rank in every net (99 = not retrieved) plus the exact name /
  address / blended cosines; the union is de-duplicated and capped at 600 pairs per
  entity (a safety cap that is never binding in practice).
* **Candidate pairs generated:** raw union {{RAW_PAIRS}} pairs on the training split
  (~{{RAW_DENSITY}} per Source 1 entity), reduced by the stage-0 filter to {{FINAL_PAIRS}}
  pairs ({{DENSITY}} per entity); on the test split {{TEST_PAIRS}} pairs
  ({{TEST_DENSITY}} per entity, i.e. inside the required 15–30 window).
* **How we ensured true matches were not lost:** blocking recall is measured on the
  ground truth after every change ({{BLOCK_RECALL}} of all true pairs are in the raw
  union; the remaining misses are dominated by records with *no address and a name shared
  by many entities*, which no retrieval can disambiguate). The stage-0 filter is a
  LightGBM model trained on retrieval-only features (per-net ranks, cosines, gaps to the
  best competitor on both sides, name/address duplication counts, record shape); its
  operating point was chosen on a sensitivity sweep to keep >= 99.5% of the positives
  that blocking found ({{FILTER_RECALL}} on the held-out fold) at the target density.
  Because the pruning uses per-entity / per-record *ranks*, the top candidates of every
  entity and of every record are always retained.

---

## 4. Matching Model

**Features used** (`src/features.py`, 97 per pair, plus 9 competition features):
- Name features: RapidFuzz ratio / partial / token-set / token-sort / WRatio / Jaro-Winkler
  on the canonical name, on the name without spaces, and on the "core" name (legal forms
  stripped); IDF-weighted token overlap and cosine; first-token / last-token equality;
  legal-form set agreement; length and token-count differences; prefix/suffix containment;
  acronym match; transliteration / non-Latin and web-domain flags.
- Address features: full-address, street, locality, city, region and postal-code
  similarities (exact, fuzzy, IDF overlap); house-number agreement; "record address is a
  subset of entity address" indicators; emptiness indicators for both sides.
- Other: reference-side ambiguity (how many Source 1 entities share the name / address /
  street), stage-0 probability and retrieval ranks, per-entity and per-record best /
  second-best / gap statistics of the cosines (stage 1) and of the stage-1 probability
  (stage 2: `p1`, per-record max and gap, per-entity max, gap, sum and confidence).

**Model type:** LightGBM (binary objective, 255 leaves, 0.05 learning rate, 800 / 400
rounds) in two stages; the stage-2 competition features are computed out-of-fold on the
training data (two half-models) so that they are unbiased. MIT-licensed gradient boosting
with < 10^5 parameters — no pre-trained or external model.

**Threshold selection method:** the stage-2 probabilities are decoded with (a) an
owner constraint — a Source 2/3 record may only be assigned to its highest-probability
entity — and (b) a per-entity *expected-F0.5* maximiser: candidates are sorted by
probability and the prefix that maximises the expectation of F0.5 under a Poisson-binomial
model of the true-match count is emitted (the empty prefix has expected value equal to the
probability that the entity is a singleton, so singletons are handled without any extra
rule). Plain thresholds (0.3 … 0.7) with and without the owner constraint were evaluated
as baselines on the same held-out fold; the best strategy is stored in the model bundle
and applied unchanged at inference ({{BEST_STRATEGY}}).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** {{VAL_F05}} on 20% entity-held-out validation (fold 0 by
  `crc32(entity_id) % 5`, {{VAL_N}} Source 1 entities, singletons included, models never
  trained on that fold). Oracle with perfect decisions on the shipped candidate set:
  {{ORACLE}}; blocking recall {{BLOCK_RECALL}}; average precision of the pair classifier
  stage 1 {{AP1}} / stage 2 {{AP2}}. In-sample macro-F0.5 on the full training split with
  the final models: {{TRAIN_F05}}.
- **Common false positives (wrong merges):** {{FP_TEXT}}
- **Common false negatives (missed matches):** {{FN_TEXT}}

---

## 6. Conclusion

A carefully measured retrieval layer (seven complementary nets, recall audited per net),
a learned recall-preserving filter, and a metric-aware decoder together give
{{VAL_F05}} macro-F0.5 on unseen entities while shipping only {{DENSITY}} candidates per
entity. The main lesson: on precision-heavy entity resolution the decoder and the explicit
modelling of *competition* between candidates matter as much as the pairwise classifier,
and every remaining error class is now one where the record simply lacks the information
needed to decide.

---

## Appendix

### A. Code Artefacts

```text
code/business_entity_resolution/
  README.md              exact reproduction steps (train -> final fit -> predict -> validate)
  requirements.txt       pinned versions (numpy, pandas, scipy, scikit-learn, lightgbm,
                         rapidfuzz, sparse_dot_topn, pyarrow, Unidecode)
  artifacts/translit_lexicon.json   Indic->Latin lexicon mined from the training data
  scripts/build_lexicon.py          regenerates the lexicon;  scripts/run.sh  background runner
  src/normalize.py, segment.py, lexicon.py   text normalisation
  src/blocking.py        candidate generation (nets + union)
  src/pipeline.py        stage-0 features, pruning, LightGBM settings, model bundle
  src/features.py        pair features + stage-2 competition features
  src/decode.py          owner constraint, expected-F0.5 decoder, TSV writers
  src/evaluate.py        challenge metric
  src/runner.py          sharded, memory-bounded driver shared by training and inference
  src/train.py           `python src/train.py all ...` then `... fit --final`
  src/predict.py         `python src/predict.py --split test ...` -> output/*.tsv
```

Entry points: `python src/train.py all` (trains and prints the validation report),
`python src/train.py fit --final` (refit on all folds), `python src/predict.py --split test`
(writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`). See README.md.

### B. Additional Results

{{APPENDIX_B}}

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
