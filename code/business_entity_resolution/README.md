# Business Entity Resolution — pipeline

End-to-end, reproducible pipeline that links every Source 1 business entity to its
Source 2 / Source 3 records (`output/matching_results.tsv`) and writes the exact
candidate set that the matcher scored (`output/candidate_pairs.tsv`).

The pipeline is **blocking + learned cascade**:

```text
raw TSV --> normalise (transliterate, fold, legal-form canon, alias split, web-domain
            segmentation, address parsing)                                 [src/normalize.py]
        --> per-country candidate generation: IDF-weighted word nets (name, address,
            blended, reverse) + exact-key groups (same name / address / street)
                                                                            [src/blocking.py]
        --> stage-0 LightGBM filter on blocking scores -> final candidate set
            (this is what candidate_pairs.tsv contains)                     [src/pipeline.py]
        --> ~110 pairwise features (RapidFuzz string similarities incl. digit-for-letter
            fixes, IDF token overlaps and asymmetric extra-token counts, address components,
            house-number edit distance, reference-side ambiguity, competition among candidates)
                                                                            [src/features.py]
        --> stage-1 LightGBM matcher -> stage-2 LightGBM (adds per-entity / per-record
            competition of stage-1 probabilities and *sibling agreement*: do other candidate
            records of the same entity repeat this record's address, house number or spelling,
            and does another entity claim the same value)                   [src/pipeline.py]
        --> decoding: one owner per S2/S3 record + per-entity expected-F0.5 optimal
            prefix (handles singletons)                                     [src/decode.py]
```

Nothing in the pipeline depends on the country label itself: the country is only used
to partition the data, so unseen countries in the test set (e.g. France) go through
exactly the same code path (open-set country handling).

## Requirements

* Linux/macOS, Python 3.11+ (developed with CPython 3.13), 16 GB RAM (every stage
  runs per country *shard* of <= 2.2M Source 2/3 records and peaks at ~12 GB; the shard
  size is adjustable with `--shard-rows`), 8+ cores.
* `pip install -r requirements.txt`

No external data, APIs, look-ups or pre-trained models are used. The only artefact
shipped with the code is `artifacts/translit_lexicon.json`, a Devanagari/Indic ->
Latin word lexicon mined from the **training** data by `scripts/build_lexicon.py`
(it can be regenerated with that script; unknown scripts fall back to Unidecode).

## Data layout

```text
<data_dir>/
  train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
  test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

## Reproduce end-to-end

All commands are run from this directory (`code/business_entity_resolution/`).
`PYTHONPATH=src` is required. Intermediate files go to the `--cache` / `--work`
directories (a few GB); they are safe to delete afterwards.

```bash
export PYTHONPATH=src
DATA=/path/to/dataset          # folder containing train/ and test/
CACHE=./cache; WORK=./work; MODELS=./models

# 0) (optional, already shipped) rebuild the transliteration lexicon from the training data
python scripts/build_lexicon.py $DATA artifacts/translit_lexicon.json

# 1) train: normalise -> block -> stage-0 filter -> features -> stage-1/2 models
#    (prints an entity-held-out validation report with macro-F0.5)
python src/train.py all --data $DATA --cache $CACHE --work $WORK --models $MODELS \
       --lexicon artifacts/translit_lexicon.json

# 2) refit the final models on all training folds (same hyper-parameters, same features)
python src/train.py fit --data $DATA --cache $CACHE --work $WORK --models $MODELS \
       --lexicon artifacts/translit_lexicon.json --final

# 3) inference on the test split -> output/matching_results.tsv + output/candidate_pairs.tsv
#    (per-shard intermediate files go to --work; default <out>/work)
python src/predict.py --data $DATA --split test --cache $CACHE --models $MODELS \
       --lexicon artifacts/translit_lexicon.json --out output --work $WORK/test

# 4) validate (organiser script)
python utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir $DATA/test

# 5) (optional) in-sample score of the final models on the training split
python src/predict.py --data $DATA --split train --cache $CACHE --models $MODELS \
       --lexicon artifacts/translit_lexicon.json --out output_train --work $WORK

# 6) assemble the submission zip (output/ + code/ + methodology document)
scripts/package.sh output ../<team>_submission.zip <team>
```

Approximate wall-clock on a 16-core / 32 GB machine (8 threads, 6 worker processes):
normalisation 5 min per split, blocking 25 min (India) + 35 min (US), stage-0 filter
fit + scoring 70 min, features 30 min, stage-1/2 fitting 50 min per fit (validation fit
and final fit), test inference ~2.5 h. Peak RSS of any step is below 15 GB.

`scripts/run.sh <name> <python args>` runs any of the above in the background under a
memory watchdog and writes `logs/<name>.log`; `scripts/package.sh` builds the submission zip.

Every step is resumable: each (country, shard) writes its own file into `--work`
(`<split>_cand_*`, `<split>_p0_*`, `<split>_keep_*`, `<split>_feat_*`) and existing files
are skipped on re-runs.

### Candidate density (15-30 per Source 1 entity)

The stage-0 operating point (`PruneConfig` in `src/pipeline.py`: `p0_min=0.0002`,
`top_b=3`, `top_a=3`, `max_per_a=60`) was chosen with `train.py stage0` (which prints a
density / positive-recall sweep) so that the training split gets ~18.6 candidates per
entity (India 20.8, US 17.1) while keeping 99.99% of the true pairs that blocking found.
At inference the same rule is applied per country; if a country's density leaves the
safety band [16, 28] (possible for a country unseen in training), `Runner._density_guard`
re-tunes only the probability floor `p0_min` by bisection towards 20 per entity, so the
shipped `candidate_pairs.tsv` always satisfies the 15-30 requirement.

## Source files

| file | purpose |
| --- | --- |
| `src/data.py` | TSV reading, chunked/multiprocess normalisation cache, per-country partition loading |
| `src/normalize.py` | name / address normalisation, legal-form canonicalisation, alias split, address components |
| `src/segment.py` | word segmentation of concatenated web-domain / handle style names (Viterbi over unigram log-probabilities estimated from Source 1) |
| `src/lexicon.py`, `scripts/build_lexicon.py` | mining the Indic -> Latin transliteration lexicon from matched training pairs |
| `src/blocking.py` | candidate generation (token spaces, sparse top-k nets, exact-key groups, union) |
| `src/pipeline.py` | stage-0 filter features, pruning, LightGBM parameters, model bundle |
| `src/features.py` | pairwise feature engineering |
| `src/decode.py` | owner constraint + expected-F0.5 decoding, TSV writers |
| `src/evaluate.py` | macro-F0.5 metric (identical definition to the challenge) |
| `src/runner.py` | sharded, memory-bounded driver shared by training and inference (blocking -> stage-0 scoring -> global per-country pruning -> features; one file per country shard) |
| `src/train.py` | training / validation driver (`block`, `stage0`, `feats`, `fit`, `all`) |
| `src/predict.py` | inference driver producing the two output files |

## Determinism

Folds are hash-based (`crc32(entity_id) % 5`), all random subsampling uses fixed seeds,
LightGBM is run with fixed seeds and a fixed thread count; re-running the commands above
reproduces the shipped outputs up to floating-point noise in the boosters.
