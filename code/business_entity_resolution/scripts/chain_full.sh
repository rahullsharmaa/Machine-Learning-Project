#!/usr/bin/env bash
# End-to-end chain from the raw TSVs (resumable: every step skips work whose outputs already exist).
#   normalize -> block -> stage0 (fit or reuse models/stage0.txt) + prune -> feats -> fit (validation)
#   -> fit --final -> predict test -> validator -> copy deliverables to /mnt/documents
# Usage: chain_full.sh [from-step]     from-step in {normalize,block,stage0,feats,fit,final,predict,validate}
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S="$ROOT/src"
ER="${ER_HOME:-/root/er}"
export PYTHONPATH="${ER_PYLIB:-/root/pylib}:$S"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export ARROW_DEFAULT_MEMORY_POOL=system
PCFG="${ER_PCFG:-p0_min=0.0002,top_b=3,top_a=3,max_per_a=60}"
COMMON=(--data "$ER/dataset" --cache "$ER/cache" --models "$ER/models" --lexicon "$ROOT/artifacts/translit_lexicon.json" --pcfg "$PCFG")
FROM="${1:-normalize}"
log() { echo "$(date +%T) CHAIN $*"; }
step_ge() {  # true when step $1 is at or after $FROM in the chain order
  local order="normalize block stage0 feats fit final predict validate" i=0 a=0 b=0
  for s in $order; do i=$((i+1)); [ "$s" = "$1" ] && a=$i; [ "$s" = "$FROM" ] && b=$i; done
  [ "$a" -ge "$b" ]
}
mkdir -p "$ER/models" /mnt/documents/er_models_valid /mnt/documents/er_models_final /mnt/documents/er_output

if step_ge normalize; then
  log "normalize"; python3 "$ROOT/scripts/run_normalize.py" "$ER/dataset" "$ER/cache" 6 train,test || exit 1
fi
if step_ge block; then
  log "block";  python3 "$S/train.py" block  "${COMMON[@]}" --work "$ER/work" || exit 1
fi
if step_ge stage0; then
  # reuse a previously fitted stage-0 filter when available (identical blocking features)
  if [ ! -f "$ER/models/stage0.txt" ] && [ -f /mnt/documents/er_models_valid/stage0.txt ]; then
    cp /mnt/documents/er_models_valid/stage0.txt /mnt/documents/er_models_valid/stage0_cols.json "$ER/models/"
  fi
  log "stage0"; python3 "$S/train.py" stage0 "${COMMON[@]}" --work "$ER/work" || exit 1
fi
if step_ge feats; then
  log "feats";  python3 "$S/train.py" feats  "${COMMON[@]}" --work "$ER/work" || exit 1
  bash "$ROOT/scripts/backup.sh" > /dev/null
fi
if step_ge fit; then
  log "fit (validation)"; python3 "$S/train.py" fit "${COMMON[@]}" --work "$ER/work" || exit 1
  cp "$ER"/models/* /mnt/documents/er_models_valid/ && cp "$ER/work/valid_preds.npz" /mnt/documents/er_models_valid/
fi
if step_ge final; then
  log "fit --final"; python3 "$S/train.py" fit "${COMMON[@]}" --work "$ER/work" --final || exit 1
  cp "$ER"/models/* /mnt/documents/er_models_final/
fi
if step_ge predict; then
  log "predict test"; python3 "$S/predict.py" --data "$ER/dataset" --cache "$ER/cache" --models "$ER/models" \
      --lexicon "$ROOT/artifacts/translit_lexicon.json" --split test --out "$ER/output" --work "$ER/work_test" --save-scores || exit 1
fi
if step_ge validate; then
  log "validate"; python3 "$ER/utils/validate_submission.py" -m "$ER/output/matching_results.tsv" -c "$ER/output/candidate_pairs.tsv" -t "$ER/dataset/test" --check-ids
  cp "$ER"/output/*.tsv /mnt/documents/er_output/
fi
log "done"
