#!/usr/bin/env bash
# Assemble the final submission zip:
#   scripts/package.sh <output_dir with matching_results.tsv + candidate_pairs.tsv> <zip path> [team_name]
# The zip follows the layout required by the challenge README:
#   output/{matching_results.tsv,candidate_pairs.tsv}
#   code/business_entity_resolution/{src,README.md,requirements.txt,artifacts,scripts}
#   Documentation_template.md
set -euo pipefail
OUT_DIR="$1"; ZIP="$2"; TEAM="${3:-team}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# stage on disk next to the zip (not in /tmp, which may be RAM-backed and small)
STAGE_BASE="${ER_STAGE_DIR:-$(cd "$(dirname "$ZIP")" && pwd)/.package_stage}"
rm -rf "$STAGE_BASE"; mkdir -p "$STAGE_BASE"
STAGE="$STAGE_BASE/${TEAM}_submission"
mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution"
cp "$OUT_DIR/matching_results.tsv" "$OUT_DIR/candidate_pairs.tsv" "$STAGE/output/"
for item in src README.md requirements.txt artifacts scripts; do
  cp -r "$ROOT/$item" "$STAGE/code/business_entity_resolution/"
done
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} +
cp "$ROOT/Documentation_template.md" "$STAGE/"
rm -f "$ZIP"
(cd "$(dirname "$STAGE")" && zip -qr "$ZIP" "$(basename "$STAGE")")
unzip -l "$ZIP" | tail -n +4 | head -60
rm -rf "$STAGE_BASE"
echo "wrote $ZIP"
