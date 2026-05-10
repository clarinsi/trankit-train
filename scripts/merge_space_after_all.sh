#!/bin/bash
# Transfer SpaceAfter=No from original CoNLL-U into Trankit-annotated output
# for ccKres, GOS and Solar. Writes to separate *_spaceafter.conllu files —
# never overwrites the annotation output.

set -e
cd "$(dirname "$0")/.."

echo "=== ccKres ==="
python scripts/merge_space_after.py \
    --original  data/datasets/ccKres/cckres.conllu \
    --annotated data/datasets/ccKres/cckres_annotated.conllu \
    --output    data/datasets/ccKres/cckres_annotated_spaceafter.conllu

echo "=== GOS ==="
python scripts/merge_space_after.py \
    --original  data/datasets/GOS/gos.conllu \
    --annotated data/datasets/GOS/gos_annotated.conllu \
    --output    data/datasets/GOS/gos_annotated_spaceafter.conllu

echo "=== Solar ==="
python scripts/merge_space_after.py \
    --original  data/datasets/solar/Solar.CoNLL-U/solar-orig.conllu \
    --annotated data/datasets/solar/solar_annotated.conllu \
    --output    data/datasets/solar/solar_annotated_spaceafter.conllu

echo "=== All done ==="
