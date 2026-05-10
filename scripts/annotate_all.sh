#!/bin/bash
# Run Trankit annotation for ccKres, GOS and Solar sequentially.
# Adjust CHUNK_SIZE to tune throughput (larger = faster, more RAM).

set -e
cd "$(dirname "$0")/.."

CHUNK_SIZE=500

echo "=== Solar ==="
python scripts/annotate.py \
    --input      data/datasets/solar/Solar.CoNLL-U/solar-orig.conllu \
    --output     data/datasets/solar/solar_annotated.conllu \
    --save_dir   data/models/save_dir_ssj2.14+sst2.15-stan+pog \
    --embedding  xlm-roberta-large \
    --chunk_size $CHUNK_SIZE

echo "=== GOS ==="
python scripts/annotate.py \
    --input      data/datasets/GOS/gos.conllu \
    --output     data/datasets/GOS/gos_annotated.conllu \
    --save_dir   data/models/save_dir_ssj2.14+sst2.15-stan+pog \
    --embedding  xlm-roberta-large \
    --chunk_size $CHUNK_SIZE

echo "=== ccKres ==="
python scripts/annotate.py \
    --input      data/datasets/ccKres/cckres.conllu \
    --output     data/datasets/ccKres/cckres_annotated.conllu \
    --save_dir   data/models/save_dir_ssj2.14+sst2.15-stan+pog \
    --embedding  xlm-roberta-large \
    --chunk_size $CHUNK_SIZE

echo "=== All done ==="
