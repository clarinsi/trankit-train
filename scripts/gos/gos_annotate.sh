#!/bin/bash
# Annotate GOS CoNLL-U with Trankit (step 2).
# Run step 1 first: python scripts/gos/gos_tei_to_conllu.py

set -e
cd "$(dirname "$0")/../.."

INPUT=data/datasets/GOS/gos.conllu

if [ ! -f "$INPUT" ]; then
    echo "ERROR: $INPUT not found. Run step 1 first:"
    echo "  python scripts/gos/gos_tei_to_conllu.py"
    exit 1
fi

python scripts/annotate.py \
    --input     data/datasets/GOS/gos.conllu \
    --output    data/datasets/GOS/gos_annotated.conllu \
    --save_dir  data/models/save_dir_ssj2.14+sst2.15-stan+pog \
    --embedding xlm-roberta-large \
    --chunk_size 1000 \
    2>&1 | tee scripts/gos/gos_annotate.out
