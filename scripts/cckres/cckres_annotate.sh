#!/bin/bash
# Annotate ccKres CoNLL-U with Trankit (step 2).
# Processes in chunks of 1000 sentences with progress logging.
# Run step 1 first: python scripts/cckres_tei_to_conllu.py

set -e
cd "$(dirname "$0")/../.."

INPUT=data/datasets/ccKres/cckres.conllu

if [ ! -f "$INPUT" ]; then
    echo "ERROR: $INPUT not found. Run step 1 first:"
    echo "  python scripts/cckres_tei_to_conllu.py"
    exit 1
fi

python scripts/annotate.py \
    --input     data/datasets/ccKres/cckres.conllu \
    --output    data/datasets/ccKres/cckres_annotated.conllu \
    --save_dir  data/models/save_dir_ssj2.14+sst2.15-stan+pog \
    --embedding xlm-roberta-large \
    --chunk_size 1000 \
    2>&1 | tee scripts/cckres/cckres_annotate.out
