#!/bin/bash
# Annotate Solar CoNLL-U with Trankit, preserving original tokenisation.
# Input:  data/datasets/solar/Solar.CoNLL-U/solar-orig.conllu
# Output: data/datasets/solar/solar_annotated.conllu

set -e
cd "$(dirname "$0")/../.."

python scripts/annotate.py \
    --input     data/datasets/solar/Solar.CoNLL-U/solar-orig.conllu \
    --output    data/datasets/solar/solar_annotated.conllu \
    --save_dir  data/models/save_dir_ssj2.14+sst2.15-stan+pog \
    --embedding xlm-roberta-large \
    --chunk_size 1000 \
    2>&1 | tee scripts/solar/solar_annotate.out
