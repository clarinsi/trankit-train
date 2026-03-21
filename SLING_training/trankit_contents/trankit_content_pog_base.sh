#!/bin/bash

echo 'Training: SSJ2.14+SST2.15-pog [xlm-roberta-base]'

singularity exec --nv containers/trankit.sif python trankit-train/train.py \
    --tokenize --lemmatize --posdep \
    --category customized \
    --save_dir trankit-train/data/models/save_dir_ssj2.14+sst2.15-pog \
    --embedding xlm-roberta-base \
    --train_conllu_fpath trankit-train/data/datasets/ssj2.14+sst2.15-pog/sl_ssj+sst-ud-train-formatted.conllu \
    --dev_conllu_fpath  trankit-train/data/datasets/ssj2.14+sst2.15-pog/sl_ssj+sst-ud-dev-formatted.conllu \
    --train_txt_fpath   trankit-train/data/datasets/ssj2.14+sst2.15-pog/train.txt \
    --dev_txt_fpath     trankit-train/data/datasets/ssj2.14+sst2.15-pog/dev.txt

echo 'Done training!'
