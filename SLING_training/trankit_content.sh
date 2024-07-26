#!/bin/bash

echo 'Training!'

singularity exec --nv containers/trankit.sif python trankit-train/train.py --tokenize --lemmatize --posdep --category customized --save_dir trankit-train/data/save_dir_ssj2.14 --dev_conllu_fpath trankit-train/data/ssj2.14/sl_ssj-ud-dev-formatted.conllu --train_conllu_fpath trankit-train/data/ssj2.14/sl_ssj-ud-train-formatted.conllu --train_txt_fpath trankit-train/data/ssj2.14/train.txt --dev_txt_fpath trankit-train/data/ssj2.14/dev.txt --embedding xlm-roberta-base

echo 'Done training!'
