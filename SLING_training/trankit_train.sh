#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --error=log/%J.trankit_train.err
#SBATCH --output=log/%J.trankit_train.out
#SBATCH --job-name=trankit_train

echo "$SLURM_JOB_ID --> Preparing trankit training."

LANG=C.UTF-8
LC_ALL=C.UTF-8

srun \
  trankit_content.sh

