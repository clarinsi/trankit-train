
#!/bin/bash
set -x

SLURM_PARTITION=gpu

LC_LANG=C.UTF-8
LC_ALL=C.UTF-8

mkdir -p ./log

PATCH=$( \
    sbatch \
        -p $SLURM_PARTITION \
	--parsable \
	trankit_train.sh \
)
