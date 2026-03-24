
#!/bin/bash
set -x

SLURM_PARTITION=gpu

LC_LANG=C.UTF-8
LC_ALL=C.UTF-8

mkdir -p ./log

CONFIG_FILE=$1

PATCH=$( \
    sbatch \
        -p $SLURM_PARTITION \
	--parsable \
	--export=ALL,TRANKIT_CONFIG="$CONFIG_FILE" \
	trankit_train.sh \
)
