#!/bin/bash
# Submit a named training config as a SLURM job.
#
# Usage:
#   ./run_training.sh <config_name>
#
# Available configs (pass the name without path or extension):
#   pog_base        SSJ2.14+SST2.15-pog,      xlm-roberta-base
#   pog_large       SSJ2.14+SST2.15-pog,      xlm-roberta-large
#   stanpog_base    SSJ2.14+SST2.15-stan+pog, xlm-roberta-base
#   stanpog_large   SSJ2.14+SST2.15-stan+pog, xlm-roberta-large

CONFIG=$1

if [ -z "$CONFIG" ]; then
    echo "Usage: ./run_training.sh <config_name>"
    echo ""
    echo "Available configs:"
    for f in trankit_contents/trankit_content_*.sh; do
        basename "$f" | sed 's/trankit_content_//' | sed 's/\.sh//'
    done
    exit 1
fi

CONFIG_FILE="trankit_contents/trankit_content_${CONFIG}.sh"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config not found: $CONFIG_FILE"
    exit 1
fi

cp "$CONFIG_FILE" trankit_content.sh
echo "Activated config: $CONFIG"
./trankit_sbatch.sh
