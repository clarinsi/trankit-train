# Train instructions on SLING

## Connect to SLING

Add to `~/.ssh/config`:
```
Host SLING-hpc
    Hostname hpc-login.arnes.si
    User lkrsnik
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```

Then connect:
```bash
ssh SLING-hpc
```

## Directory layout on the cluster

The scripts expect the following structure in your home directory:
```
~/trankit/
├── trankit-train/          # cloned repo
│   └── data/               # training data (unzipped here)
├── trankit_contents/       # saved copies of training configs
├── trankit_content.sh      # active training config
├── trankit_sbatch.sh       # SLURM submission script
├── trankit_train.sh        # job execution wrapper
├── containers/
│   └── trankit.sif         # Singularity container
└── log/                    # job output/error logs
```

## Set up on SLING-hpc

```bash
# Clone repo
cd ~/trankit/
git clone <repo_url> trankit-train

# Copy SLURM scripts to working directory
cp trankit-train/SLING_training/trankit_sbatch.sh .
cp trankit-train/SLING_training/trankit_train.sh .
cp trankit-train/SLING_training/trankit_content.sh .

# Build Singularity container on the cluster (bootstraps from Docker, no sudo needed)
mkdir -p containers
singularity build containers/trankit.sif trankit-train/SLING_training/trankit.def

# Transfer and unzip training data
# (upload trankit-data.zip via scp first, then:)
cd trankit-train/data/
unzip trankit-data.zip
```

## Managing multiple training configurations

Store each configuration as a named copy in `trankit_contents/`, then copy the desired one to the active `trankit_content.sh` before submitting:

```bash
# Save current config
cp trankit_content.sh trankit_contents/trankit_content_ssj2.14+sst2.15_large.sh

# Switch to a different config
cp trankit_contents/trankit_content_sst2.15_base.sh trankit_content.sh
vim trankit_content.sh   # review/adjust paths
./trankit_sbatch.sh
```

## Run training
```bash
vim trankit_content.sh   # set dataset paths, save_dir, embedding

# NOTE: If you get CUDA out of memory error, lower batch size in training_config
# (i.e. `'batch_size': 12` for posdep in train.py)

./trankit_sbatch.sh
```

## Track progress
```bash
# Check job status
squeue -u $USER

# Follow logs (both .out and .err are useful)
tail -f log/<job_id>.trankit_train.out
tail -f log/<job_id>.trankit_train.err

# Read full error log
vim log/<job_id>.trankit_train.err
```

## Re-running a failed job

Remove the incomplete save directory before resubmitting, otherwise Trankit may skip training:

```bash
rm -rf trankit-train/data/save_dir_<name>/
vim trankit_content.sh
./trankit_sbatch.sh
```

## Packaging trained models for download

Before zipping, remove the cached embedding files that Trankit downloads during training — they are not part of the model and can be re-downloaded:

```bash
cd trankit-train/data/

# Remove cached embeddings (adjust embedding name as needed)
rm -rf save_dir_<name>/xlm-roberta-large/customized/xlm-roberta-large/
rm -rf save_dir_<name>/xlm-roberta-base/customized/xlm-roberta-base/

# Zip one or more model directories
zip -r models.zip save_dir_<name1>/* save_dir_<name2>/*
```

## Container manipulation (singularity)
```bash
singularity exec containers/python-nvidia.sif bash
singularity cache clean
singularity exec pytorch.sif python -V
singularity build -f pytorch.sif docker://nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
```
