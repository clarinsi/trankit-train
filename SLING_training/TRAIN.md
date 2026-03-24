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
├── trankit-train/              # cloned repo
│   └── data/
│       ├── datasets/           # training data (unzipped here)
│       └── models/             # trained model output (save_dir_*)
├── trankit_contents/           # named training configs (copied from repo)
├── trankit_content.sh          # active config (set by run_training.sh)
├── run_training.sh             # submit a named config as a SLURM job
├── trankit_sbatch.sh           # SLURM submission script
├── trankit_train.sh            # job execution wrapper
├── containers/
│   └── trankit.sif             # Singularity container
└── log/                        # job output/error logs
```

## Set up on SLING-hpc

```bash
cd ~/trankit/

# Clone repo
git clone <repo_url> trankit-train

# Copy SLURM scripts to working directory
cp trankit-train/SLING_training/trankit_sbatch.sh .
cp trankit-train/SLING_training/trankit_train.sh .
cp trankit-train/SLING_training/run_training.sh .
cp -r trankit-train/SLING_training/trankit_contents .

# Build Singularity container (bootstraps from Docker, no sudo needed)
mkdir -p containers
singularity build --fakeroot containers/trankit.sif trankit-train/SLING_training/trankit.def

# Transfer and unzip training data via scp, then:
cd trankit-train/data/
unzip trankit-data3.zip    # extracts datasets/ssj2.14+sst2.15-pog/ and datasets/ssj2.14+sst2.15-stan+pog/
```

## Training configurations

| Config name | Script | Dataset | Embedding | Save dir |
|---|---|---|---|---|
| `pog_base` | `trankit_content_pog_base.sh` | `ssj2.14+sst2.15-pog` | xlm-roberta-base | `save_dir_ssj2.14+sst2.15-pog` |
| `pog_large` | `trankit_content_pog_large.sh` | `ssj2.14+sst2.15-pog` | xlm-roberta-large | `save_dir_ssj2.14+sst2.15-pog` |
| `stanpog_base` | `trankit_content_stanpog_base.sh` | `ssj2.14+sst2.15-stan+pog` | xlm-roberta-base | `save_dir_ssj2.14+sst2.15-stan+pog` |
| `stanpog_large` | `trankit_content_stanpog_large.sh` | `ssj2.14+sst2.15-stan+pog` | xlm-roberta-large | `save_dir_ssj2.14+sst2.15-stan+pog` |

Base and large variants of the same dataset share one save dir — Trankit writes each under its own `xlm-roberta-base/` or `xlm-roberta-large/` subdirectory inside it.

## Run a training

```bash
cd ~/trankit/

# List available configs
./run_training.sh

# Submit one of the 4 new training jobs
./run_training.sh pog_base        # SSJ+SST-pog,      xlm-roberta-base
./run_training.sh pog_large       # SSJ+SST-pog,      xlm-roberta-large
./run_training.sh stanpog_base    # SSJ+SST-stan+pog, xlm-roberta-base
./run_training.sh stanpog_large   # SSJ+SST-stan+pog, xlm-roberta-large
```

`run_training.sh` copies the named config to `trankit_content.sh` and calls `./trankit_sbatch.sh`.

> **Note:** If you get a CUDA out of memory error, lower `batch_size` for the posdep task
> in `train.py` (currently hardcoded to 12).

## Track progress

```bash
# Check job status
squeue -u $USER

# Follow logs (both .out and .err are useful)
tail -f log/<job_id>.trankit_train.out
tail -f log/<job_id>.trankit_train.err
```

## Re-running a failed job

Remove the incomplete save directory before resubmitting, otherwise Trankit may skip training:

```bash
rm -rf trankit-train/data/models/save_dir_<name>/
./run_training.sh <config_name>
```

## Packaging trained models for download

```bash
cd trankit-train/data/

# Remove cached embeddings (not part of the model, re-downloaded on inference)
rm -rf models/save_dir_<name>/xlm-roberta-large/customized/xlm-roberta-large/
rm -rf models/save_dir_<name>/xlm-roberta-base/customized/xlm-roberta-base/

# Zip
zip -r models_pog.zip models/save_dir_ssj2.14+sst2.15-pog models/save_dir_ssj2.14+sst2.15-stan+pog
```

## Adding a new training config

1. Copy an existing script from `SLING_training/trankit_contents/` and adjust paths/embedding.
2. On SLING: `cp trankit-train/SLING_training/trankit_contents/<new>.sh trankit_contents/`
3. Run: `./run_training.sh <new_name>`

## Container manipulation (singularity)

```bash
singularity exec containers/python-nvidia.sif bash
singularity cache clean
singularity exec pytorch.sif python -V
singularity build -f pytorch.sif docker://nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
```
