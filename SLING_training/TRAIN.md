# Train instructions on SLING

## Create singularity container
```bash
# Creates container using definition file
sudo singularity build trankit.sif trankit.def
```

## Connect to SLING
Get access to SLING. I suggest SLING-hpc node.

## Transfer necessary data to access node
You should clone this repo up there, transfer `trankit.sif`, adapt and if necessary move `trankit_content.sh`, `trankit_sbatch.sh` and `trankit_train.sh`. 

## Run training
```bash
# Adapt the file if necessary
vim trankit_content.sh

# NOTE: If you get CUDA out of memory error lower batch size in training_config 
# (ie. `'batch_size': 12` for posdep in train.py)

./trankit_sbatch.sh
```
## Track progress
```bash
# see the execution status
squeue -u $USER
sinfo

# see task logs
tail -f log/45704901.trankit_train.out
```

## Container manipulation (singularity) 
Probably not necessary, but gere are some useful commands:
```bash
singularity exec containers/python-nvidia.sif bash
singularity cache clean
singularity exec pytorch.sif python -V
singularity build -f pytorch.sif docker://nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
```