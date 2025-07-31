#!/bin/bash
# This SLURM batch script is configured for running the PFA learner with maximal GPU usage on Great Lake.

# Job name:
#SBATCH --job-name=ssm

# Email notifications:
#SBATCH --mail-user=huteng@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Resources:
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # One task (non-distributed training)
#SBATCH --ntasks-per-node=1         # can be used to parallelize across multiple GPUs
#SBATCH --cpus-per-task=8          # same value as GPU
#SBATCH --mem-per-cpu=36G           # max = 36G; twice the GPU memory # Max CPU RAM = 360 GB < Max GPU node RAM = 372 GB
#SBATCH --time=96:00:00             # Request a walltime of 96 hours
#SBATCH --account=huteng0           # or ling702w25_class Replace with your account if different

# GPU resources:
#SBATCH --partition=spgpu           # Specify the GPU partition (or spgpu "single-precsion")
#SBATCH --gres=gpu:8                # Request 8 GPUs (48GB each)

# Log file location (uses username (%u), job name (%x) and job ID (%j)):
#SBATCH --output=/home/%u/ssm/output/slurm_logs/%x-%j.log
#SBATCH --error=/home/%u/ssm/output/slurm_logs/%x-%j.err

# --------------------------
# Environment Setup
# --------------------------
# Load necessary modules. Adjust the python version if needed.
module load python
module load cuda/11.2


# Create a fresh virtual environment
source ~/.bashrc
conda activate pytorch_tutorial
# pip install -r requirements.txt

# --------------------------
# Run the Application
# --------------------------
/sw/pkgs/arc/python/3.12.1/bin/python mlregtest.py

# (Optional) After successful run, you might push changes to git:
# git push
