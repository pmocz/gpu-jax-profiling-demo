#!/usr/bin/bash
#SBATCH --job-name=gpu_profiling
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00-00:05

module purge
module load python/3.11.11
module load uv

export PYTHONUNBUFFERED=TRUE

# source .venv/bin/activate
srun uv run python main.py
