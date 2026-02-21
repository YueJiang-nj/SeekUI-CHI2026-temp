#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=create_data.out

module load mamba
module load triton/2025.1-gcc
module load cuda

source activate /scratch/work/guoz3/environment/qwen

export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/work/guoz3/environment/qwen
export TORCH_HOME=/scratch/work/guoz3/environment/qwen

python create_think_data.py

source deactivate
