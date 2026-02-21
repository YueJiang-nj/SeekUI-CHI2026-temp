#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --constraint=hopper
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --output=Eye-tracking-RFT-setup.out

module load mamba
module load triton/2025.1-gcc
module load cuda

source activate /scratch/cs/imagedb/picsom/databases/vsgui/env/Visual-RFT

cd src/virft
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation

# vLLM support
pip install vllm==0.7.2

# fix transformers version
pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
