#!/bin/bash
#SBATCH --account=project_
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-cpu=8000
#SBATCH --time=00:20:00
#SBATCH --output=SeekUI.out

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

export PYTHONUSERBASE=/path/to/RFT_env
export HF_HOME=/path/to/RFT_env
export TORCH_HOME=/path/to/RFT_env

export DIR_PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PYTHONPATH:$DIR_PWD"

echo $PYTHONPATH

export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1

export DATA_PATH=/path/to/data/vsgui_train_seekui_qwen2_5_explanation
export CKPT_PATH=/path/to/model/SeekUI_sft
export SAVE_PATH=./share_models/SeekUI
export WANDB_API_KEY=your_wandb_api_key

export DEBUG_MODE="true"
export LOG_PATH="./SeekUI.txt"

srun python -m torch.distributed.run \
--nnodes=$SLURM_JOB_NUM_NODES \
--nproc_per_node=8 \
--rdzv_id=$SLURM_JOB_ID \
--rdzv_backend=c10d \
--rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
src/open_r1/grpo.py \
--output_dir ${SAVE_PATH}  \
--model_name_or_path ${CKPT_PATH} \
--dataset_name ${DATA_PATH} \
--deepspeed ./local_scripts/zero3.json \
--max_prompt_length 1024 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--logging_steps 1 \
--bf16 true \
--report_to wandb \
--gradient_checkpointing false \
--attn_implementation flash_attention_2 \
--max_pixels 401408 \
--num_train_epochs 30 \
--run_name SeekUI \
--save_steps 500 \
--save_only_model true \
--num_generations 4 \
--max_completion_length 512