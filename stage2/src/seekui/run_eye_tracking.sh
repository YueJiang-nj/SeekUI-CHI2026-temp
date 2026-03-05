#!/bin/bash
#SBATCH --account=project_462001077
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-cpu=8000
#SBATCH --time=00:20:00
#SBATCH --output=Eye-tracking-RFT-Qwen2.5-think-test.out

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

export PYTHONUSERBASE=/scratch/project_462000803/zixin/RFT_env
export HF_HOME=/scratch/project_462000803/zixin/RFT_env
export TORCH_HOME=/scratch/project_462000803/zixin/RFT_env

export DIR_PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PYTHONPATH:$DIR_PWD"

echo $PYTHONPATH

export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1

export DATA_PATH=/scratch/project_462000803/zixin/data/vis_gui_train_dataset/vis_gui_train_qwen_sft_think
export CKPT_PATH=/scratch/project_462000803/zixin/RFT_env/model/checkpoints_nlp_full_think
export SAVE_PATH=./share_models/Qwen2.5-VL-3B-Instruct_GRPO_visgui_think_re
export WANDB_API_KEY=bf8ca0549e275efca1c336d0c11a2ef64949212d

export DEBUG_MODE="true"
export LOG_PATH="./GRPO_visgui_test.txt"

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
--run_name Qwen2.5-VL-3B_GRPO_visgui_think_re \
--save_steps 500 \
--save_only_model true \
--num_generations 4 \
--max_completion_length 512