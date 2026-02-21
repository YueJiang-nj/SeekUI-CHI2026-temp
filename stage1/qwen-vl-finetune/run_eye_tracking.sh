#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --constraint=hopper
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --output=seekui-sft.out

module load mamba
module load triton/2025.1-gcc
module load cuda

source activate /scratch/work/guoz3/environment/qwen

export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/work/guoz3/environment/qwen
export TORCH_HOME=/scratch/work/guoz3/environment/qwen

MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
CACHE_DIR="/scratch/work/guoz3/environment/qwen/hub"
DATASETS="vsgui_text%100"

python -m torch.distributed.run --nproc_per_node=1 --master_port=4622 \
qwenvl/train/train_qwen.py \
--model_name_or_path $MODEL_PATH \
--tune_mm_llm True \
--tune_mm_vision False \
--tune_mm_mlp True \
--dataset_use $DATASETS \
--output_dir /scratch/cs/imagedb/picsom/databases/vsgui/zixin/checkpoints_seekui_sft \
--cache_dir $CACHE_DIR \
--bf16 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--learning_rate 2e-7 \
--mm_projector_lr 1e-5 \
--vision_tower_lr 1e-6 \
--optim adamw_torch \
--model_max_length 4096 \
--data_flatten True \
--data_packing True \
--max_pixels 451584 \
--min_pixels 12544 \
--base_interval 2 \
--video_max_frames 8 \
--video_min_frames 4 \
--video_max_frame_pixels 1304576 \
--video_min_frame_pixels 200704 \
--num_train_epochs 200 \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--logging_steps 10 \
--save_steps 2000 \
--save_total_limit 3 \
--dataloader_num_workers 4 \
--deepspeed ./scripts/zero3.json \

source deactivate
