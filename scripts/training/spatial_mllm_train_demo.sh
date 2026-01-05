#!/bin/bash
set -euo pipefail

# Set environment variables
# export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_API_KEY=YOUR_WANDB_API_KEY  # Replace with your WandB API key 
export WANDB_PROJECT="Spatial-MLLM-SFT"
export WANDB_ENTITY=YOUR_WANDB_ENTITY  # Replace with your WandB entity/team name

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/training/zero3.json

# Model configuration
model_type=spatial-mllm
vggt_checkpoints_path=checkpoints/VGGT-1B/model.safetensors
spatial_embeds_layer_idx=-1
connector_type=mlp_add 
pretrained_model_name_or_path=checkpoints/Qwen2.5-VL-3B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=7e-6
mm_projector_lr=2e-5
weight_decay=0.1
max_grad_norm=1.0
batch_size=1 
grad_accum_steps=8

# Training entry point
entry_file=src/qwenvl/train/train_qwen.py

# Dataset configuration
datasets="spatial_mllm_mix_133k,route_plan_scannet_2k"

# Data configuration
max_pixels=324576
min_pixels=293216
video_max_frame_pixels=324576
video_min_frame_pixels=293216
video_max_frames=16
video_min_frames=16
video_frame_fps=4

# Output configuration
timestamp=$(date +'%Y%m%d_%H%M%S')
base_run_name="spatial-mllm-sft"
run_name="${timestamp}_${base_run_name}"
output_dir=./output/${run_name}
mkdir -p ${output_dir}
logfile="${output_dir}/$(date +'%Y%m%d_%H%M%S')_train.log"

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_type ${model_type} \
    --vggt_checkpoints_path ${vggt_checkpoints_path} \
    --spatial_embeds_layer_idx ${spatial_embeds_layer_idx} \
    --pretrained_model_name_or_path "${pretrained_model_name_or_path}" \
    --dataset_use ${datasets} \
    --tune_mm_vision False \
    --tune_mm_spatial_encoder False \
    --tune_mm_connector True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --video_max_frame_pixels ${video_max_frame_pixels} \
    --video_min_frame_pixels ${video_min_frame_pixels} \
    --video_max_frames ${video_max_frames} \
    --video_min_frames ${video_min_frames} \
    --video_frame_fps ${video_frame_fps} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.0625 \
    --learning_rate ${lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --weight_decay ${weight_decay} \
    --warmup_ratio 0.03 \
    --max_grad_norm ${max_grad_norm} \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=6 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} 2>&1 | tee -a "${logfile}"