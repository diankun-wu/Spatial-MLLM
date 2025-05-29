#!/bin/bash

# change to workspace root directory
cd "$(dirname "$0")"
cd ..

export CUDA_VISIBLE_DEVICES=1,2,3,5,6,7

python evaluate/eval_vsibench.py \
    --model_path Diankun/Spatial-MLLM-subset-sft \
    --video_root evaluate/annotation/VSIBench \
    --model_type spatial-mllm-subset-sft \
    --batch_size 8 \
    --nframes 8 \
