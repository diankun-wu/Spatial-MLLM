#!/bin/bash

cd "$(dirname "$0")"
cd ../..

OUTPUT_ROOT="results/vsibench"
mkdir -p "$OUTPUT_ROOT"

MODEL_PATH="checkpoints/Spatial-MLLM-v1.1-Instruct-135K"
MODEL_NAME=$(echo "$MODEL_PATH" | cut -d'/' -f2)
MODEL_TYPE="spatial-mllm"

nframes=(16)

for nframe in "${nframes[@]}"; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXP_DIR="${OUTPUT_ROOT}/${MODEL_NAME}-${nframe}f"
    LOG_FILE="${EXP_DIR}/run.log"

    mkdir -p "$EXP_DIR"
    
    echo "----------------------------------------------------------------"
    echo "Starting Sweep: [nframes=$nframe]"
    echo "Artifacts dir: $EXP_DIR"
    echo "----------------------------------------------------------------"

    {
        echo "================ EXPERIMENT INFO ================"
        echo "Time: $TIMESTAMP"
        echo "Params: NFRAMES=$nframe"
        echo "Commit: $(git rev-parse HEAD)"
        echo "================================================="
    } > "$LOG_FILE"

    # --- run experiment ---
    python src/evaluation/vsibench/eval_vsibench.py \
        --model_path $MODEL_PATH \
        --model_type $MODEL_TYPE \
        --nframes $nframe \
        --annotation_dir "datasets/evaluation/vsibench" \
        --video_dir "datasets/evaluation/vsibench" \
        --batch_size 1 \
        --output_dir "$EXP_DIR" \
        --output_name "eval_result" \
        2>&1 | tee -a "$LOG_FILE"
        
    echo ">>> Experiment Finished. Results in $EXP_DIR"
done