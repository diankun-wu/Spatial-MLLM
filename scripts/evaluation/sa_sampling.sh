# !/bin/bash

cd "$(dirname "$0")"
cd ../..

# download VGGT-1B model checkpoint
# hf download facebook/VGGT-1B --local-dir checkpoints/VGGT-1B

# run sampling
BASE_DIR="datasets/evaluation/vsibench"
mkdir -p "$BASE_DIR/sa_sampling_16f"
mkdir -p "$BASE_DIR/uniform_sampling_16f"

run_sampling() {
    dataset=$1
    temp_dir="${BASE_DIR}/${dataset}_temp"
    
    python src/sampling/sa_sampling.py \
        --video_folder "${BASE_DIR}/${dataset}" \
        --model_path checkpoints/VGGT-1B \
        --output_folder "$temp_dir"

    mv "${temp_dir}/sa_sampling" "${BASE_DIR}/sa_sampling_16f/${dataset}"
    mv "${temp_dir}/uniform_sampling" "${BASE_DIR}/uniform_sampling_16f/${dataset}"
    rmdir "$temp_dir"
}

run_sampling "scannet"
run_sampling "scannetpp"
run_sampling "arkitscenes"