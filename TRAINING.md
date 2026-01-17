# Spatial-MLLM Training Guide

We provide a SFT training guide for Spatial-MLLM-Instruct-v1.1 models. 

## Prepare Pretrained Checkpoints

First, prepare the necessary pretrained model checkpoints and place them in the `checkpoints` directory.

```bash
mkdir -p checkpoints

# Download Qwen2.5-VL-3B-Instruct and VGGT-1B checkpoints
hf download  Qwen/Qwen2.5-VL-3B-Instruct --local-dir checkpoints/Qwen2.5-VL-3B-Instruct
hf download facebook/VGGT-1B --local-dir checkpoints/VGGT-1B
```

## Prepare Datasets Annotations

The Spatial-MLLM-v1.1-Instruct-135k model is trained on the following datasets:
- `spatial_mllm_mix_133k`: A mixture of our self-created data and ScanQA/SQA3D data. The annotations are available [here](https://huggingface.co/datasets/Diankun/Spatial-MLLM-Data/tree/main/annotation).
- `route_plan_scannet_2k`: A subset of route planning data used in [VLM-3R](https://github.com/VITA-Group/VLM-3R), containing around 2k samples from ScanNet.

The Spatial-MLLM-v1.1-Instruct-820k model is trained on the following datasets:
- `spatial_mllm_mix_203k`: A mixture of our self-created data and ScanQA/SQA3D data. The annotations are available [here](https://huggingface.co/datasets/Diankun/Spatial-MLLM-Data/tree/main/annotation).
- `route_plan_4k`: Route planning data used in [VLM-3R](https://github.com/VITA-Group/VLM-3R).
- `vsi_590k`: The 590k dataset from [Cambrian-S](https://github.com/cambrian-mllm/cambrian-s).
- `mindcube_21k`: The 21k dataset from [MindCube](https://github.com/mll-lab-nu/MindCube).

For `spatial_mllm_mix_133k` and `spatial_mllm_mix_203k`, please download the annotations from the provided links and place them in the `datasets/annotations` directory. 

For other annotation files, you may need to process them to align with our expected format (similar to [this instruction](https://github.com/QwenLM/Qwen3-VL/blob/e5c7e5c26af6a8bd65aec9388f3642cf6ea9d75c/qwen-vl-finetune/README.md?plain=1#L53)). We provide some scripts in the `scripts/preprocess` for your reference.

## Prepare Datasets Visual Data
For `vsi_590k` and `mindcube_21k`, they provide the corresponding visual data. 

For `spatial_mllm_mix` and `route_plan` data, you need download and process raw video data from [scannet](https://github.com/ScanNet/ScanNet), [scannetpp](https://scannetpp.mlsg.cit.tum.de/scannetpp/) and [arkitscenes](https://github.com/apple/ARKitScenes), and place them in the `datasets/visuals` directory. 

Before starting training, you may need to modify the [dataset configuration file](src/qwenvl/data/__init__.py) to ensure `annotation_path` and `data_path` are set correctly.

## Start Training
You can follow the instructions in [scripts/training/spatial_mllm_train_demo.sh](scripts/training/spatial_mllm_train_demo.sh) to start training. 
