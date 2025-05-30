<div align="center">

# ✨Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence✨

<p align="center">
    <a href="https://github.com/diankun-wu/">Diankun Wu</a><sup>1*</sup>,
    <a href="https://liuff19.github.io/">Fangfu Liu</a><sup>1*</sup>,
    <a href="https://github.com/CindyHung20/">Yi-Hsin Hung</a><sup>1</sup>,
    <a href="https://duanyueqi.github.io/">Yueqi Duan</a><sup>1</sup>,
    <br>
    <sup>*</sup>Equal Contribution.
    <br>
    <sup>1</sup>Tsinghua University
</p>

<a href='https://arxiv.org/abs/2505.23747'><img src='https://img.shields.io/badge/arXiv-2505.23747-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://diankun-wu.github.io/Spatial-MLLM/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a><img src='https://img.shields.io/badge/License-MIT-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;


![Teaser Visualization](assets/teaser-spatialmllm.png)

</div>
<strong>Spatial-MLLM:</strong> We propose Spatial-MLLM, a method that significantly enhances the visual-based spatial intelligence of existing video MLLMs. As shown, Spatial-MLLM can understand and reason about the underlying scene based on video input and achieves SOTA performance in a wide range of spatial reasoning tasks.
</div>

## 📢 News
- 🎉[05/30/2025] We release [Spatial-MLLM-subset-sft](https://huggingface.co/Diankun/Spatial-MLLM-subset-sft), which is training on a subset of our proposed Spatial-MLLM-120k dataset. We also release the evaluation code on VSI-Bench.
- 🔥[05/30/2025] We release "Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence". Check our [project page](https://diankun-wu.github.io/Spatial-MLLM/) and [arXiv paper](https://arxiv.org/pdf/).

## 🌟 Overview

![Pipeline Visualization](assets/pipeline-spatialmllm.png)

</div>

Overview of Spatial-MLLM. Our model is composed of a 2D visual encoder, a spatial encoder which is initialized from a feed-forward visual geometry foundation model, a connector, and a large language model backbone. At inference time, we incorporate a space-aware frame sampling strategy to select spatially informative frames when the number of input frames is limited due to GPU memory constraints.

## 🎉 Performance

![Results Visualization](assets/eval_VSIbench.png)
![Results Visualization](assets/eval_scanqa_sqa3d.png)

## ⚙️ Setup

### 1. Clone Repository
```bash
git clone https://github.com/diankun-wu/Spatial-MLLM
cd Spatial-MLLM
```

### 2. Environment Setup

1. **Create conda environment:**

```bash
conda create -n spatial-mllm python=3.10 -y
conda activate spatial-mllm
```

2. **Install required packages for inference and evaluation:**

```bash
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # Adjust the CUDA version as needed
pip install transformers==4.51.3 accelerate==1.5.2 qwen_vl_utils decord ray Levenshtein
pip install pip install flash-attn --no-build-isolation
```

## 💻 Inference and Evaluation

### Inference

For inference, please refer to scripts/inference.py.
```bash
python scripts/inference.py
```
This command will download the Spatial-MLLM-subset-sft model from huggingface and perform inference on the provided video input. You can specify the video path and other parameters in the script.

### Evaluation on VSI-Bench

To evaluate the model on VSI-Bench, you should first download the VSI-Bench dataset and place it in the `evaluate/annotation/VSIBench` directory. You can use the following command:
```bash
# download the VSI-Bench dataset from Hugging Face
huggingface-cli download --resume-download nyu-visionx/VSI-Bench --local-dir evaluate/annotation/VSIBench --repo-type dataset

# extract the downloaded dataset
unzip evaluate/annotation/VSIBench/arkitscenes.zip -d evaluate/annotation/VSIBench
unzip evaluate/annotation/VSIBench/scannet.zip -d evaluate/annotation/VSIBench
unzip evaluate/annotation/VSIBench/scannetpp.zip -d evaluate/annotation/VSIBench
```

Then you can use the following command to evaluate the model:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set the GPU devices you want to use

python evaluate/eval_vsibench.py \
    --model_path Diankun/Spatial-MLLM-subset-sft \
    --video_root evaluate/annotation/VSIBench \
    --model_type spatial-mllm-subset-sft \
    --batch_size 8 \
```
or you can use the provided bash script:
```bash
bash scripts/evaluate_vsibench.sh
```


## 🚀Todo List

- [ ] Release the full Spatial-MLLM model and the code for space-aware frame sampling.
- [ ] Release the evaluation code on ScanQA and SQA3D.
- [ ] Release the training code for Spatial-MLLM.
- [ ] Release the Spatial-MLLM-120k dataset and its creation scripts.


## 📚  Citation

If you find it useful for your research and applications, please cite our paper using this BibTeX:
```bibtex
@article{}
```

## Acknowledgements

Thanks to these great repositories: [thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space), [VGGT](https://github.com/facebookresearch/vggt), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL),[open-r1](https://github.com/huggingface/open-r1), [R1-V](https://github.com/Deep-Agent/R1-V), [VLM-R1](https://github.com/om-ai-lab/VLM-R1) and many other inspiring works in the community.
