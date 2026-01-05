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
    <br>
    NeurIPS 2025 (Spotlight)
</p>

<a href='https://arxiv.org/abs/2505.23747'><img src='https://img.shields.io/badge/arXiv-2505.23747-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://diankun-wu.github.io/Spatial-MLLM/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a><img src='https://img.shields.io/badge/License-MIT-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;


![Teaser Visualization](assets/teaser-spatialmllm.png)

</div>
<strong>Spatial-MLLM:</strong> We propose Spatial-MLLM, a method that significantly enhances the visual-based spatial intelligence of existing video MLLMs. As shown, Spatial-MLLM can understand and reason about the underlying scene based on video input and achieves SOTA performance in a wide range of spatial reasoning tasks.
</div>

## 📢 News
- 🎉[01/05/2026] We release two new SFT models: [Spatial-MLLM-v1.1-Instruct-135K](https://huggingface.co/Diankun/Spatial-MLLM-v1.1-Instruct-135K) and [Spatial-MLLM-v1.1-Instruct-820K](https://huggingface.co/Diankun/Spatial-MLLM-v1.1-Instruct-820K). 
- 🎉[01/05/2026] We refactor our repo and release the refined SFT training code for Spatial-MLLM-v1.1-Instruct. We also release code for space-aware frame sampling. 
- 🎉[05/30/2025] We release [Spatial-MLLM-subset-sft](https://huggingface.co/Diankun/Spatial-MLLM-subset-sft), which is trained on a subset of our proposed Spatial-MLLM-120k dataset. We also release the evaluation code on VSI-Bench. You can refer to `previous_version` to use and evaluate this model. 
- 🔥[05/30/2025] We release "Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence". Check our [project page](https://diankun-wu.github.io/Spatial-MLLM/) and [arXiv paper](https://arxiv.org/pdf/).

## 🌟 Overview

![Pipeline Visualization](assets/pipeline-spatialmllm.png)

</div>

Overview of Spatial-MLLM. Our model is composed of a 2D visual encoder, a spatial encoder which is initialized from a feed-forward visual geometry foundation model, a connector, and a large language model backbone. At inference time, we incorporate a space-aware frame sampling strategy to select spatially informative frames when the number of input frames is limited due to GPU memory constraints.


## ⚙️ Setup

### 1. Clone Repository
```bash
git clone https://github.com/diankun-wu/Spatial-MLLM
cd Spatial-MLLM
```
### 2. Environment Setup

We use conda to manage the environment. First, create conda environment:

```bash
conda create -n spatial-mllm python=3.10 -y
conda activate spatial-mllm
```

Install PyTorch 2.6.0 with CUDA 12.4 support:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Install other required packages:

```bash
pip install transformers==4.51.3
pip install accelerate datasets decord deepspeed einops matplotlib pandas python_Levenshtein qwen_vl_utils ray safetensors tqdm tyro wandb
```

Finally, download and install the pre-built wheel for Flash Attention 2:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## 💻 Inference

To run inference, you can use the script `src/inference.py`. For example:
```bash
python src/inference.py \
    --model_path Diankun/Spatial-MLLM-v1.1-Instruct-135K \
    --model_type spatial-mllm \
    --text "How many chair(s) are in this room?\nPlease answer the question using a single word or phrase."
```

## 📊 Evaluation
### Evaluation on VSI-Bench

To evaluate the model on VSI-Bench, you should first download the VSI-Bench dataset and place it in the `datasets/evaluation/vsibench` directory. You can use the following command:
```bash
# download the VSI-Bench dataset from Hugging Face
hf download nyu-visionx/VSI-Bench \
    --local-dir datasets/evaluation/vsibench \
    --repo-type dataset

# extract the downloaded dataset
for f in datasets/evaluation/vsibench/*.zip; do
    unzip "$f" -d datasets/evaluation/vsibench
done
```

Download the Spatial-MLLM-v1.1-Instruct models to the `checkpoints` directory (recommended when using multiple GPUs):
```bash
mkdir -p checkpoints

# download Spatial-MLLM-v1.1-Instruct-135K
hf download Diankun/Spatial-MLLM-v1.1-Instruct-135K \
    --local-dir checkpoints/Spatial-MLLM-v1.1-Instruct-135K

# download Spatial-MLLM-v1.1-Instruct-820K
hf download Diankun/Spatial-MLLM-v1.1-Instruct-820K \
    --local-dir checkpoints/Spatial-MLLM-v1.1-Instruct-820K
```

Then you can use the provided bash script to evaluate the model. 
```bash
bash scripts/evaluation/evaluate_vsibench_spatial_mllm.sh
```
The script will automatically use all available GPUs. If you want to specify the GPUs to use, you can set the `CUDA_VISIBLE_DEVICES` environment variable before running the script.

### Using Space-aware Frame Sampling
To use the space-aware frame sampling strategy during evaluation, we recommend using our pre-sampled frames. You can download them using the following command:
```bash
# Download the zip file
hf download Diankun/Spatial-MLLM-Data evaluation/vsibench/sa_sampling_16f.zip \
    --repo-type dataset \
    --local-dir . 

# Unzip the file
unzip evaluation/vsibench/sa_sampling_16f.zip -d datasets/evaluation/vsibench/arkitscenes_sampling_16f
```

You can also sample frames using our provided script:
```bash
bash scripts/evaluation/sa_sampling.sh
```

This script will use space-aware frame sampling to sample frames for all videos in `datasets/evaluation/vsibench` and save the sampled frames to `datasets/evaluation/vsibench/sa_sampling_16f`. 

Then, you can use the provided bash script to evaluate the model with the sampled frames.
```bash
bash scripts/evaluation/evaluate_vsibench_spatial_mllm_w_sa_sampling.sh
```

Here are our evaluation results for Spatial-MLLM-v1.1-Instruct and baseline models on VSI-Bench (16 frames input):
<table>
<thead>
  <tr>
    <th rowspan="2" style="text-align: center; vertical-align: middle;">Model</th>
    <th colspan="3" style="text-align: center; border-bottom: 1px solid #ddd;">VSIBench Micro</th>
    <th colspan="3" style="text-align: center; border-bottom: 1px solid #ddd;">VSIBench Macro</th>
  </tr>
  <tr>
    <th style="text-align: center;">Acc</th>
    <th style="text-align: center;">MRA</th>
    <th style="text-align: center;">All</th>
    <th style="text-align: center;">Acc</th>
    <th style="text-align: center;">MRA</th>
    <th style="text-align: center;">All</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><strong>Qwen2.5-VL-3B-Instruct</strong></td>
    <td style="text-align: center;">35.42</td>
    <td style="text-align: center;">20.72</td>
    <td style="text-align: center;">27.86</td>
    <td style="text-align: center;">37.12</td>
    <td style="text-align: center;">21.65</td>
    <td style="text-align: center;">30.93</td>
  </tr>
  <tr>
    <td><strong>Qwen2.5-VL-3B-Instruct-135K</strong></td>
    <td style="text-align: center;">46.91</td>
    <td style="text-align: center;">52.60</td>
    <td style="text-align: center;">49.84</td>
    <td style="text-align: center;">46.16</td>
    <td style="text-align: center;">52.81</td>
    <td style="text-align: center;">48.82</td>
  </tr>
  <tr>
    <td><strong>Spatial-MLLM-v1.1-Instruct-135K</strong></td>
    <td style="text-align: center;">49.28</td>
    <td style="text-align: center;">52.88</td>
    <td style="text-align: center;">51.13</td>
    <td style="text-align: center;">49.12</td>
    <td style="text-align: center;">53.88</td>
    <td style="text-align: center;">51.02</td>
  </tr>
  <tr>
    <td><strong>Spatial-MLLM-v1.1-Instruct-135K (SA Sampling)</strong></td>
    <td style="text-align: center;">52.13</td>
    <td style="text-align: center;">53.33</td>
    <td style="text-align: center;">52.75</td>
    <td style="text-align: center;">52.84</td>
    <td style="text-align: center;">54.46</td>
    <td style="text-align: center;">53.49</td>
  </tr>
  <tr>
    <td><strong>Spatial-MLLM-v1.1-Instruct-820K</strong></td>
    <td style="text-align: center;">49.56</td>
    <td style="text-align: center;">57.27</td>
    <td style="text-align: center;">53.53</td>
    <td style="text-align: center;">48.02</td>
    <td style="text-align: center;">57.39</td>
    <td style="text-align: center;">51.77</td>
  </tr>
  <tr>
    <td><strong>Spatial-MLLM-v1.1-Instruct-820K  (SA Sampling)</strong></td>
    <td style="text-align: center;">50.60</td>
    <td style="text-align: center;">57.68</td>
    <td style="text-align: center;">54.24</td>
    <td style="text-align: center;">50.12</td>
    <td style="text-align: center;">58.09</td>
    <td style="text-align: center;">53.30</td>
  </tr>
  
</tbody>
</table>

### Evaluation on ScanQA
We also provide evaluation scripts for ScanQA:
```
bash scripts/evaluation/evaluate_scanqa_spatial_mllm.sh
``` 
Note that you need to download and preprocess the scannet raw video data and place them in `datasets/visuals/scannet/videos` before evaluation.

After evaluation, you will get the results saved in `results/scanqa/Spatial-MLLM-v1.1-Instruct-135K-16f.json`. Then use the following command to calculate the metrics:
```python
python src/evaluation/scanqa/score_scanqa.py \
    --input-file results/scanqa/Spatial-MLLM-v1.1-Instruct-135K-16f.json
```

## Training

You can refer to our [TRAINING.md](TRAINING.md) for detailed training instructions.

## 📚  Citation

If you find it useful for your research and applications, please cite our paper using this BibTeX:
```bibtex
@article{wu2025spatialmllmboostingmllmcapabilities,
    title={Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence},
    author={Wu, Diankun  and Liu, Fangfu and Hung, Yi-Hsin and Duan, Yueqi},
    journal={arXiv preprint arXiv:2505.23747},
    year={2025}
}
```

## Acknowledgements

Thanks to these great repositories: [thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space), [VGGT](https://github.com/facebookresearch/vggt), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [open-r1](https://github.com/huggingface/open-r1), [R1-V](https://github.com/Deep-Agent/R1-V), [Video-3D-LLM](https://github.com/LaVi-Lab/Video-3D-LLM), [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [VLM-3R](https://github.com/VITA-Group/VLM-3R), [MindCube](https://github.com/mll-lab-nu/MindCube), [Cambrian-S](https://github.com/cambrian-mllm/cambrian-s) and many other inspiring works in the community.
