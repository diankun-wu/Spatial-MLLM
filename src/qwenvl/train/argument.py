from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import transformers

@dataclass
class ModelArguments:
    model_type: str = field(default="spatial-mllm")  # spatial-mllm, qwen2.5-vl, qwen2-vl
    vggt_checkpoints_path: Optional[str] = field(default="checkpoints/VGGT-1B/model.safetensors")
    spatial_embeds_layer_idx: int = field(default=-1)
    connector_type: str = field(default="mlp_add")  # mlp_add, mlp_cat, cross_attn

    pretrained_model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_spatial_encoder: bool = field(default=False)
    tune_mm_connector: bool = field(default=False)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    # If set, treat the input video frames as if they were sampled at this FPS (nominal FPS).
    # Used to compute the temporal spacing (second_per_grid_ts) for RoPE, especially when videos
    # are already provided as pre-extracted frames and the original FPS is unknown/unreliable.
    video_frame_fps: Optional[int] = field(default=None)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
