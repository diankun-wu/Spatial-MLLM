import os
import sys
import torch 
# add workspace to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class ModelConfig:
    """Arguments related to model loading and generation parameters."""
    model_path: str
    model_type: str
    temperature: float = 0.1
    top_p: float = 0.001
    max_tokens: int = 1024
    use_vllm: bool = False

def get_model_and_processor(config: ModelConfig):
    if "spatial-mllm" in config.model_type:
        from src.models import Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor
        model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
            config.model_path, 
            torch_dtype="auto", 
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5_VLProcessor.from_pretrained(config.model_path)
    else:
        from transformers import AutoModelForCausalLM, AutoProcessor
        model = AutoModelForCausalLM.from_pretrained(config.model_path)
        processor = AutoProcessor.from_pretrained(config.model_path)
    return model, processor