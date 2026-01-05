import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs):
    """
        Prepare inputs for Spatial MLLM model.
        Batch: Dict return by the processor
        video_input and image_inputs is returned by process_vision_info
        
        video_inputs: List[torch.Tensor[Int]] | List[torch.Tensor[Float]] | List[List[PIL.Image]]
        image_inputs: List[PIL.Image]
    """
    video_tchw = []
    image_tchw = []

    if video_inputs:
        for video_input in video_inputs:
            if isinstance(video_input, torch.Tensor):
                video_input = video_input.float() / 255.0  # Normalize to [0, 1]
            elif isinstance(video_input, list) and all(isinstance(img, Image.Image) for img in video_input):
                # Convert list of PIL Images to tensor
                video_input = torch.stack([torch.tensor(np.array(img)).permute(2, 0, 1) for img in video_input]).float() / 255.0
            else:
                raise ValueError("Unsupported video input format.")
            video_tchw.append(video_input)
    
    if image_inputs:
        for image_input in image_inputs:
            if isinstance(image_input, Image.Image):
                image_input = torch.tensor(np.array(image_input)).permute(2, 0, 1).float() / 255.0
            else:
                raise ValueError("Unsupported image input format.")
            image_tchw.append(image_input)

    batch.update({
        "video_tchw": video_tchw if video_tchw else None,
        "image_tchw": image_tchw if image_tchw else None,
    })

    return batch
        
def chunk_dataset(dataset: List[Dict], num_shards: int) -> List[List[Dict]]:
    """Split dataset into roughly equal shards."""
    if num_shards <= 0:
        return [dataset]

    chunk_size = math.ceil(len(dataset) / num_shards)
    return [
        [dataset[i] for i in range(start, min(start + chunk_size, len(dataset)))]
        for start in range(0, len(dataset), chunk_size)
    ]

def flatten(nested: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists."""
    return [item for sublist in nested for item in sublist]

def save_json(output_path: str | Path, data: Any):
    """Save data to json file."""
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error writing results to output file: {e}")

def save_jsonl(output_path: str | Path, data: List[Any]):
    """Save list of data to jsonl file."""
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
                
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error writing results to output file: {e}")
