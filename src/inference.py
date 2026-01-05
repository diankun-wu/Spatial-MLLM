import os
import sys
import time

import numpy as np
import torch
import tyro
from PIL import Image

# add workspace to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_vl_utils import process_vision_info


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


def load_model_and_processor(model_type: str, model_path: str):
    """Load model and processor based on type."""
    if "spatial-mllm" in model_type:
        from transformers import Qwen2_5_VLProcessor

        from src.qwenvl.model.spatial_mllm import SpatialMLLMConfig, SpatialMLLMForConditionalGeneration

        config = SpatialMLLMConfig.from_pretrained(model_path)
        model = SpatialMLLMForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype="bfloat16",
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        return model, processor

    if "qwen2.5-vl" in model_type:
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="bfloat16",
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        return model, processor

    raise ValueError(f"Unknown model type: {model_type}")


def main(
    video_path: str = "assets/arkitscenes_41069025.mp4",
    text: str = "How many chair(s) are in this room?\nPlease answer the question using a single word or phrase.",  # na question
    # text: str = "Measuring from the closest point of each object, what is the distance between the sofa and the stove (in meters)?\nPlease answer the question using a single word or phrase.",  # na question
    # text: str = "If I am standing by the stove and facing the tv, is the sofa to my front-left, front-right, back-left, or back-right?\nThe directions refer to the quadrants of a Cartesian plane (if I am standing at the origin and facing along the positive y-axis).Options:\nA. back-left\nB. front-right\nC. back-right\nD. front-left\nAnswer with the option's letter from the given choices directly.",  # mca question
    model_type: str = "spatial-mllm",
    model_path: str = "checkpoints/Spatial-MLLM-v1.1-Instruct-135K",
):
    torch.cuda.empty_cache()

    # load the model
    model, processor = load_model_and_processor(model_type, model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "nframes": 16,
                },
                {
                    "type": "text",
                    "text": text,
                },
            ],
        }
    ]

    # Preparation for inference
    prompts_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    batch = processor(
        text=[prompts_text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )

    if "spatial-mllm" in model_type:
        batch = prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs)

    batch.to(model.device)
    if "image_tchw" in batch and batch["image_tchw"] is not None:
        batch["image_tchw"] = [image_tchw_i.to(model.device) for image_tchw_i in batch["image_tchw"]]
    if "video_tchw" in batch and batch["video_tchw"] is not None:
        batch["video_tchw"] = [video_tchw_i.to(model.device) for video_tchw_i in batch["video_tchw"]]

    generation_kwargs = dict(
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_p=0.001,
        use_cache=True,
    )

    # Start time measurement
    time_0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**batch,**generation_kwargs)
    time_taken = time.time() - time_0

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch["input_ids"], generated_ids)
    ]
    num_generated_tokens = sum(len(ids) for ids in generated_ids_trimmed)

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(f"Time taken for inference: {time_taken:.2f} seconds")
    print(f"GPU Memory taken for inference: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Number of generated tokens: {num_generated_tokens}")
    print(f"Time taken per token: {time_taken / num_generated_tokens:.4f} seconds/token")
    print(f"Output: {output_text}")


if __name__ == "__main__":
    tyro.cli(main, description="Run inference.")
