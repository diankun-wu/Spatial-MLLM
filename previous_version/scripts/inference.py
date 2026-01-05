import os
import sys
import torch 
import tyro
import time

# add workspace to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    Qwen2_5_VL_VGGTForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


def main(
    video_path: str = "assets/arkitscenes_41069025.mp4",
    text: str = "How many chair(s) are in this room? Please answer with the only numerical value (e.g., 42, 3.14, etc.) within the <answer> </answer> tags.",
    model_type: str = "spatial-mllm-subset-sft",
    model_path: str = "Diankun/Spatial-MLLM-subset-sft",
    device: str = "cuda",
):
    torch.cuda.empty_cache()
    # load the model
    if "spatial-mllm" in model_type:
        model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
        )
        # Load the processor
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    elif "qwen2-5-vl" in model_type:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model.to(device)

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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if "spatial-mllm" in model_type:
        inputs.update({"videos_input": torch.stack(video_inputs) / 255.0})

    inputs = inputs.to(device)

    # Start time measurement
    time_0 = time.time()
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_p=0.001,
        use_cache=True,
    )
    time_taken = time.time() - time_0

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
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
