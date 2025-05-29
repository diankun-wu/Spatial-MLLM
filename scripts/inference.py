import os
import sys
import torch 
# add workspace to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "Diankun/Spatial-MLLM-subset-sft"
video_path = "assets/arkitscenes_41069025.mp4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model
model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype="auto", 
    attn_implementation="flash_attention_2",
)
model = model.to(device)

# Load the processor
processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_PATH)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "nframes": 16,
            },
            {"type": "text", "text": "How many chair(s) are in this room? Please answer with the only numerical value (e.g., 42, 3.14, etc.) within the <answer> </answer> tags."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
_, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
if video_inputs is not None and len(video_inputs) > 0:
    inputs.update({"videos_input": torch.stack(video_inputs) / 255.0})
else:
    assert False, "No video inputs found in the messages."

inputs = inputs.to(device)
                
# Inference: Generation of the output
generated_ids = model.generate(
    **inputs, 
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.1,
    top_p=0.001,
    use_cache=True,
)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)