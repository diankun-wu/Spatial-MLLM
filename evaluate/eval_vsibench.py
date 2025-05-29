import argparse
import json
import os
import sys
import time
from copy import deepcopy
# add workspace to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
import torch
from decord import VideoReader  # Use decord for video resolution checking
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from utils import clean_text, vsi_reward

from evaluate.model import ModelConfig, get_model_and_processor

SFT_QUESTION_TEMPLATE = "{Question}"
SFT_TYPE_TEMPLATE = {
    "multiple choice": " Please answer with the option's letter from the given choices (e.g., A, B, etc.) within the <answer> </answer> tags.",
    "numerical": " Please answer with the only numerical value (e.g., 42, 3.14, etc.) within the <answer> </answer> tags.",
    "regression": " Please answer with the only numerical value (e.g., 42, 3.14, etc.) within the <answer> </answer> tags.",
    "verbal": " Please answer the question simply within the <answer> </answer> tags",
}

def load_vsi_evalset():
    file_path = os.path.abspath(__file__)
    vsi_annotation_path = os.path.join(os.path.dirname(file_path), "annotation", "eval_vsibench.json")
    with open(vsi_annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def prepare_single_message_eval(item, video_root, video_nframes):
    """Prepare message structure for a single eval data point."""
    if item["problem_type"] == "multiple choice":
        question = item["problem"] + "Options:\n"
        for op in item["options"]:
            question += op + "\n"
    else:
        question = item["problem"]

    content = []
    data_type = item["data_type"]
    data_path = os.path.normpath(os.path.join(video_root, item["path"]))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found for path {data_path}")
    
    mm_content = {"type": data_type}
    if data_type == "image":
        mm_content["image"] = data_path
        content.append(mm_content)
    elif data_type == "video":
        mm_content["video"] = data_path
        if video_nframes != -1:
            mm_content["nframes"] = video_nframes
        content.append(mm_content)
    else:
        raise ValueError(f"Unsupported data_type '{data_type}' found for path {data_path}.")

    content.append(
            {
                "type": "text",
                "text": SFT_QUESTION_TEMPLATE.format(Question=question)
                + SFT_TYPE_TEMPLATE[item["problem_type"]]
            }
        )
    msg = [{"role": "user", "content": content}]
    return msg

def preprocess_batch(batch_data, processor, model_config, video_root, video_nframes):
    batch_messages = []
    for i, x in enumerate(batch_data):
        msg = prepare_single_message_eval(x, video_root, video_nframes)
        batch_messages.append(msg)
    
    prompts_text = [
        processor.apply_chat_template(
            example, tokenize=False, add_generation_prompt=True
        )
        for example in batch_messages
    ]
    prompts_text_for_log = deepcopy(prompts_text)
        
    video_inputs = []
    for example in batch_messages:
        imgs, vids = process_vision_info(example)
        video_inputs.extend(vids)
        
    batch = processor(
        text=prompts_text,
        images=None,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    
    if "spatial-mllm" in model_config.model_type:
        if video_inputs is not None and len(video_inputs) > 0: # Check if video_inputs is not empty
            video_inputs = torch.stack(video_inputs) / 255.0 # [B, T, C, H, W]
            batch.update({"videos_input": video_inputs})
            
    return batch, prompts_text_for_log

def inference_batch(batch_inputs, model, processor, model_config):
    inputs = {key: val.to(model.device) if isinstance(val, torch.Tensor) else val 
            for key, val in batch_inputs.items()}
        
    # Generate response
    start_time = time.time()
    with torch.no_grad(), torch.amp.autocast(device_type=str(model.device), dtype=model.dtype):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=model_config.max_tokens,
            do_sample=True if model_config.temperature > 0 else False,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            use_cache=True,
        )
        end_time = time.time()
        print(f"Time taken for generation: {end_time - start_time} seconds")

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def postprocess_batch(batch_data, batch_output_text, prompts_text):
    batch_results = []
    for batch_idx, sample in enumerate(batch_data):
        model_output = batch_output_text[batch_idx]
        result_sample = {}
        result_sample['sample'] = sample.copy()
        result_sample["prompt"] = prompts_text[batch_idx]
        result_sample["model_output"] = model_output
        
        # --- if contains <answer> tags, extract answer, else use model_output as answer ---
        clean_ans = clean_text(model_output)
        result_sample["cleaned_model_output"] = clean_ans
        
        # --- get cleaned gt answer ---
        clean_ans_gt = clean_text(sample.get("solution", ""))
        result_sample["cleaned_gt_answer"] = clean_ans_gt
        
        # --- calculate reward ---
        result_sample["reward"] = vsi_reward(clean_ans_gt, clean_ans, sample['problem_type'])
        result_sample["correct"] = result_sample["reward"] == 1.0
        batch_results.append(result_sample)
    return batch_results

def calculate_metrics(results):
    """Calculate metrics from a list of results."""
    mean_acc_rewards = [s["reward"] for s in results if s["sample"].get("problem_type") != "regression" and "reward" in s]
    mean_mra_rewards = [s["reward"] for s in results if s["sample"].get("problem_type") == "regression" and "reward" in s and s.get("prediction") != "error"]

    final_metrics = {"mean_acc": 0.0, "mean_mra": 0.0, "mean_all": 0.0}
    if mean_acc_rewards:
            final_metrics["mean_acc"] = torch.tensor(mean_acc_rewards, dtype=torch.float32).mean().item()
    if mean_mra_rewards:
            final_metrics["mean_mra"] = torch.tensor(mean_mra_rewards, dtype=torch.float32).mean().item()
    if mean_acc_rewards or mean_mra_rewards:
        all_rewards = torch.cat([torch.tensor(mean_acc_rewards, dtype=torch.float32), torch.tensor(mean_mra_rewards, dtype=torch.float32)])
        final_metrics["mean_all"] = all_rewards.mean().item()
    return final_metrics

def save_results(output_path: str, results, final_acc):
    """Save evaluation results to file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"results": results, "final_acc": [final_acc]},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error writing results to output file: {e}")

@ray.remote(num_gpus=1)
def evaluate_vsibench(vsi_data, model_config, output_path, video_root, video_nframes, batch_size):
    """Evaluate model on a specific dataset. Batching rule: accumulate up to BATCH_SIZE samples; if a video's resolution is different from the first video in the current batch, flush the batch before this sample."""

    # --- cache video resolutions to avoid reopening the same file repeatedly ---
    resolution_cache = {}

    def get_resolution(path):
        """Return (width, height) of the video using decord; utilize cache when available."""
        if path in resolution_cache:
            return resolution_cache[path]
        try:
            vr = VideoReader(path, num_threads=1)
            # Decord frame shape: (H, W, C)
            h, w, _ = vr[0].shape
            resolution_cache[path] = (w, h)
        except Exception as e:
            raise RuntimeError(f"Failed to read video {path} with decord: {e}")
        return resolution_cache[path]

    model, processor = get_model_and_processor(model_config)
    
    final_output = []

    # Helper function to process the accumulated batch and flush results
    def handle_batch(batch_data, processed_idx):
        """Run inference on one accumulated batch, update metrics & save."""
        nonlocal final_output
        if not batch_data:
            return
        batch_llm_inputs, prompts_text = preprocess_batch(batch_data, processor, model_config, video_root, video_nframes)
        batch_output_text = inference_batch(batch_llm_inputs, model, processor, model_config)
        batch_results = postprocess_batch(batch_data, batch_output_text, prompts_text)
        final_output.extend(batch_results)

        # --- calculate metrics ---
        current_metrics = calculate_metrics(final_output)
        save_results(output_path, final_output, current_metrics)
        processed_count = len(final_output)
        print(
            f"Processed up to overall index {processed_idx}, saved {processed_count} samples."
        )

    current_batch = []
    current_resolution = None  # (w,h) of first video in the current batch

    for idx, item in enumerate(tqdm(vsi_data, desc="Processing vsibench batches")):
        video_path = os.path.normpath(os.path.join(video_root, item["path"]))
        video_res = get_resolution(video_path)

        # If starting a new batch
        if not current_batch:
            current_resolution = video_res

        # Check if resolution changes OR batch size limit reached
        if video_res != current_resolution or len(current_batch) >= batch_size:
            # Flush the current batch before adding this item
            handle_batch(current_batch, idx - 1)
            current_batch = []
            current_resolution = video_res

        current_batch.append(item)

        # In case the batch exactly reaches BATCH_SIZE after appending, flush now
        if len(current_batch) >= batch_size:
            handle_batch(current_batch, idx)
            current_batch = []
            current_resolution = None

    # Flush remaining samples after the loop ends
    if current_batch:
        handle_batch(current_batch, len(vsi_data) - 1)

    return final_output


def main(args):
    model_config = ModelConfig(
        model_path=args.model_path,
        model_type=args.model_type,
    )
    output_dir = os.path.join("eval_results", f"eval_vsibench")
    os.makedirs(output_dir, exist_ok=True)
    vsi_data = load_vsi_evalset()
    
    n_gpu = torch.cuda.device_count()
    
    ray.init()
    features = []
    per_gpu_data_length = len(vsi_data) // n_gpu
    for i in range(n_gpu):
        data_gpu = vsi_data[i * per_gpu_data_length : (i + 1) * per_gpu_data_length]
        output_path_gpu = os.path.join(output_dir, f"results_{model_config.model_type}_{i}.json")
        features.append(evaluate_vsibench.remote(
            data_gpu, 
            model_config=model_config, 
            output_path=output_path_gpu,
            video_root=args.video_root,
            video_nframes=args.nframes,
            batch_size=args.batch_size,
        )
    )

    ret = ray.get(features)
    final_output = []
    for item in ret:
        final_output.extend(item)
        
    # --- calculate final metrics ---
    final_acc_dict = calculate_metrics(final_output)
    save_results(os.path.join(output_dir, f"results_{model_config.model_type}.json"), final_output, final_acc_dict)
    print(f"Finished evaluation for vsibench.")
    print(f"Final Metrics: {final_acc_dict}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on VSIBench dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--video_root", type=str, required=True, help="Root directory for video files.")
    parser.add_argument("--model_type", type=str, default="spatial-mllm-tiny", help="Type of the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--nframes", type=int, default=16, help="Number of frames to sample from each video.")
    args = parser.parse_args()


    main(args)