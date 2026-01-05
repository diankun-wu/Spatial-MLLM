import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch.multiprocessing as mp

sys.path.append(str(Path(__file__).resolve().parents[3]))

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from datasets import load_dataset
from src.evaluation.utils.common_utils import (
    chunk_dataset,
    flatten,
    prepare_spatial_mllm_inputs,
    save_json,
    setup_logging,
)
from src.evaluation.vsibench.dataset_utils import MCA_QUESTION_TYPES, NA_QUESTION_TYPES, clean_text, vsi_reward

# Constants
SFT_QUESTION_TEMPLATE = "{Question}"
SFT_TYPE_TEMPLATE = {
    "mca": "Answer with the option's letter from the given choices directly.",
    "na": "Please answer the question using a single word or phrase.",
}

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


def build_user_message(item: Dict, video_dir: Path, video_nframes: int) -> Dict:
    """Create the chat-style message payload for a single sample."""
    # build question
    raw_question = SFT_QUESTION_TEMPLATE.format(Question=item["question"])
    q_type = item["question_type"]
    if q_type in MCA_QUESTION_TYPES:
        options = item.get("options") or []
        if not options:
            raise ValueError("Multiple-choice samples must include 'options'.")
        options_text = "Options:\n" + "\n".join(options)
        question = f"{raw_question}\n{options_text}\n{SFT_TYPE_TEMPLATE['mca']}"
    elif q_type in NA_QUESTION_TYPES:
        question = f"{raw_question}\n{SFT_TYPE_TEMPLATE['na']}"
    else:
        raise ValueError(f"Unknown question type: {q_type}")

    text_content = {"type": "text", "text": question}
    video_content = {"type": "video"}

    if (video_dir / item["dataset"] / (item["scene_name"] + ".mp4")).exists():  # mp4 video file
        video_path = (video_dir / item["dataset"] / (item["scene_name"] + ".mp4")).resolve()
        video_content["video"] = str(video_path)
        video_content["nframes"] = video_nframes if video_nframes > 0 else None
    elif (video_dir / item["dataset"] / item["scene_name"]).exists():  # folder of frames
        frame_folder = (video_dir / item["dataset"] / item["scene_name"]).resolve()
        video_path = sorted(glob.glob(str(frame_folder / "*.png")))
        assert (
            len(video_path) == video_nframes
        ), f"Number of frames in {frame_folder} ({len(video_path)}) does not match expected {video_nframes}."
        video_content["video"] = video_path
    else:
        raise FileNotFoundError(
            f"Data file not found for video_dir" f"{video_dir}, dataset {item['dataset']}, scene {item['scene_name']}"
        )

    return {
        "role": "user",
        "content": [video_content, text_content],
    }


def prepare_chat_batch(
    batch_data: List[Dict],
    processor: Any,
    model_type: str,
    video_dir: Path,
    video_nframes: int,
) -> Tuple[Dict, List[str]]:
    """Prepare batch for inference: build prompts, process video, and tokenize."""
    batch_messages = [[build_user_message(item, video_dir, video_nframes)] for item in batch_data]

    prompts_text = [
        processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True) for example in batch_messages
    ]
    prompts_text_copy = prompts_text.copy()

    video_inputs = []
    image_inputs = []
    for example in batch_messages:
        images, videos = process_vision_info(example)
        if images:
            image_inputs.extend(images)
        elif videos:
            video_inputs.extend(videos)
        else:
            raise ValueError("Each example must contain either images or videos.")

    batch = processor(
        text=prompts_text,
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )

    if "spatial-mllm" in model_type:
        batch = prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs)

    return batch, prompts_text_copy


def inference_batch(batch_inputs: Dict, model: Any, processor: Any) -> List[str]:
    """Run inference on the batch inputs."""
    batch_inputs.to(model.device)
    if "image_tchw" in batch_inputs and batch_inputs["image_tchw"] is not None:
        batch_inputs["image_tchw"] = [image_tchw_i.to(model.device) for image_tchw_i in batch_inputs["image_tchw"]]
    if "video_tchw" in batch_inputs and batch_inputs["video_tchw"] is not None:
        batch_inputs["video_tchw"] = [video_tchw_i.to(model.device) for video_tchw_i in batch_inputs["video_tchw"]]

    generation_kwargs = dict(
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_p=0.001,
        use_cache=True,
    )

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**batch_inputs,**generation_kwargs)

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch_inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text


def postprocess_batch(
    batch_data: List[Dict], batch_output_text: List[str], prompts_text: List[str]
) -> List[Dict]:
    """Post-process outputs: clean text, calculate rewards, and structure results."""
    batch_results = []
    for sample, model_output, prompt in zip(batch_data, batch_output_text, prompts_text):
        clean_ans = clean_text(model_output)
        clean_ans_gt = clean_text(sample.get("ground_truth", ""))
        reward = vsi_reward(clean_ans_gt, clean_ans, sample["question_type"])

        batch_results.append(
            {
                "sample": sample,
                "prompt": prompt,
                "model_output": model_output,
                "cleaned_model_output": clean_ans,
                "cleaned_gt_answer": clean_ans_gt,
                "reward": reward,
                "correct": reward == 1.0,
            }
        )

    return batch_results


def calculate_metrics(results):
    """Calculate detailed metrics (per-type scores/counts, micro/macro)."""
    if not results:
        return {
            "per_question_type": {},
            "acc": {"micro": 0.0, "macro": 0.0},
            "mra": {"micro": 0.0, "macro": 0.0},
            "all": {"micro": 0.0, "macro": 0.0},
            "prune_ratio": {"mean": None},
        }

    df = pd.DataFrame(
        [
            {
                "reward": res.get("reward", 0.0),
                "question_type": res["sample"].get("question_type"),
            }
            for res in results
        ]
    )
    df["is_na"] = df["question_type"].isin(NA_QUESTION_TYPES)

    def safe_mean(series):
        return float(series.mean()) if len(series) else 0.0

    # Per-question-type scores and counts
    per_qtype = {
        qtype: {"score": float(group["reward"].mean()), "count": int(len(group))}
        for qtype, group in df.groupby("question_type")
    }

    # Micro scores
    acc_mask = ~df["is_na"]
    mra_mask = df["is_na"]
    micro_acc = safe_mean(df.loc[acc_mask, "reward"])
    micro_mra = safe_mean(df.loc[mra_mask, "reward"])
    micro_all = safe_mean(df["reward"])

    # Macro scores (average of per-type scores)
    acc_qtypes = [q for q in per_qtype if q not in NA_QUESTION_TYPES]
    mra_qtypes = [q for q in per_qtype if q in NA_QUESTION_TYPES]

    macro_acc = safe_mean(pd.Series([per_qtype[q]["score"] for q in acc_qtypes]))
    macro_mra = safe_mean(pd.Series([per_qtype[q]["score"] for q in mra_qtypes]))
    macro_all = safe_mean(pd.Series([v["score"] for v in per_qtype.values()]))

    return {
        "per_question_type": per_qtype,
        "acc": {"micro": micro_acc, "macro": macro_acc},
        "mra": {"micro": micro_mra, "macro": macro_mra},
        "all": {"micro": micro_all, "macro": macro_all},
    }


def evaluate_vsibench(vsi_data, model_type, model_path, batch_size, video_dir, output_path, video_nframes):
    """Evaluate model on a specific dataset. Forces batch size to 1."""

    setup_logging()
    model, processor = load_model_and_processor(model_type, model_path)
    final_output = []

    for i in tqdm(range(0, len(vsi_data), batch_size), desc="Evaluating VSIBench"):
        batch_data = vsi_data[i : i + batch_size]
        batch_llm_inputs, prompts_text = prepare_chat_batch(batch_data, processor, model_type, video_dir, video_nframes)
        batch_output_text = inference_batch(batch_llm_inputs, model, processor)
        batch_results = postprocess_batch(batch_data, batch_output_text, prompts_text)
        final_output.extend(batch_results)

        # Checkpoint partial results every 10 batches or at the end
        if (i + 1) % 10 == 0 or (i + 1) == len(vsi_data):
            save_json(output_path, final_output)

    return final_output


def run_worker(gpu_id, vsi_data, model_type, model_path, batch_size, video_dir, output_path, video_nframes):
    """Worker function to run evaluation on a specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    evaluate_vsibench(vsi_data, model_type, model_path, batch_size, video_dir, output_path, video_nframes)


def main(args):
    setup_logging()

    # Set start method to spawn for CUDA compatibility
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    output_dir = Path(args.output_dir).resolve() / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_dir = Path(args.annotation_dir).resolve()
    if args.video_dir:
        video_dir = Path(args.video_dir).resolve()
    else:
        video_dir = annotation_dir

    vsi_data = load_dataset(str(annotation_dir), "full")["test"]
    n_gpu = torch.cuda.device_count()
    if n_gpu <= 0:
        raise RuntimeError("VSIBench evaluation requires at least one CUDA device.")

    print(f"Starting evaluation on {n_gpu} GPUs...")

    # Parse CUDA_VISIBLE_DEVICES to handle specific GPU selection
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        gpu_ids = [x.strip() for x in cuda_visible_devices.split(",") if x.strip()]
    else:
        gpu_ids = [str(i) for i in range(n_gpu)]

    processes = []
    output_paths = []

    for idx, data_chunk in enumerate(chunk_dataset(vsi_data, n_gpu)):
        output_path_gpu = output_dir / f"results_{args.model_type}_{idx}.json"
        output_paths.append(output_path_gpu)

        # Select GPU ID
        gpu_id = gpu_ids[idx] if idx < len(gpu_ids) else str(idx)

        p = mp.Process(
            target=run_worker,
            args=(
                gpu_id,
                data_chunk,
                args.model_type,
                args.model_path,
                args.batch_size,
                video_dir,
                output_path_gpu,
                args.nframes,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    final_output = []
    for path in output_paths:
        if path.exists():
            with open(path, "r") as f:
                final_output.extend(json.load(f))
        else:
            print(f"Warning: Output file {path} not found.")

    # Compute the overall metrics across shards.
    final_acc_dict = calculate_metrics(final_output)
    save_json(
        output_dir / f"results_{args.model_type}.json",
        final_output,
    )
    save_json(
        output_dir / f"metrics_{args.model_type}.json",
        final_acc_dict,
    )
    print(f"Finished evaluation for vsibench.")
    print(f"Final Metrics: {final_acc_dict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on VSIBench dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--model_type", type=str, default="spatial-mllm", help="Type of the model.")
    parser.add_argument("--nframes", type=int, default=16, help="Number of frames to sample from each video.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation (forced to 1).")
    parser.add_argument(
        "--annotation_dir", type=str, required=True, help="Directory containing the VSIBench data files."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory containing the video frame files, if none, use annotation_dir.",
    )
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation results.")
    parser.add_argument(
        "--output_name", type=str, default="eval_vsibench", help="Directory to save evaluation results."
    )
    args = parser.parse_args()

    main(args)
