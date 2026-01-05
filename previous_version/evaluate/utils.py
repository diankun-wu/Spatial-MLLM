import re
import torch
from typing import Optional, Dict, Any
from Levenshtein import ratio


def extract_think(output_str: str) -> str:
    """Extract the thinking process from model output."""
    pattern = r"<think>\s*(.*?)\s*</think>"
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_answer(text: str) -> str:
    """Extract the answer from model output."""
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def clean_text(text, exclue_chars=["\n", "\r"]):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]

    for char in exclue_chars:
        if char in ["\n", "\r"]:
            # If there is a space before the newline, remove the newline
            text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
        else:
            text = text.replace(char, " ")

    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip(".").lower()

def normalize_number(num_str: str) -> Optional[float]:
    """Convert string number to float, handling commas."""
    try:
        num_str = num_str.replace(",", "")
        return float(num_str)
    except Exception:
        return None


def mean_relative_accuracy(
    pred: float,
    target: float,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    """Calculate mean relative accuracy for regression tasks."""
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)

    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)

    thresholds = torch.arange(start, end + interval / 2, interval, dtype=torch.float32)
    conditions = rel_error < (1 - thresholds)
    mra = conditions.float().mean()
    return mra.item()


def vsi_reward(clean_ans_gt: str, clean_ans_pred: str, question_type: str) -> float:
    """Calculate reward based on question type and model output."""
    if question_type == "multiple choice":
        return 1.0 if clean_ans_pred.strip() == clean_ans_gt.strip() else 0.0
    elif question_type == "regression" or question_type == "numerical":
        gt_number = normalize_number(clean_ans_gt)
        pred_number = normalize_number(clean_ans_pred)
        if gt_number is None or pred_number is None:
            return 0.0
        return mean_relative_accuracy(pred_number, gt_number)
    else:
        raise ValueError(f"Unsupported question type: {question_type}")
