import os
import re
from pathlib import Path


def find_repo_path():
    """
    Find the path to a repository based on .git directory.
    """
    file_path = Path(__file__).resolve()
    while file_path != file_path.parent:
        if (file_path / ".git").exists():
            return file_path
        file_path = file_path.parent
    raise FileNotFoundError("No .git directory found in the path hierarchy.")


REPO_PATH = find_repo_path()

### Spatial-MLLM-Mix Data
SPATIAL_MLLM_MIX_133K = {
    "annotation_path": os.path.join(REPO_PATH, "datasets/annotations/spatial-mllm-mix-133k.jsonl"),
    "data_path": os.path.join(REPO_PATH, "datasets/visuals"),
}

SPATIAL_MLLM_MIX_203K = {
    "annotation_path": os.path.join(REPO_PATH, "datasets/annotations/spatial-mllm-mix-203k.jsonl"),
    "data_path": os.path.join(REPO_PATH, "datasets/visuals"),
}

# Route Plan Data From VLM-3R
ROUTE_PLAN_SCANNET_2K = {
    "annotation_path": os.path.join(
        REPO_PATH, "datasets/annotations/routeplan-2k.jsonl"
    ),
    "data_path": os.path.join(REPO_PATH, "datasets/visuals")
}

ROUTE_PLAN_4K = {
    "annotation_path": os.path.join(
        REPO_PATH, "datasets/annotations/routeplan-4k.jsonl"
    ),
    "data_path": os.path.join(REPO_PATH, "datasets/visuals")
}

### VSI-590K Data From Cambrian-S
VSI_590K = {
    "annotation_path": os.path.join(REPO_PATH, "datasets/annotations/vsi-590k-processed.jsonl"),
    "data_path": os.path.join(REPO_PATH, "datasets/visuals"),
}

### MindCube Data
MINDCUBE_21K = {
    "annotation_path": os.path.join(REPO_PATH, "datasets/annotations/mindcube-processed.jsonl"),
    "data_path": os.path.join(REPO_PATH, "datasets/visuals"),
}

data_dict = {
    "spatial_mllm_mix_133k": SPATIAL_MLLM_MIX_133K,
    "spatial_mllm_mix_203k": SPATIAL_MLLM_MIX_203K,
    "route_plan_scannet_2k": ROUTE_PLAN_SCANNET_2K,
    "route_plan_4k": ROUTE_PLAN_4K,
    "vsi_590k": VSI_590K,
    "mindcube_21k": MINDCUBE_21K,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list
