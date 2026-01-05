import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

import shutil
import tempfile
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch import cuda
from torchvision import transforms as TF

try:
    from decord import VideoReader, cpu  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    VideoReader = None
    cpu = None
    
from src.qwenvl.external.vggt.models.vggt import VGGT  # type: ignore
from src.qwenvl.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore


def compute_voxel_sets(world_points, world_points_conf_mask, x_min, y_min, z_min, voxel_size):
    """
    Compute the voxel set covered by each frame.

    Args:
        world_points: Tensor of shape (1, T, H, W, 3) containing 3D coordinates.
        world_points_conf_mask: Boolean tensor of shape (1, T, H, W) indicating valid points.
        x_min, y_min, z_min: Minimum scene coordinates.
        voxel_size: Size of each voxel.

    Returns:
        List[Set[Tuple[int]]]: Voxel coordinate set for each frame.
    """
    device = world_points.device
    T = world_points.shape[1]
    voxel_sets = []
    
    # Ensure coordinate parameters are tensors on the correct device
    x_min = torch.tensor(x_min, device=device) if not isinstance(x_min, torch.Tensor) else x_min.to(device)
    y_min = torch.tensor(y_min, device=device) if not isinstance(y_min, torch.Tensor) else y_min.to(device)
    z_min = torch.tensor(z_min, device=device) if not isinstance(z_min, torch.Tensor) else z_min.to(device)
    voxel_size = torch.tensor(voxel_size, device=device) if not isinstance(voxel_size, torch.Tensor) else voxel_size.to(device)

    for t in range(T):
        # Retrieve valid points for the current frame
        mask = world_points_conf_mask[0, t].flatten()  # (H*W,)
        points = world_points[0, t].reshape(-1, 3)    # (H*W, 3)
        valid_points = points[mask]  # (N_valid, 3)
        
        if valid_points.size(0) == 0:
            voxel_sets.append(set())
            continue
            
        # Compute voxel coordinates
        offset = torch.tensor([x_min, y_min, z_min], device=device)
        voxel_coords = ((valid_points - offset) / voxel_size).floor().long()
        
        # Remove duplicates and convert to a CPU set
        unique_voxels = torch.unique(voxel_coords, dim=0)
        voxel_set = set(map(tuple, unique_voxels.cpu().numpy().tolist()))
        
        voxel_sets.append(voxel_set)
    
    return voxel_sets

def maximum_coverage_sampling(voxel_sets, K):
    """
    Greedy maximum-coverage sampling.

    Args:
        voxel_sets: List of voxel sets for each frame.
        K: Maximum number of frames to select.

    Returns:
        List[int]: Indices of selected frames.
    """
    selected = []
    covered = set()
    remaining_frames = set(range(len(voxel_sets)))
    
    for _ in range(K):
        if not remaining_frames:
            break
            
        max_gain = -1
        best_frame = None
        
        # Find the frame with the maximum marginal gain
        for frame in remaining_frames:
            gain = len(voxel_sets[frame] - covered)
            if gain > max_gain:
                max_gain = gain
                best_frame = frame
                
        if best_frame is None or max_gain <= 0:
            break  # No more new coverage
            
        selected.append(best_frame)
        covered.update(voxel_sets[best_frame])
        remaining_frames.remove(best_frame)
    
    return selected

def space_aware_frame_sampling(vggt, images, K, dtype):
    """
    Perform space-aware frame sampling on a video tensor.

    Args:
        vggt: Pretrained VGGT model.
        images: Tensor of shape (T, 3, H, W) representing the video frames.
        K: Number of frames to sample.
    Returns:
        List[int]: Indices of selected frames.
    """
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = vggt(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print(predictions['world_points'].shape)

    world_points = predictions['world_points']  # shape (1, T, H, W, 3)
    world_points_flat = world_points.reshape(-1, 3)  # shape (B, 3)
    world_points_conf = predictions['world_points_conf']  # shape (1, T, H, W)
    world_points_conf_flat = world_points_conf.reshape(-1)  # shape (B)

    init_threshold_val = np.percentile(world_points_conf_flat.cpu().numpy(), 50)
    world_points_conf_mask = (world_points_conf >= init_threshold_val) & (world_points_conf > 0.1)
    world_points_conf_flat_mask = (world_points_conf_flat >= init_threshold_val) & (world_points_conf_flat > 0.1)

    # get bounding box of world_points
    x_min, y_min, z_min = world_points_flat[world_points_conf_flat_mask].min(dim=0)[0]
    x_max, y_max, z_max = world_points_flat[world_points_conf_flat_mask].max(dim=0)[0]
    print(x_min, y_min, z_min, x_max, y_max, z_max)

    voxel_size = min(x_max - x_min, y_max - y_min, z_max - z_min) / 20
    print(voxel_size)

    voxel_sets = compute_voxel_sets(
        world_points=world_points,
        world_points_conf_mask=world_points_conf_mask,
        x_min=x_min.item(),
        y_min=y_min.item(),
        z_min=z_min.item(),
        voxel_size=voxel_size.item()
    )

    selected_frames = sorted(maximum_coverage_sampling(voxel_sets, K))

    return selected_frames


def load_and_preprocess_images(image_path_list):
    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    for image_path in image_path_list:
        img = Image.open(image_path)

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")

        width, height = img.size
        if height > width:
            img = img.rotate(-90, expand=True)

        width, height = img.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    if len(shapes) > 1:
        print(f"Warning: Images have varying shapes {shapes}, padding to the largest size.")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)

    if len(image_path_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)

    return images


def process_videos_on_device(device_id, video_paths, args):
    if not video_paths:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    device = "cuda"
    dtype = torch.bfloat16
    model = VGGT.from_pretrained(args.model_path).to(device)

    for video_path in tqdm(video_paths, desc=f"GPU {device_id} processing videos"):
        tmp_dir = Path(tempfile.mkdtemp(prefix="sw_sampling_frames_"))
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)

        if num_frames == 0:
            raise ValueError(f"No frames found in video: {video_path}")

        sample_count = min(128, num_frames)
        frame_indices = np.linspace(0, num_frames - 1, num=sample_count, dtype=int)

        for i, frame_idx in enumerate(frame_indices):
            frame = vr[frame_idx].asnumpy()
            image = Image.fromarray(frame)
            frame_path = tmp_dir / f"frame_{frame_idx:04d}.png"
            image.save(frame_path)

        print(f"[GPU {device_id}] Saved {len(frame_indices)} frames to {tmp_dir}")

        frame_image_paths = sorted(glob(str(tmp_dir / "*.png")))
        images = load_and_preprocess_images(frame_image_paths).to(device, dtype=dtype)
        K = min(args.num_frames, images.shape[0])
        selected_frames = space_aware_frame_sampling(model, images, K, dtype)
        print(f"[GPU {device_id}] Selected frames: {selected_frames}")

        selected_original_indices = [int(frame_indices[idx]) for idx in selected_frames]

        video_name = Path(video_path).stem

        sa_dir = os.path.join(args.output_folder, "sa_sampling", video_name)
        os.makedirs(sa_dir, exist_ok=True)
        for orig_idx in selected_original_indices:
            src_path = tmp_dir / f"frame_{orig_idx:04d}.png"
            if not src_path.exists():
                continue
            dst_name = f"{video_name}_frame_{orig_idx:06d}.png"
            dst_path = os.path.join(sa_dir, dst_name)
            shutil.copy2(src_path, dst_path)

        print(f"[GPU {device_id}] Saved {len(selected_original_indices)} selected frames to {sa_dir}")

        uniform_dir = os.path.join(args.output_folder, "uniform_sampling", video_name)
        os.makedirs(uniform_dir, exist_ok=True)
        if len(frame_indices) <= args.num_frames:
            sampled_indices = frame_indices
        else:
            sampled_indices = np.linspace(0, len(frame_indices) - 1, num=args.num_frames, dtype=int)
            sampled_indices = frame_indices[sampled_indices]
        for orig_idx in sampled_indices:
            src_path = tmp_dir / f"frame_{orig_idx:04d}.png"
            if not src_path.exists():
                continue
            dst_name = f"{video_name}_frame_{orig_idx:06d}.png"
            dst_path = os.path.join(uniform_dir, dst_name)
            shutil.copy2(src_path, dst_path)

        print(f"[GPU {device_id}] Saved {len(sampled_indices)} uniform sampled frames to {uniform_dir}")

        shutil.rmtree(tmp_dir)
        cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Space-Aware Frame Sampling")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the input video folder.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained VGGT model.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder for selected frames.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to sample using space-aware sampling.")
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()

    # Parse CUDA_VISIBLE_DEVICES to handle specific GPU selection
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        gpu_ids = [x.strip() for x in cuda_visible_devices.split(",") if x.strip()]
    else:
        gpu_ids = [str(i) for i in range(n_gpu)]


    all_videos = sorted(glob(os.path.join(args.video_folder, "*.mp4")))
    if not all_videos:
        print("No videos found to process.")
        sys.exit(0)

    num_gpus = min(len(gpu_ids), len(all_videos))
    video_splits = [list(split) for split in np.array_split(all_videos, num_gpus) if len(split) > 0]

    processes = []
    for idx, video_subset in enumerate(video_splits):
        device_id = gpu_ids[idx]
        process = mp.Process(target=process_videos_on_device, args=(device_id, video_subset, args))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
