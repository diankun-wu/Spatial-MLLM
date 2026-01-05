from collections import defaultdict
from typing import List

import deepspeed
import torch
from safetensors.torch import load_file
from transformers import PretrainedConfig, PreTrainedModel
from transformers.integrations import is_deepspeed_zero3_enabled

from src.qwenvl.external.vggt.models.vggt import VGGT


class VGGTSpatialEncoderConfig(PretrainedConfig):
    model_type = "vggt_spatial_encoder"
    base_config_key = "spatial_config"

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim


class VGGTSpatialEncoderPreTrainedModel(PreTrainedModel):
    config_class = VGGTSpatialEncoderConfig
    base_model_prefix = "spatial_encoder"

    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vggt_model = VGGT(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
        ).eval()

    def _init_weights(self, module):
        pass

    def load_pretrained_weights(self, pretrained_weight: str):
        if is_deepspeed_zero3_enabled():
            self.load_pretrained_weights_zero3(pretrained_weight)
        else:
            self._load_pretrained_weights(pretrained_weight)
            
    def load_pretrained_weights_zero3(self, pretrained_weight):
        with deepspeed.zero.GatheredParameters(list(self.vggt_model.parameters()), modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                self._load_pretrained_weights(pretrained_weight)
                
    def _load_pretrained_weights(self, pretrained_weight):
        print(f"Loading external VGGT weights from: {pretrained_weight}")
        vggt_state_dict = load_file(pretrained_weight, device="cpu")
        missing_keys, unexpected_keys = self.vggt_model.load_state_dict(vggt_state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys when loading VGGT state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading VGGT state dict: {unexpected_keys}")

    def preprocess_video_tensors(self, video_tensor: List[torch.Tensor]) -> List[torch.Tensor]:
        return video_tensor

    def forward(self, video_tensor: List[torch.Tensor], **kwargs):
        """
        video_tensor: List of [T_i, C, H_i, W_i].
        """
        group_map = defaultdict(list)
        for original_idx, v in enumerate(video_tensor):
            group_map[v.shape].append((original_idx, v))

        final_outputs = [None] * len(video_tensor)
        final_indices = [None] * len(video_tensor)

        for (T, C, H, W), items in group_map.items():
            indices = [item[0] for item in items]
            tensors = [item[1] for item in items]
        
            batch_input = torch.stack(tensors) 
            batch_out, batch_start_idx = self.vggt_model.aggregator(batch_input)            
            B_curr = len(indices)
            
            for i in range(B_curr):
                real_idx = indices[i]
                sample_output = []
                for layer_tensor in batch_out:
                    sample_output.append(layer_tensor[i])
                
                final_outputs[real_idx] = sample_output
                final_indices[real_idx] = batch_start_idx
        
        return final_outputs, final_indices
    
    def print_trainable_parameters(self) -> None:
        """
        Prints the trainable status of all spatial encoder components
        """
        # Check trainable status of merger module
        is_spatial_encoder_trainable = any(param.requires_grad for param in self.parameters())

        # Print results
        print(f"Spatial Encoder Trainable: {is_spatial_encoder_trainable}")