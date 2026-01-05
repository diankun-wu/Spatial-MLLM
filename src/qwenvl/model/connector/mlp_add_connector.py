from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


class MLPAddConnector(nn.Module):
    def __init__(
        self, vggt_dim, language_dim, spatial_embeds_layer_idx, visual_temporal_merge_size, visual_spatial_merge_size
    ) -> None:
        super().__init__()
        self.vggt_dim = vggt_dim
        self.language_dim = language_dim

        self.spatial_embeds_layer_idx = spatial_embeds_layer_idx
        print(f"Using spatial_embeds_layer_idx: {self.spatial_embeds_layer_idx}")

        self.visual_temporal_merge_size = visual_temporal_merge_size
        self.visual_spatial_merge_size = visual_spatial_merge_size

        self.merged_dim = (self.vggt_dim * 2) * self.visual_temporal_merge_size * self.visual_spatial_merge_size**2
        self.ln_q = Qwen2RMSNorm(self.merged_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.merged_dim, self.merged_dim),
            nn.GELU(),
            nn.Linear(self.merged_dim, self.language_dim),
        )

    def preprocess_spatial_embeds(
        self,
        spatial_embeds_list: List[List[torch.Tensor]],
        patch_start_idx: List[int],
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        all_spatial_embeds = []
        grid_idx = 0

        for i, spatial_embeds_item in enumerate(spatial_embeds_list):

            # spatial_embeds_list: List[List[Float[Tensor, "S+5,P,2D"]]]
            spatial_embeds = spatial_embeds_item[self.spatial_embeds_layer_idx].unsqueeze(0)
            # spatial_embeds: Float[Tensor, "B,S,P,2D"]
            spatial_embeds = spatial_embeds[:, :, patch_start_idx[i]:]

            B, S, P, DD = spatial_embeds.shape
            assert B == 1, "batch size should be 1"

            # Find corresponding grid_thw rows
            accumulated_t = 0

            if grid_idx >= len(grid_thw):
                raise ValueError(f"Not enough grid_thw rows for spatial_embeds {i}")

            while accumulated_t * self.visual_temporal_merge_size < S:
                if grid_idx >= len(grid_thw):
                    raise ValueError(
                        f"Not enough grid_thw rows for spatial_embeds {i}. Accumulated T={accumulated_t}, Target S={S}"
                    )

                t, h, w = grid_thw[grid_idx].tolist()

                if accumulated_t == 0:
                    npatch_h, npatch_w = h, w
                else:
                    assert h == npatch_h and w == npatch_w, f"Spatial dimensions mismatch within video {i}"

                accumulated_t += t
                grid_idx += 1

            npatch_t = accumulated_t

            assert P == npatch_h * npatch_w, "patch number mismatch"
            assert npatch_t == S // self.visual_temporal_merge_size, "temporal patch number mismatch"

            # reshape spatial embeddings to 2D grid
            spatial_embeds = (
                spatial_embeds.view(B, S, npatch_h, npatch_w, DD).permute(0, 1, 4, 2, 3).contiguous()
            )  # [B, S, DD, np_h, np_w]

            spatial_embeds = (
                spatial_embeds.view(
                    B,
                    npatch_t,
                    self.visual_temporal_merge_size,
                    DD,
                    npatch_h // self.visual_spatial_merge_size,
                    self.visual_spatial_merge_size,
                    npatch_w // self.visual_spatial_merge_size,
                    self.visual_spatial_merge_size,
                )
                .permute(0, 1, 4, 6, 5, 7, 3, 2)
                .contiguous()
            )

            spatial_embeds = spatial_embeds.reshape(
                B * npatch_t * npatch_h * npatch_w, DD * self.visual_temporal_merge_size
            )
            spatial_embeds = spatial_embeds.view(
                -1, DD * self.visual_temporal_merge_size * self.visual_spatial_merge_size**2
            )
            all_spatial_embeds.append(spatial_embeds)

        all_spatial_embeds_concated = torch.cat(all_spatial_embeds, dim=0)
        return all_spatial_embeds_concated

    def forward(
        self,
        image_embeds: Optional[torch.Tensor] = None,
        video_embeds: Optional[torch.Tensor] = None,
        spatial_embeds_list: List[List[torch.Tensor]] = None,
        patch_start_idx: List[int] = None,
        grid_thw: torch.Tensor = None,
    ) -> torch.Tensor:
        assert video_embeds is not None or image_embeds is not None, "Either video_embeds or image_embeds must be provided."

        # FIXME: only handle video_embeds input
        # merge spatial and temporal tokens
        spatial_embeds = self.preprocess_spatial_embeds(spatial_embeds_list,  patch_start_idx, grid_thw)

        # LayerNorm and project to language space
        spatial_embeds = self.ln_q(spatial_embeds)
        spatial_embeds = self.mlp(spatial_embeds)

        if image_embeds is not None:
            return image_embeds + spatial_embeds
        else:
            return video_embeds + spatial_embeds

    def print_trainable_parameters(self) -> None:
        """
        Prints the trainable status of all connector components
        """
        # Check trainable status of merger module
        is_connector_trainable = any(param.requires_grad for param in self.parameters())

        # Print results
        print(f"MLPAddConnector Trainable: {is_connector_trainable}")