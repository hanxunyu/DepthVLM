
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDepthHead(nn.Module):
    """
    Two-layer MLP Depth Head for dense depth prediction (Ablation of DPT).

    For each image token, directly predicts the depth values of its corresponding
    patch region via a two-layer MLP. No multi-scale feature fusion is involved;
    serves as an ablation baseline to DPT's multi-scale fusion.

    Args:
        dim_in (int): Input feature dimension (e.g., 2560 for Qwen3-VL-4B).
        hidden (int): MLP hidden dimension. Default 1024.
        effective_patch (int): Pixel patch size corresponding to each image token,
            equals vit_patch_size * spatial_merge_size (16*2=32 for Qwen3-VL).
        feature_mode (str): How to use one of the 4 input feature layers:
            - 'llm_last' : Use only the LLM last layer (default, simplest baseline).
            - 'concat'   : Concatenate the 4 layer features (dim_in*4 -> hidden).
            - 'sum'      : Sum the 4 layers.
    """

    def __init__(
        self,
        dim_in: int,
        hidden: int = 1024,
        effective_patch: int = 32,
        feature_mode: str = "llm_last",
    ) -> None:
        super().__init__()

        assert feature_mode in ("llm_last", "concat", "sum"), (
            f"feature_mode must be 'llm_last'/'concat'/'sum', got {feature_mode}"
        )
        self.feature_mode = feature_mode
        self.effective_patch = effective_patch
        self.patch_pixels = effective_patch * effective_patch  # e.g., 32*32=1024

        # Determine MLP input dimension based on feature_mode
        if feature_mode == "concat":
            mlp_in = dim_in * 4
        else:
            mlp_in = dim_in

        # LayerNorm for training stability (consistent with DPT head)
        self.norm = nn.LayerNorm(mlp_in)

        # ===== Two-layer MLP =====
        # Linear(mlp_in -> hidden) -> GELU -> Linear(hidden -> patch_pixels)
        # Each token independently outputs effective_patch^2 depth values
        # (i.e., the dense depth of that patch region).
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.patch_pixels),
        )

        # Ensure depth > 0
        self.act = nn.Softplus()

    def forward(
        self,
        features_list: List[torch.Tensor],
        patch_h: int,
        patch_w: int,
        output_size: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            features_list: List of 4 feature tensors, each of shape (B, N, C).
                Order (consistent with DPTDepthHead):
                    [deepstack_layer5, deepstack_layer11, deepstack_layer17, llm_last_layer]
            patch_h: Token grid height (e.g., 45).
            patch_w: Token grid width (e.g., 60).
            output_size: Target resolution (H_out, W_out). If None, defaults to
                (patch_h*p, patch_w*p), where p = effective_patch.

        Returns:
            depth: (B, H_out, W_out), > 0 after Softplus.

        Data flow (with patch_h=45, patch_w=60, effective_patch=32, dim_in=2560,
                   feature_mode='llm_last'):
            Input: 4 x (B, 2700, 2560)
            -> select llm_last
            (B, 2700, 2560)
            -> LayerNorm + Linear(2560->1024) + GELU + Linear(1024->1024)
            (B, 2700, 1024)     # 1024 = 32*32
            -> reshape: (B, patch_h, patch_w, p, p) -> (B, patch_h*p, patch_w*p)
            (B, 1440, 1920)
            -> bilinear interpolate if output_size differs
            (B, H_out, W_out)
            -> Softplus
            depth > 0
        """
        assert len(features_list) == 4, f"Expected 4 feature maps, got {len(features_list)}"

        # ===== Step 1: Select/fuse features based on feature_mode =====
        if self.feature_mode == "llm_last":
            x = features_list[3]                         # (B, N, dim_in)
        elif self.feature_mode == "concat":
            x = torch.cat(features_list, dim=-1)         # (B, N, dim_in*4)
        else:  # 'sum'
            x = sum(features_list)                       # (B, N, dim_in)

        B, N, _ = x.shape
        assert N == patch_h * patch_w, (
            f"Token count {N} != patch_h*patch_w ({patch_h}*{patch_w}={patch_h * patch_w})"
        )

        # ===== Step 2: Two-layer MLP =====
        x = self.norm(x)
        x = self.mlp(x)                                  # (B, N, p*p)

        # ===== Step 3: Reshape to dense depth map =====
        p = self.effective_patch
        # (B, patch_h, patch_w, p, p) -> (B, patch_h, p, patch_w, p) -> (B, patch_h*p, patch_w*p)
        x = x.view(B, patch_h, patch_w, p, p)
        x = x.permute(0, 1, 3, 2, 4).contiguous()        # (B, patch_h, p, patch_w, p)
        depth = x.view(B, patch_h * p, patch_w * p)      # (B, H0, W0)

        # ===== Step 4: Bilinear interpolate if output resolution differs =====
        if output_size is None:
            output_size = (patch_h * p, patch_w * p)

        if depth.shape[-2:] != output_size:
            depth = F.interpolate(
                depth.unsqueeze(1), size=output_size,
                mode="bilinear", align_corners=True,
            ).squeeze(1)

        # ===== Step 5: Softplus to ensure > 0 =====
        depth = self.act(depth)
        return depth
