
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPTDepthHead(nn.Module):
    """
    DPT Head for dense depth prediction, adapted for Qwen3-VL.

    Takes 4 feature maps (from ViT DeepStack layers + LLM last layer),
    reshapes them to 2D, constructs a multi-scale pyramid, fuses features
    bottom-up, and outputs a per-token depth map.

    Args:
        dim_in (int): Input feature dimension (e.g., 2560 for Qwen3-VL-4B).
        features (int): Intermediate feature channels for fusion. Default: 256.
        out_channels (List[int]): Output channels after projection for each layer.
            Default: [256, 512, 1024, 1024].
    """

    def __init__(
        self,
        dim_in: int,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
    ) -> None:
        """
        Args:
            dim_in: Input feature dimension. All 4 layer features share the same dim.
                    For Qwen3-VL-4B, ViT DeepStack merger output dim = LLM hidden_size = 2560.
            features: Unified channel count for the RefineNet fusion stage (all layer
                      features are first unified to this channel count before additive
                      fusion). Standard 256 from the original DPT paper; too large
                      causes high memory cost (params ~ features^2), too small loses
                      fusion precision.
            out_channels: Channel counts after 1x1 Conv projection for each layer
                          [layer0, layer1, layer2, layer3]. Follows FPN design where
                          higher-resolution layers have fewer channels and lower-
                          resolution layers have more: [256, 512, 1024, 1024],
                          mimicking the 4-stage output channels of ResNet.
        """
        super().__init__()

        # LayerNorm: normalize input features, shape unchanged: (B, N, dim_in) -> (B, N, dim_in)
        self.norm = nn.LayerNorm(dim_in)

        # ===== Projection layers =====
        # Project each layer's features from dim_in (e.g., 2560) to out_channels[i]
        # Input:  (B, dim_in, patch_h, patch_w)
        # Output: (B, out_channels[i], patch_h, patch_w)
        # Layer 0: (B, 2560, H, W) -> (B, 256,  H, W)
        # Layer 1: (B, 2560, H, W) -> (B, 512,  H, W)
        # Layer 2: (B, 2560, H, W) -> (B, 1024, H, W)
        # Layer 3: (B, 2560, H, W) -> (B, 1024, H, W)
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
        )

        # ===== Resize layers: turn the 4 same-resolution token grids into a multi-scale pyramid =====
        # All 4 input layers have spatial size (patch_h, patch_w), i.e. the ViT
        # token grid size (e.g., 45x60). With patch_size=32, token grid = 1/32 of
        # the original image. Build a multi-scale pyramid via upsampling, with
        # each layer differing by a factor of 2:
        #   Layer 0 (shallow, 256ch):  ConvTranspose ^8x  -> (8H, 8W) = 1/4  of original (360x480)
        #   Layer 1 (mid,     512ch):  ConvTranspose ^4x  -> (4H, 4W) = 1/8  of original (180x240)
        #   Layer 2 (deep,    1024ch): ConvTranspose ^2x  -> (2H, 2W) = 1/16 of original (90x120)
        #   Layer 3 (LLM,     1024ch): Identity unchanged -> (H,  W)  = 1/32 of original (45x60)
        self.resize_layers = nn.ModuleList(
            [
                # Layer 0: ^8x upsample
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],   # 256
                    out_channels=out_channels[0],   # 256
                    kernel_size=8,
                    stride=8,
                    padding=0,
                ),
                # Layer 1: ^4x upsample
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],   # 512
                    out_channels=out_channels[1],   # 512
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                # Layer 2: ^2x upsample
                nn.ConvTranspose2d(
                    in_channels=out_channels[2],   # 1024
                    out_channels=out_channels[2],   # 1024
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                # Layer 3: identity, keep token grid resolution (= 1/32 of original)
                nn.Identity(),
            ]
        )

        # ===== Scratch layers: 3x3 Conv to unify all layers to `features` channels =====
        # The pyramid's 4 layers have channel counts [256, 512, 1024, 1024];
        # they need to be unified to features=256 for additive fusion.
        # Layer 0: (B, 256,  8H, 8W) -> (B, 256, 8H, 8W)   channels unchanged
        # Layer 1: (B, 512,  4H, 4W) -> (B, 256, 4H, 4W)   channels halved
        # Layer 2: (B, 1024, 2H, 2W) -> (B, 256, 2H, 2W)   channels reduced 4x
        # Layer 3: (B, 1024,  H,  W) -> (B, 256,  H,  W)   channels reduced 4x
        self.scratch = _make_scratch(out_channels, features)

        # ===== RefineNet fusion blocks: bottom-up progressive fusion =====
        # Fusion order: layer4 -> +layer3 -> +layer2 -> +layer1
        # Each step upsamples x2 to the resolution of the next (shallower) layer.
        self.scratch.refinenet1 = _make_fusion_block(features)       # final output stage, fuses layer1
        self.scratch.refinenet2 = _make_fusion_block(features)       # fuses layer2
        self.scratch.refinenet3 = _make_fusion_block(features)       # fuses layer3
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)  # starting stage, only layer4, no skip

        # ===== Output Head: bilinear interpolation to full resolution =====
        # RefineNet output: (B, 256, 16H, 16W) = 1/2 of original (720x960)
        # Use bilinear interpolation to full resolution (aligned with reference VGGT code).
        head_features_1 = features    # 256
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Softplus(),  # ensure depth > 0
        )

    def forward(
        self,
        features_list: List[torch.Tensor],
        patch_h: int,
        patch_w: int,
        output_size: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        DPT Depth Head forward pass.

        Args:
            features_list: List of 4 feature tensors, each of shape (B, N, C).
                Order: [deepstack_layer5, deepstack_layer11, deepstack_layer17, llm_last_layer]
            patch_h: Height of the token grid (e.g., 45).
            patch_w: Width of the token grid (e.g., 60).
            output_size: Target spatial size (H_out, W_out) of the output depth map.
                - None: full resolution (patch_h*32, patch_w*32), i.e. (1440, 1920).
                - (H, W): the specified resolution.

        Returns:
            depth: shape (B, H_out, W_out), depth values (meters), > 0 via Softplus.

        Data flow (example: patch_h=45, patch_w=60, dim_in=2560):
            Input: 4 x (B, 2700, 2560)
            -> LayerNorm + reshape
            4 x (B, 2560, 45, 60)
            -> 1x1 Conv projection
            Layer 0: (B, 256,  45, 60)  Layer 1: (B, 512,  45, 60)
            Layer 2: (B, 1024, 45, 60)  Layer 3: (B, 1024, 45, 60)
            -> Resize (build multi-scale pyramid)
            Layer 0: (B, 256,  360, 480)  -- ^8x = 1/4  of original
            Layer 1: (B, 512,  180, 240)  -- ^4x = 1/8  of original
            Layer 2: (B, 1024,  90, 120)  -- ^2x = 1/16 of original
            Layer 3: (B, 1024,  45,  60)  -- id  = 1/32 of original
            -> Scratch 3x3 Conv (unify to 256ch)
            4 x (B, 256, respective sizes)
            -> RefineNet bottom-up fusion (each step ^2 upsample, including refinenet1)
            (B, 256, 720, 960)  = 1/2 of original
            -> output_conv1: 256->128
            (B, 128, 720, 960)
            -> bilinear interpolate to output_size (1440, 1920)
            (B, 128, 1440, 1920)
            -> output_conv2: 128->32->1 + Softplus
            (B, 1, 1440, 1920) -> squeeze -> (B, 1440, 1920)
        """
        assert len(features_list) == 4, f"Expected 4 feature maps, got {len(features_list)}"

        out = []
        for idx, x in enumerate(features_list):
            B = x.shape[0]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, x.shape[-1], patch_h, patch_w)
            x = self.projects[idx](x)
            x = self.resize_layers[idx](x)
            out.append(x)

        # Bottom-up fusion (refinenet1 also performs ^2 upsample) + output_conv1
        fused = self._scratch_forward(out)
        # fused: (B, 128, 16*patch_h, 16*patch_w) = (B, 128, 720, 960) = 1/2 of original

        # Determine target output resolution
        if output_size is None:
            output_size = (patch_h * 32, patch_w * 32)

        # Bilinear interpolation to output_size (aligned with reference VGGT code)
        fused = F.interpolate(fused, size=output_size, mode="bilinear", align_corners=True)

        # output_conv2: 128->32->1 + Softplus
        depth = self.scratch.output_conv2(fused).squeeze(1)  # (B, H_out, W_out)

        return depth

    def _scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Bottom-up RefineNet feature fusion.

        Data flow (example: patch_h=45, patch_w=60, features=256):
            Input 4 pyramid layer features (different channel counts, to be
            unified to 256ch first):
                layer_1: (B, 256,  360, 480)  -- highest resolution (shallow ^8x = 1/4)
                layer_2: (B, 512,  180, 240)  -- ^4x = 1/8
                layer_3: (B, 1024,  90, 120)  -- ^2x = 1/16
                layer_4: (B, 1024,  45,  60)  -- lowest resolution (LLM id = 1/32)

            Scratch 3x3 Conv unifies to 256ch:
                layer_1_rn: (B, 256, 360, 480)
                layer_2_rn: (B, 256, 180, 240)
                layer_3_rn: (B, 256,  90, 120)
                layer_4_rn: (B, 256,  45,  60)

            RefineNet bottom-up fusion (each step ^2 upsample, including refinenet1):
                refinenet4(layer_4_rn)          -> (B, 256,  90, 120)  ^x2 to layer3 size
                refinenet3(out, layer_3_rn)     -> (B, 256, 180, 240)  ^x2 to layer2 size
                refinenet2(out, layer_2_rn)     -> (B, 256, 360, 480)  ^x2 to layer1 size
                refinenet1(out, layer_1_rn)     -> (B, 256, 720, 960)  ^x2 (default scale_factor=2)

            output_conv1: 256->128
                -> (B, 128, 720, 960)  = 1/2 of original

        Returns:
            (B, 128, 16*patch_h, 16*patch_w) = (B, 128, 720, 960) = 1/2 of original
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)  # (B, 256, 8H, 8W)  <- 256->256
        layer_2_rn = self.scratch.layer2_rn(layer_2)  # (B, 256, 4H, 4W)  <- 512->256
        layer_3_rn = self.scratch.layer3_rn(layer_3)  # (B, 256, 2H, 2W)  <- 1024->256
        layer_4_rn = self.scratch.layer4_rn(layer_4)  # (B, 256,  H,  W)  <- 1024->256

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        # -> (B, 256, 2H, 2W)
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])   # -> (B, 256, 4H, 4W)
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])   # -> (B, 256, 8H, 8W)
        out = self.scratch.refinenet1(out, layer_1_rn)                              # -> (B, 256, 16H, 16W) default ^x2

        out = self.scratch.output_conv1(out)  # (B, 128, 16H, 16W)
        return out


################################################################################
# Sub-modules
################################################################################


def _make_fusion_block(
    features: int, has_residual: bool = True
) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        has_residual=has_residual,
    )


def _make_scratch(
    in_shape: List[int], out_shape: int
) -> nn.Module:
    """
    Create the Scratch layer: 4 x 3x3 Conv that unify the pyramid layers from
    different channel counts to out_shape.

    Args:
        in_shape: List of channel counts for the 4 pyramid layers, e.g., [256, 512, 1024, 1024].
        out_shape: Unified output channel count, e.g., 256 (= the `features` argument).

    Per-layer channel transforms:
        layer1_rn: in_shape[0] -> out_shape  (256  -> 256, unchanged)
        layer2_rn: in_shape[1] -> out_shape  (512  -> 256, halved)
        layer3_rn: in_shape[2] -> out_shape  (1024 -> 256, reduced 4x)
        layer4_rn: in_shape[3] -> out_shape  (1024 -> 256, reduced 4x)

    Note: bias=False, since the subsequent ResidualConvUnit in RefineNet has its own bias.
    """
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )  # 256 -> 256
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )  # 512 -> 256
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )  # 1024 -> 256
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )  # 1024 -> 256
    return scratch


class ResidualConvUnit(nn.Module):
    """
    Residual conv unit: x -> ReLU -> Conv3x3 -> ReLU -> Conv3x3 -> + x (residual).

    Input/output shape preserved: (B, features, H, W) -> (B, features, H, W).
    Both 3x3 Convs preserve the channel count `features` (features -> features).
    """

    def __init__(self, features: int, activation: nn.Module):
        super().__init__()
        # Two 3x3 Convs, channels unchanged: features -> features
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, features, H, W) -> (B, features, H, W)"""
        out = self.activation(x)    # (B, features, H, W) -- ReLU
        out = self.conv1(out)       # (B, features, H, W) -- 3x3 Conv
        out = self.activation(out)  # (B, features, H, W) -- ReLU
        out = self.conv2(out)       # (B, features, H, W) -- 3x3 Conv
        return out + x              # residual: (B, features, H, W)


class FeatureFusionBlock(nn.Module):
    """
    Feature fusion block: add the (upsampled) deeper-layer features to the
    shallower-layer skip features, then refine.

    Data flow:
        If has_residual=True (refinenet1/2/3):
            xs[0]: features from the previous (deeper) fusion stage  (B, features, H_deep, W_deep)
            xs[1]: skip features at the current scale                (B, features, H, W)
            -> xs[1] is refined by ResidualConvUnit
            -> output = xs[0] + refined_xs[1]
            -> output is refined by ResidualConvUnit
            -> upsample to target size
            -> 1x1 Conv output

        If has_residual=False (refinenet4, the bottom of the pyramid, no skip):
            xs[0]: layer4 features  (B, features, H, W)
            -> refined by ResidualConvUnit
            -> upsample to target size
            -> 1x1 Conv output
    """

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        has_residual: bool = True,
    ):
        super().__init__()
        self.has_residual = has_residual
        # 1x1 Conv: features -> features, used for the final output
        self.out_conv = nn.Conv2d(
            features, features, kernel_size=1, stride=1, padding=0, bias=True
        )

        if has_residual:
            # Used to refine the skip-connection features
            self.resConfUnit1 = ResidualConvUnit(features, activation)
        # Used to refine the fused features
        self.resConfUnit2 = ResidualConvUnit(features, activation)

    def forward(self, *xs, size=None):
        """
        Args:
            xs[0]: features from the previous (deeper) fusion stage, shape (B, features, H1, W1).
            xs[1]: (optional, when has_residual=True) skip features at the current
                scale, shape (B, features, H2, W2).
            size: target spatial size (H_target, W_target). If None, use 2x upsampling.

        Returns:
            Fused and upsampled features: (B, features, H_target, W_target).
        """
        output = xs[0]  # (B, features, H1, W1) -- from the deeper layer

        if self.has_residual:
            # Refine the skip features and add to the deeper-layer features
            res = self.resConfUnit1(xs[1])  # (B, features, H2, W2)
            output = output + res           # element-wise add (sizes must match)

        # Refine the fused features
        output = self.resConfUnit2(output)  # (B, features, H, W)

        # Upsample to the target size
        if size is None:
            # Default 2x upsample
            output = F.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=True
            )
        else:
            # Upsample to the specified size (to match the spatial size of the
            # next-layer skip features)
            output = F.interpolate(
                output, size=size, mode="bilinear", align_corners=True
            )

        # 1x1 Conv output: (B, features, H_out, W_out) -> (B, features, H_out, W_out)
        output = self.out_conv(output)
        return output
