#!/usr/bin/env python3
"""
Compact Swin Transformer for 64x64 Handwritten Chinese Character Recognition (HCCR).
Adapted from Liu et al. (ICCV 2021) and recent DSP 2024/2025 literature for small-resolution offline Hanzi.

Key Specifications:
- Input Resolution: 64x64, 1-channel Grayscale
- Patch Size: 4x4 (produces 16x16 = 256 tokens) or 2x2 (produces 32x32 = 1024 tokens)
- Window Size: M = 4
- Embedding Dims: (48, 96, 192, 384)
- Depths: (2, 2, 6, 2)
- Shifted Window Multi-Head Self-Attention (W-MSA & SW-MSA) with Relative Position Bias
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size M
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention with Relative Position Bias.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Coordinate grid for relative position indices
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, N
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with alternating Shifted Window Self-Attention.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=4, shift_size=0,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature size mismatch: {L} vs {H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift if shifted window
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x

class PatchMerging(nn.Module):
    """
    Patch Merging Layer: 2x2 downsampling, doubles channel dimension.
    """
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature size mismatch"

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x

class SwinHCCR_64(nn.Module):
    """
    Compact Swin Transformer for 64x64 Grayscale Chinese Characters.
    Stage dims: (48, 96, 192, 384), Depths: (2, 2, 4, 2).
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=15,
        patch_size=4,
        embed_dim=48,
        depths=(2, 2, 4, 2),
        num_heads=(3, 6, 12, 24),
        window_size=4,
        feature_dim=256,
        drop_rate=0.0
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Patch Embedding: 64x64 -> 16x16 with embed_dim=48
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Build Stages
        self.stages = nn.ModuleList()
        current_res = (64 // patch_size, 64 // patch_size)  # (16, 16)
        current_dim = embed_dim

        for i_stage in range(len(depths)):
            # Swin Blocks
            blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=current_dim,
                    input_resolution=current_res,
                    num_heads=num_heads[i_stage],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    drop=drop_rate
                )
                for i in range(depths[i_stage])
            ])

            # Downsample (except for last stage)
            if i_stage < len(depths) - 1:
                downsample = PatchMerging(current_res, current_dim)
                next_res = (current_res[0] // 2, current_res[1] // 2)
                next_dim = current_dim * 2
            else:
                downsample = None
                next_res = current_res
                next_dim = current_dim

            self.stages.append(nn.ModuleDict({
                'blocks': blocks,
                'downsample': downsample
            }))
            current_res = next_res
            current_dim = next_dim

        self.norm = nn.LayerNorm(current_dim)
        self.fc = nn.Linear(current_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        # Patch Embed: (B, 1, 64, 64) -> (B, 48, 16, 16)
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 256, 48)
        x = self.pos_drop(x)

        for stage in self.stages:
            for block in stage['blocks']:
                x = block(x)
            if stage['downsample'] is not None:
                x = stage['downsample'](x)

        x = self.norm(x)  # (B, 4, 384)
        x = x.mean(dim=1)  # Global Average Pooling -> (B, 384)
        feats = F.gelu(self.fc(x))
        return feats

    def forward(self, x):
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits, feats
