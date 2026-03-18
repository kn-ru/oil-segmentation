"""
Bottleneck: Hybrid Global-Context Blocks.

2 блока, каждый:
  - Window Self-Attention (8 heads, window 7)
  - Depthwise Conv MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    """Разбить feature map на окна.

    Args:
        x: (B, C, H, W)
        window_size: размер окна

    Returns:
        windows: (B * nH * nW, C, ws, ws)
        nH, nW: количество окон по высоте и ширине
    """
    B, C, H, W = x.shape
    ws = window_size

    # Pad если нужно
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    _, _, Hp, Wp = x.shape
    nH, nW = Hp // ws, Wp // ws

    # (B, C, nH, ws, nW, ws) -> (B*nH*nW, C, ws, ws)
    x = x.reshape(B, C, nH, ws, nW, ws)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B * nH * nW, C, ws, ws)
    return x, nH, nW


def window_unpartition(windows: torch.Tensor, nH: int, nW: int,
                       B: int, H: int, W: int) -> torch.Tensor:
    """Собрать окна обратно."""
    ws = windows.shape[-1]
    C = windows.shape[1]

    x = windows.reshape(B, nH, nW, C, ws, ws)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, nH * ws, nW * ws)

    # Убрать padding
    if x.shape[2] > H or x.shape[3] > W:
        x = x[:, :, :H, :W]
    return x


class WindowAttention(nn.Module):
    """Multi-head self-attention в окнах."""

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        # Register relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = coords.reshape(2, -1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*nWindows, C, ws, ws)
        Returns:
            (B*nWindows, C, ws, ws)
        """
        Bw, C, H, W = x.shape
        N = H * W  # tokens per window

        # Flatten spatial -> (Bw, N, C)
        x_flat = x.reshape(Bw, C, N).permute(0, 2, 1)

        qkv = self.qkv(x_flat).reshape(Bw, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, Bw, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Compute attention in fp32 to prevent overflow in fp16
        attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale

        # Add relative position bias
        rel_bias = self.rel_pos_bias[self.relative_position_index.view(-1)].view(
            N, N, -1).permute(2, 0, 1)
        attn = attn + rel_bias.unsqueeze(0).float()

        attn = attn.softmax(dim=-1).to(v.dtype)

        out = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        out = self.proj(out)

        return out.permute(0, 2, 1).reshape(Bw, C, H, W)


class DepthwiseConvMLP(nn.Module):
    """Depthwise Conv + MLP block."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.dw_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.dw_conv(x))
        x = self.fc2(self.act(self.fc1(x)))
        return x


class HybridGlobalContextBlock(nn.Module):
    """Window Self-Attention + Depthwise Conv MLP с residual."""

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 7):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = DepthwiseConvMLP(dim)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Pre-norm для attention (channel-last для LayerNorm)
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm1(x_norm).permute(0, 3, 1, 2)  # (B, C, H, W)

        # Window attention
        windows, nH, nW = window_partition(x_norm, self.window_size)
        windows = self.attn(windows)
        attn_out = window_unpartition(windows, nH, nW, B, H, W)

        x = x + attn_out

        # Pre-norm для MLP
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm2(x_norm).permute(0, 3, 1, 2)
        x = x + self.mlp(x_norm)

        return x


class Bottleneck(nn.Module):
    """Стек из N hybrid global-context blocks."""

    def __init__(self, dim: int = 768, num_heads: int = 8,
                 window_size: int = 7, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            HybridGlobalContextBlock(dim, num_heads, window_size)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
