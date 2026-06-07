import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1dSubLayer(nn.Module):
    """
    Causal Conv1D sublayer that preserves sequence length.

    Input:  [B, L, d_model]
    Output: [B, L, d_model]
    """

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()

        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")

        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=0,   # causal padding handled manually
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        x = x.transpose(1, 2)  # [B, d_model, L]

        pad_left = self.kernel_size - 1
        x = F.pad(x, (pad_left, 0))  # causal left padding

        x = self.conv(x)             # [B, d_model, L]
        x = self.act(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)        # [B, L, d_model]
        return x


class ConvFeedForward(nn.Module):
    """
    Convolutional replacement for the FFN.

    Input:  [B, L, d_model]
    Output: [B, L, d_model]
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")

        self.kernel_size = kernel_size

        self.in_proj = nn.Linear(d_model, d_ff)
        self.conv = nn.Conv1d(
            in_channels=d_ff,
            out_channels=d_ff,
            kernel_size=kernel_size,
            padding=0,   # causal padding handled manually
        )
        self.act = nn.GELU()
        self.out_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        x = self.in_proj(x)          # [B, L, d_ff]
        x = x.transpose(1, 2)        # [B, d_ff, L]

        pad_left = self.kernel_size - 1
        x = F.pad(x, (pad_left, 0))  # causal left padding

        x = self.conv(x)             # [B, d_ff, L]
        x = x.transpose(1, 2)        # [B, L, d_ff]

        x = self.act(x)
        x = self.dropout(x)
        x = self.out_proj(x)         # [B, L, d_model]
        x = self.dropout(x)

        return x