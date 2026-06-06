import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.positional_encoding import (
    RotaryPositionalEmbedding,
    ALiBiBias,
    RelativePositionBias,
)


class SlidingWindowCausalAttention(nn.Module):
    """
    Sliding-window causal self-attention with support for:

        positional_encoding_type="none"
        positional_encoding_type="rope"
        positional_encoding_type="alibi"
        positional_encoding_type="relative"

    Token i can attend to token j iff:

        i - window_size + 1 <= j <= i

    So the model remains causal and only uses recent context directly.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        positional_encoding_type: str = "none",
        relative_max_distance: int = 128,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if window_size <= 0:
            raise ValueError("window_size must be positive")

        positional_encoding_type = positional_encoding_type.lower()

        allowed_position_types = {"none", "rope", "alibi", "relative"}

        if positional_encoding_type not in allowed_position_types:
            raise ValueError(
                f"Unknown positional_encoding_type={positional_encoding_type}. "
                f"Expected one of: {sorted(allowed_position_types)}"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.positional_encoding_type = positional_encoding_type
        self.relative_max_distance = relative_max_distance

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if positional_encoding_type == "rope":
            self.position_module = RotaryPositionalEmbedding(
                head_dim=self.head_dim,
                max_seq_len=max_seq_len,
            )

        elif positional_encoding_type == "alibi":
            self.position_module = ALiBiBias(
                n_heads=n_heads,
            )

        elif positional_encoding_type == "relative":
            self.position_module = RelativePositionBias(
                n_heads=n_heads,
                max_distance=relative_max_distance,
            )

        else:
            self.position_module = None

    def _build_sliding_window_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns:
            mask of shape [1, 1, seq_len, seq_len]

        mask[..., i, j] = True iff token i can attend to token j.
        """

        positions = torch.arange(seq_len, device=device)

        query_pos = positions[:, None]  # [T, 1]
        key_pos = positions[None, :]    # [1, T]

        causal_mask = key_pos <= query_pos
        window_mask = key_pos >= query_pos - self.window_size + 1

        mask = causal_mask & window_mask

        return mask[None, None, :, :]  # [1, 1, T, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            out: [B, T, d_model]
        """

        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, T, D], got {tuple(x.shape)}")

        batch_size, seq_len, d_model = x.shape

        if d_model != self.d_model:
            raise ValueError(
                f"Expected hidden dimension {self.d_model}, got {d_model}"
            )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, D] -> [B, H, T, Hd]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE modifies q and k before attention scores are computed.
        if self.positional_encoding_type == "rope":
            q, k = self.position_module(q, k)

        # [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ALiBi and relative position bias modify attention scores directly.
        if self.positional_encoding_type == "alibi":
            scores = scores + self.position_module(seq_len, x.device)

        elif self.positional_encoding_type == "relative":
            scores = scores + self.position_module(seq_len, x.device)

        mask = self._build_sliding_window_causal_mask(
            seq_len=seq_len,
            device=x.device,
        )

        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        # [B, H, T, Hd] -> [B, T, D]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        out = self.out_proj(out)

        return out