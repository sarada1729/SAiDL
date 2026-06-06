import torch
import torch.nn as nn
import torch.nn.functional as F


class FullCausalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, d_model] -> [B, H, L, D]
        B, L, _ = x.shape
        x = x.view(B, L, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, L, D] -> [B, L, d_model]
        B, H, L, D = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, L, H * D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        q = self._split_heads(self.q_proj(x))  # [B, H, L, D]
        k = self._split_heads(self.k_proj(x))  # [B, H, L, D]
        v = self._split_heads(self.v_proj(x))  # [B, H, L, D]

        # Full scaled dot-product attention.
        scores = q @ k.transpose(-2, -1)  # [B, H, L, L]
        scores = scores / (self.head_dim ** 0.5)

        # Causal mask: token t cannot attend to future tokens.
        causal_mask = torch.tril(
            torch.ones(L, L, device=x.device, dtype=torch.bool)
        )

        scores = scores.masked_fill(~causal_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # [B, H, L, D]
        out = self._merge_heads(out)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out
