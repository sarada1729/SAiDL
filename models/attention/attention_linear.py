import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearCausalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

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

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        # positive feature map
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        device = x.device
        dtype = x.dtype

        q = self._split_heads(self.q_proj(x))  # [B, H, L, D]
        k = self._split_heads(self.k_proj(x))  # [B, H, L, D]
        v = self._split_heads(self.v_proj(x))  # [B, H, L, D]

        q = self.feature_map(q)
        k = self.feature_map(k)

        kv_running = torch.zeros(
            B, self.n_heads, self.head_dim, self.head_dim,
            device=device, dtype=dtype
        )
        k_running = torch.zeros(
            B, self.n_heads, self.head_dim,
            device=device, dtype=dtype
        )

        outputs = []

        for t in range(L):
            q_t = q[:, :, t, :]  # [B, H, D]
            k_t = k[:, :, t, :]  # [B, H, D]
            v_t = v[:, :, t, :]  # [B, H, D]

            kv_running = kv_running + torch.einsum("bhd,bhe->bhde", k_t, v_t)
            k_running = k_running + k_t

            numerator = torch.einsum("bhd,bhde->bhe", q_t, kv_running)  # [B, H, D]
            denominator = torch.einsum("bhd,bhd->bh", q_t, k_running).unsqueeze(-1) + self.eps

            o_t = numerator / denominator  # [B, H, D]
            outputs.append(o_t.unsqueeze(2))  # [B, H, 1, D]

        out = torch.cat(outputs, dim=2)  # [B, H, L, D]
        out = self._merge_heads(out)     # [B, L, d_model]
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out