import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockSparseCausalAttention(nn.Module):
    """
    Block-sparse causal self-attention.

    The sequence is divided into blocks of size `block_size`.

    For each query block B_r, this implementation allows attention to:
        1. the current block B_r,
        2. the previous `num_local_blocks - 1` blocks,
        3. the first `num_global_blocks` blocks.

    Example:
        block_size = 128
        num_local_blocks = 2
        num_global_blocks = 1

    Then each block attends to:
        - itself,
        - the previous block,
        - block 0 globally.

    Token-level causality is still enforced, so token i never attends to token j > i.

    Important:
        This implementation uses a dense [T, T] attention score matrix and masks it.
        So it is mathematically block-sparse, but not yet a hardware-optimized sparse kernel.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int = 128,
        num_local_blocks: int = 2,
        num_global_blocks: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if block_size <= 0:
            raise ValueError("block_size must be positive")

        if num_local_blocks <= 0:
            raise ValueError("num_local_blocks must be positive")

        if num_global_blocks < 0:
            raise ValueError("num_global_blocks cannot be negative")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.block_size = block_size
        self.num_local_blocks = num_local_blocks
        self.num_global_blocks = num_global_blocks

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _build_block_sparse_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a block-sparse causal mask.

        Returns:
            mask of shape [1, 1, seq_len, seq_len]

        mask[..., i, j] = True means query position i may attend to key position j.
        """

        positions = torch.arange(seq_len, device=device)

        query_pos = positions[:, None]   # [T, 1]
        key_pos = positions[None, :]     # [1, T]

        # Token-level causal condition: key position j must satisfy j <= i.
        causal_mask = key_pos <= query_pos

        # Convert token positions to block indices.
        query_block = query_pos // self.block_size
        key_block = key_pos // self.block_size

        # Local block attention:
        # block r attends to blocks:
        # r, r-1, ..., r-num_local_blocks+1
        local_block_mask = (
            (key_block <= query_block)
            & (key_block >= query_block - self.num_local_blocks + 1)
        )

        # Global block attention:
        # every block can attend to the first num_global_blocks blocks.
        global_block_mask = key_block < self.num_global_blocks

        # Block sparsity + token-level causality.
        block_sparse_mask = local_block_mask | global_block_mask
        mask = causal_mask & block_sparse_mask

        return mask[None, None, :, :]  # [1, 1, T, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [B, T, d_model]

        Returns:
            tensor of shape [B, T, d_model]
        """

        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, T, D], got {tuple(x.shape)}")

        batch_size, seq_len, d_model = x.shape

        if d_model != self.d_model:
            raise ValueError(
                f"Expected input hidden dimension {self.d_model}, got {d_model}"
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, D] -> [B, H, T, Hd]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = self._build_block_sparse_causal_mask(
            seq_len=seq_len,
            device=x.device,
        )

        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, T, Hd]

        # [B, H, T, Hd] -> [B, T, D]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        out = self.out_proj(out)

        return out