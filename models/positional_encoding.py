import math
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Helper function for RoPE.

    Input:
        x: [..., head_dim]

    Output:
        rotated x with pairs (x_even, x_odd) -> (-x_odd, x_even)
    """

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_rotated = torch.stack((-x_odd, x_even), dim=-1)

    return x_rotated.flatten(-2)


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE / Rotary Positional Embedding.

    RoPE modifies q and k before attention scores are computed.

    Instead of adding position vectors to token embeddings, RoPE rotates
    query and key vectors by a position-dependent angle.

    Attention score becomes:

        score(i, j) = RoPE(q_i)^T RoPE(k_j)

    This injects relative position information into q_i^T k_j.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
    ) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires head_dim to be even")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

        positions = torch.arange(max_seq_len).float()

        freqs = torch.einsum("i,j->ij", positions, inv_freq)

        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()[None, None, :, :]  # [1, 1, max_seq_len, head_dim]
        sin = emb.sin()[None, None, :, :]  # [1, 1, max_seq_len, head_dim]

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: [B, H, T, head_dim]
            k: [B, H, T, head_dim]

        Returns:
            q_rotated, k_rotated
        """

        seq_len = q.size(-2)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        cos = self.cos[:, :, :seq_len, :].to(q.device)
        sin = self.sin[:, :, :seq_len, :].to(q.device)

        q_rotated = (q * cos) + (rotate_half(q) * sin)
        k_rotated = (k * cos) + (rotate_half(k) * sin)

        return q_rotated, k_rotated


class ALiBiBias(nn.Module):
    """
    ALiBi: Attention with Linear Biases.

    ALiBi adds a head-specific linear distance penalty to attention scores.

    For causal attention:

        score(i, j) = q_i^T k_j / sqrt(d_h) + bias_h(i, j)

    where j <= i and:

        bias_h(i, j) = - slope_h * (i - j)

    Farther previous tokens receive a larger negative bias.
    """

    def __init__(self, n_heads: int) -> None:
        super().__init__()

        self.n_heads = n_heads

        slopes = self._get_alibi_slopes(n_heads)
        slopes = torch.tensor(slopes, dtype=torch.float32)

        self.register_buffer("slopes", slopes.view(1, n_heads, 1, 1))

    @staticmethod
    def _get_alibi_slopes(n_heads: int) -> list[float]:
        """
        Standard ALiBi slope construction.
        Works for any number of heads.
        """

        def get_slopes_power_of_2(n: int) -> list[float]:
            start = 2.0 ** (-2.0 ** -(math.log2(n) - 3.0))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)

        extra_slopes = ALiBiBias._get_alibi_slopes(2 * closest_power_of_2)
        extra_slopes = extra_slopes[0::2]

        slopes.extend(extra_slopes[: n_heads - closest_power_of_2])

        return slopes

    def forward(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns:
            ALiBi bias of shape [1, H, T, T]
        """

        positions = torch.arange(seq_len, device=device)

        query_pos = positions[:, None]  # [T, 1]
        key_pos = positions[None, :]    # [1, T]

        distance = query_pos - key_pos  # [T, T]

        distance = distance.clamp(min=0).float()

        bias = -self.slopes.to(device) * distance[None, None, :, :]

        return bias


class RelativePositionBias(nn.Module):
    """
    Learned relative positional bias.

    Adds a learned scalar bias depending on relative distance:

        score(i, j) = q_i^T k_j / sqrt(d_h) + b_h(i - j)

    For causal attention, i - j >= 0.

    Distances larger than max_distance are clipped.
    """

    def __init__(
        self,
        n_heads: int,
        max_distance: int = 128,
    ) -> None:
        super().__init__()

        if max_distance <= 0:
            raise ValueError("max_distance must be positive")

        self.n_heads = n_heads
        self.max_distance = max_distance

        self.relative_bias = nn.Embedding(max_distance + 1, n_heads)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns:
            relative position bias of shape [1, H, T, T]
        """

        positions = torch.arange(seq_len, device=device)

        query_pos = positions[:, None]  # [T, 1]
        key_pos = positions[None, :]    # [1, T]

        relative_distance = query_pos - key_pos  # [T, T]

        relative_distance = relative_distance.clamp(
            min=0,
            max=self.max_distance,
        )

        # [T, T, H]
        bias = self.relative_bias(relative_distance)

        # [T, T, H] -> [1, H, T, T]
        bias = bias.permute(2, 0, 1).unsqueeze(0)

        return bias