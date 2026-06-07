import torch
import torch.nn as nn

from models.conv_modules import CausalConv1dSubLayer, ConvFeedForward


class TransformerBlock(nn.Module):
    """
    Standard transformer block:
        x -> x + Attn(LN(x))
        x -> x + FFN(LN(x))
    """

    def __init__(
        self,
        d_model: int,
        attention: nn.Module,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = attention
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class HybridTransformerBlockConvBeforeAttn(nn.Module):
    """
    Hybrid Design A:
        x -> x + Conv(LN(x))
        x -> x + Attn(LN(x))
        x -> x + FFN(LN(x))
    """

    def __init__(
        self,
        d_model: int,
        attention: nn.Module,
        d_ff: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.ln_conv = nn.LayerNorm(d_model)
        self.conv = CausalConv1dSubLayer(
            d_model=d_model,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.ln_attn = nn.LayerNorm(d_model)
        self.attention = attention

        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv(self.ln_conv(x))
        x = x + self.attention(self.ln_attn(x))
        x = x + self.ffn(self.ln_ffn(x))
        return x


class HybridTransformerBlockConvFFN(nn.Module):
    """
    Hybrid Design B:
        x -> x + Attn(LN(x))
        x -> x + ConvFFN(LN(x))
    """

    def __init__(
        self,
        d_model: int,
        attention: nn.Module,
        d_ff: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.ln_attn = nn.LayerNorm(d_model)
        self.attention = attention

        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = ConvFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_attn(x))
        x = x + self.ffn(self.ln_ffn(x))
        return x