import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.

    Structure:

        x -> LayerNorm -> Attention -> residual
        x -> LayerNorm -> MLP       -> residual
    """

    def __init__(
        self,
        d_model: int,
        attention: nn.Module,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attention = attention

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x