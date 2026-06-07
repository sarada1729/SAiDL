import torch
import torch.nn as nn

from models.transformer_block import (
    TransformerBlock,
    HybridTransformerBlockConvBeforeAttn,
    HybridTransformerBlockConvFFN,
)
from models.attention.attention_sliding_window import SlidingWindowCausalAttention


def build_attention(
    attention_type: str,
    d_model: int,
    n_heads: int,
    dropout: float,
    max_seq_len: int = 1024,
    window_size: int = 128,
    positional_encoding_type: str = "rope",
    relative_max_distance: int = 128,
) -> nn.Module:
    attention_type = attention_type.lower()
    positional_encoding_type = positional_encoding_type.lower()

    if attention_type == "sliding_window":
        return SlidingWindowCausalAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout,
            max_seq_len=max_seq_len,
            positional_encoding_type=positional_encoding_type,
            relative_max_distance=relative_max_distance,
        )

    raise ValueError(
        f"Unknown attention_type={attention_type}. "
        "This file currently supports only attention_type='sliding_window'."
    )


def build_block(
    block_type: str,
    d_model: int,
    attention: nn.Module,
    d_ff: int,
    kernel_size: int,
    dropout: float,
) -> nn.Module:
    block_type = block_type.lower()

    if block_type == "standard":
        return TransformerBlock(
            d_model=d_model,
            attention=attention,
            d_ff=d_ff,
            dropout=dropout,
        )

    elif block_type == "conv_before_attn":
        return HybridTransformerBlockConvBeforeAttn(
            d_model=d_model,
            attention=attention,
            d_ff=d_ff,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    elif block_type == "conv_ffn":
        return HybridTransformerBlockConvFFN(
            d_model=d_model,
            attention=attention,
            d_ff=d_ff,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    raise ValueError(
        f"Unknown block_type={block_type}. "
        "Expected one of: ['standard', 'conv_before_attn', 'conv_ffn']"
    )


class SimpleTransformerLM(nn.Module):
    """
    Transformer language model with selectable block type.

    Supported:
        attention_type = "sliding_window"
        positional_encoding_type = "rope" | "alibi" | "relative"
        block_type = "standard" | "conv_before_attn" | "conv_ffn"
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        attention_type: str = "sliding_window",
        positional_encoding_type: str = "rope",
        block_type: str = "standard",
        window_size: int = 128,
        relative_max_distance: int = 128,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        attention_type = attention_type.lower()
        positional_encoding_type = positional_encoding_type.lower()
        block_type = block_type.lower()

        allowed_position_types = {"rope", "alibi", "relative"}
        if positional_encoding_type not in allowed_position_types:
            raise ValueError(
                f"Unknown positional_encoding_type={positional_encoding_type}. "
                f"Expected one of: {sorted(allowed_position_types)}"
            )

        allowed_block_types = {"standard", "conv_before_attn", "conv_ffn"}
        if block_type not in allowed_block_types:
            raise ValueError(
                f"Unknown block_type={block_type}. "
                f"Expected one of: {sorted(allowed_block_types)}"
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.attention_type = attention_type
        self.positional_encoding_type = positional_encoding_type
        self.block_type = block_type
        self.window_size = window_size
        self.relative_max_distance = relative_max_distance
        self.kernel_size = kernel_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList(
            [
                build_block(
                    block_type=block_type,
                    d_model=d_model,
                    attention=build_attention(
                        attention_type=attention_type,
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=dropout,
                        max_seq_len=max_seq_len,
                        window_size=window_size,
                        positional_encoding_type=positional_encoding_type,
                        relative_max_distance=relative_max_distance,
                    ),
                    d_ff=d_ff,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected input shape [B, L], got {tuple(x.shape)}")

        _, seq_len = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        h = self.token_embedding(x)

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        logits = self.lm_head(h)
        return logits