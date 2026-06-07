import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.alpha = alpha

        self.A = nn.Parameter(torch.randn(rank, base_linear.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base_linear.out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = torch.matmul(torch.matmul(x, self.A.t()), self.B.t())
        return base_out + self.alpha * delta


class SoraLikeLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.alpha = alpha

        self.A = nn.Parameter(torch.randn(rank, base_linear.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base_linear.out_features, rank))
        self.g = nn.Parameter(torch.ones(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        z = torch.matmul(x, self.A.t())
        z = z * self.g
        delta = torch.matmul(z, self.B.t())
        return base_out + self.alpha * delta

    def effective_rank(self, threshold: float = 1e-3) -> int:
        with torch.no_grad():
            return int((self.g.abs() > threshold).sum().item())


def mamba_l1_penalty(model: nn.Module) -> torch.Tensor:
    penalty = None
    for m in model.modules():
        if isinstance(m, SoraLikeLinear):
            term = m.g.abs().sum()
            penalty = term if penalty is None else penalty + term

    if penalty is None:
        device = next(model.parameters()).device
        penalty = torch.tensor(0.0, device=device)

    return penalty


def mamba_rank_info(model: nn.Module):
    rank_list = []
    for m in model.modules():
        if isinstance(m, SoraLikeLinear):
            rank_list.append(m.effective_rank())
    return {
        "effective_rank_sum": int(sum(rank_list)),
        "effective_rank_list": rank_list,
    }


def replace_selected_linear_layers(
    model: nn.Module,
    target_substrings,
    method: str,
    rank: int,
    alpha: float,
):
    named_modules = dict(model.named_modules())
    to_replace = []

    for name, module in named_modules.items():
        if isinstance(module, nn.Linear) and any(t in name for t in target_substrings):
            to_replace.append(name)

    for full_name in to_replace:
        parent_name = ".".join(full_name.split(".")[:-1])
        child_name = full_name.split(".")[-1]

        parent = model.get_submodule(parent_name) if parent_name else model
        old_layer = getattr(parent, child_name)

        if method == "lora":
            new_layer = LoraLinear(old_layer, rank=rank, alpha=alpha)
        elif method == "sora_like":
            new_layer = SoraLikeLinear(old_layer, rank=rank, alpha=alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

        setattr(parent, child_name, new_layer)

    return model


class SimpleMambaBlock(nn.Module):
    """
    Mamba-like block with named projection layers:
      - in_proj
      - state_proj
      - out_proj

    Not official Mamba; assignment-suitable SSM-style block.
    """

    def __init__(self, d_model: int, expand_factor: int = 2, kernel_size: int = 3):
        super().__init__()

        inner_dim = d_model * expand_factor

        self.in_proj = nn.Linear(d_model, inner_dim)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            groups=inner_dim,
            padding=0,
        )
        self.state_proj = nn.Linear(inner_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, d_model)

        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        z = self.in_proj(x)  # [B, L, D']

        z = z.transpose(1, 2)                    # [B, D', L]
        z = F.pad(z, (self.kernel_size - 1, 0)) # causal padding
        z = self.depthwise_conv(z)
        z = z.transpose(1, 2)                   # [B, L, D']

        z = torch.tanh(self.state_proj(z))
        z = self.out_proj(z)

        return residual + z


class MambaLikeClassifier(nn.Module):
    """
    Mamba-like sequence classifier for CoLA.

    Accepts token_type_ids and ignores them, so Hugging Face batches work.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_classes: int = 2,
        pad_token_id: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([SimpleMambaBlock(embed_dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        x = self.embedding(input_ids)  # [B, L, D]

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        if attention_mask is None:
            pooled = x[:, -1, :]
        else:
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            idx = (lengths - 1).long()
            pooled = x[torch.arange(x.size(0), device=x.device), idx, :]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return type("Output", (), {"logits": logits, "loss": loss})