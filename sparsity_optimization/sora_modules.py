import torch
import torch.nn as nn


class SoraLikeLinear(nn.Module):
    """
    Simplified SoRA-style gated low-rank adaptation:

        W = W0 + B diag(g) A

    where:
      - W0 is the frozen pretrained linear layer
      - A, B, g are trainable
      - g is sparsity-controlled by an l1 penalty added in training
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be an nn.Linear")

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = alpha

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.g = nn.Parameter(torch.ones(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)                     # [..., out_features]

        z = torch.matmul(x, self.A.t())            # [..., rank]
        z = z * self.g                             # gate rank components
        delta = torch.matmul(z, self.B.t())        # [..., out_features]

        return base_out + self.alpha * delta

    def effective_rank(self, threshold: float = 1e-3) -> int:
        with torch.no_grad():
            return int((self.g.abs() > threshold).sum().item())


def replace_linear_with_sora_like(
    model: nn.Module,
    target_substrings,
    rank: int,
    alpha: float,
):
    """
    Replace selected nn.Linear modules whose full names contain any target substring.
    """
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
        new_layer = SoraLikeLinear(old_layer, rank=rank, alpha=alpha)
        setattr(parent, child_name, new_layer)

    return model


def get_sora_like_modules(model: nn.Module):
    modules = []
    for m in model.modules():
        if isinstance(m, SoraLikeLinear):
            modules.append(m)
    return modules


def sora_l1_penalty(model: nn.Module) -> torch.Tensor:
    penalty = None
    for m in get_sora_like_modules(model):
        term = m.g.abs().sum()
        penalty = term if penalty is None else penalty + term

    if penalty is None:
        device = next(model.parameters()).device
        penalty = torch.tensor(0.0, device=device)

    return penalty