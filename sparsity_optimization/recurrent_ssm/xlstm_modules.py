import torch
import torch.nn as nn


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


def xlstm_l1_penalty(model: nn.Module) -> torch.Tensor:
    penalty = None
    for m in model.modules():
        if isinstance(m, SoraLikeLinear):
            term = m.g.abs().sum()
            penalty = term if penalty is None else penalty + term

    if penalty is None:
        device = next(model.parameters()).device
        penalty = torch.tensor(0.0, device=device)

    return penalty


def xlstm_rank_info(model: nn.Module):
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


class XLSTMCell(nn.Module):
    """
    Simple LSTM-style recurrent cell.

    Named linear submodules:
      - x2h_i, x2h_f, x2h_o, x2h_g
      - h2h_i, h2h_f, h2h_o, h2h_g
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x2h_i = nn.Linear(input_size, hidden_size)
        self.x2h_f = nn.Linear(input_size, hidden_size)
        self.x2h_o = nn.Linear(input_size, hidden_size)
        self.x2h_g = nn.Linear(input_size, hidden_size)

        self.h2h_i = nn.Linear(hidden_size, hidden_size)
        self.h2h_f = nn.Linear(hidden_size, hidden_size)
        self.h2h_o = nn.Linear(hidden_size, hidden_size)
        self.h2h_g = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        i_t = torch.sigmoid(self.x2h_i(x_t) + self.h2h_i(h_prev))
        f_t = torch.sigmoid(self.x2h_f(x_t) + self.h2h_f(h_prev))
        o_t = torch.sigmoid(self.x2h_o(x_t) + self.h2h_o(h_prev))
        g_t = torch.tanh(self.x2h_g(x_t) + self.h2h_g(h_prev))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class XLSTMClassifier(nn.Module):
    """
    Simple recurrent sequence classifier for CoLA.

    Accepts token_type_ids and ignores them, so Hugging Face batches work.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_size: int = 256,
        num_classes: int = 2,
        pad_token_id: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.cell = XLSTMCell(embed_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        x = self.embedding(input_ids)  # [B, L, E]
        B, L, _ = x.shape

        h = torch.zeros(B, self.cell.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.cell.hidden_size, device=x.device, dtype=x.dtype)

        outputs = []

        for t in range(L):
            x_t = x[:, t, :]
            h_new, c_new = self.cell(x_t, h, c)

            if attention_mask is not None:
                mask_t = attention_mask[:, t].unsqueeze(-1).to(h.dtype)
                h = mask_t * h_new + (1.0 - mask_t) * h
                c = mask_t * c_new + (1.0 - mask_t) * c
            else:
                h, c = h_new, c_new

            outputs.append(h)

        H = torch.stack(outputs, dim=1)  # [B, L, H]

        if attention_mask is None:
            pooled = H[:, -1, :]
        else:
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            idx = (lengths - 1).long()
            pooled = H[torch.arange(B, device=H.device), idx, :]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return type("Output", (), {"logits": logits, "loss": loss})