import argparse
import json
import os
import time

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from sora_modules import (
    replace_linear_with_sora_like,
    get_sora_like_modules,
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_threshold(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)


def l1_subgradient(x: torch.Tensor) -> torch.Tensor:
    g = torch.sign(x)
    g = torch.where(torch.abs(x) < 1e-12, torch.zeros_like(g), g)
    return g


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_dataloaders(model_name: str, batch_size: int, max_length: int = 256):
    dataset = load_dataset("glue", "cola")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["sentence", "idx"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    return train_loader, val_loader


@torch.no_grad()
def evaluate_cola(model: nn.Module, dataloader: DataLoader, device: torch.device):
    metric = evaluate.load("glue", "cola")
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = logits.argmax(dim=-1)

        metric.add_batch(
            predictions=preds.cpu(),
            references=batch["labels"].cpu(),
        )

        if outputs.loss is not None:
            total_loss += outputs.loss.item()
            total_batches += 1

    scores = metric.compute()
    avg_loss = total_loss / max(total_batches, 1)

    return {
        "eval_loss": avg_loss,
        "mcc": float(scores["matthews_correlation"]),
    }


def get_gate_params(model: nn.Module):
    gate_params = []
    non_gate_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".g"):
            gate_params.append((name, p))
        else:
            non_gate_params.append((name, p))

    return gate_params, non_gate_params


def get_rank_info(model: nn.Module):
    modules = get_sora_like_modules(model)
    rank_list = [m.effective_rank() for m in modules]
    return {
        "effective_rank_sum": int(sum(rank_list)),
        "effective_rank_list": rank_list,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr_main: float,
    lr_gate: float,
    weight_decay: float,
    num_epochs: int,
    warmup_ratio: float,
    lam: float,
    gate_update: str,
):
    model.to(device)

    gate_params, non_gate_params = get_gate_params(model)

    optimizer = torch.optim.AdamW(
        [p for _, p in non_gate_params],
        lr=lr_main,
        weight_decay=weight_decay,
    )

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    best_mcc = -1e9
    best_state_dict = None

    total_start = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()

        running_task_loss = 0.0
        running_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            # manually clear gate grads
            for _, g in gate_params:
                if g.grad is not None:
                    g.grad.zero_()

            outputs = model(**batch)
            task_loss = outputs.loss
            task_loss.backward()

            # update non-gate trainable parameters normally
            optimizer.step()
            scheduler.step()

            # manual gate update
            with torch.no_grad():
                for _, g in gate_params:
                    smooth_grad = g.grad.clone() if g.grad is not None else torch.zeros_like(g)

                    if gate_update == "subgrad":
                        g -= lr_gate * (smooth_grad + lam * l1_subgradient(g))
                    elif gate_update == "prox":
                        u = g - lr_gate * smooth_grad
                        g.copy_(soft_threshold(u, lr_gate * lam))
                    else:
                        raise ValueError(f"Unknown gate_update: {gate_update}")

            running_task_loss += task_loss.item()
            running_batches += 1

        epoch_time = time.time() - epoch_start
        train_loss = running_task_loss / max(running_batches, 1)
        val_metrics = evaluate_cola(model, val_loader, device)
        rank_info = get_rank_info(model)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "eval_loss": val_metrics["eval_loss"],
            "mcc": val_metrics["mcc"],
            "epoch_time_sec": epoch_time,
            "effective_rank_sum": rank_info["effective_rank_sum"],
            "effective_rank_list": rank_info["effective_rank_list"],
        }
        history.append(row)

        print(
            f"epoch={epoch+1} "
            f"train_loss={train_loss:.4f} "
            f"eval_loss={val_metrics['eval_loss']:.4f} "
            f"mcc={val_metrics['mcc']:.4f} "
            f"effective_rank_sum={rank_info['effective_rank_sum']} "
            f"time={epoch_time:.2f}s"
        )

        if val_metrics["mcc"] > best_mcc:
            best_mcc = val_metrics["mcc"]
            best_state_dict = {
                "model": model.state_dict(),
                "history": history,
            }

    total_train_time = time.time() - total_start
    return history, best_state_dict, total_train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr_main", type=float, default=2e-4)
    parser.add_argument("--lr_gate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lam", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate_update", type=str, required=True, choices=["prox", "subgrad"])
    parser.add_argument("--output_dir", type=str, default="outputs_cola_sora_gate_compare")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    # freeze whole base model first
    for p in model.parameters():
        p.requires_grad = False

    target_modules = ["query_proj", "key_proj", "value_proj", "dense"]

    model = replace_linear_with_sora_like(
        model,
        target_substrings=target_modules,
        rank=args.rank,
        alpha=args.alpha / args.rank,
    )

    # train classification head and pooler too
    for name, p in model.named_parameters():
        if "classifier" in name or "pooler" in name:
            p.requires_grad = True

    trainable_params = count_trainable_parameters(model)

    print("=" * 80)
    print("SoRA gate update comparison")
    print(f"gate_update      : {args.gate_update}")
    print(f"device           : {device}")
    print(f"model_name       : {args.model_name}")
    print(f"trainable_params : {trainable_params}")
    print(f"rank             : {args.rank}")
    print("=" * 80)

    history, best_state_dict, total_train_time = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr_main=args.lr_main,
        lr_gate=args.lr_gate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        lam=args.lam,
        gate_update=args.gate_update,
    )

    final_eval = evaluate_cola(model, val_loader, device)
    rank_info = get_rank_info(model)

    metrics = {
        "method": "sora_gate_compare",
        "gate_update": args.gate_update,
        "model_name": args.model_name,
        "device": str(device),
        "trainable_params": trainable_params,
        "configured_rank": args.rank,
        "alpha": args.alpha,
        "lambda": args.lam,
        "num_epochs": args.num_epochs,
        "lr_main": args.lr_main,
        "lr_gate": args.lr_gate,
        "final_eval_loss": final_eval["eval_loss"],
        "final_mcc": final_eval["mcc"],
        "total_train_time_sec": total_train_time,
        "history": history,
        "rank_info": rank_info,
    }

    metrics_path = os.path.join(args.output_dir, f"sora_{args.gate_update}_metrics.json")
    ckpt_path = os.path.join(args.output_dir, f"sora_{args.gate_update}_best.pt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if best_state_dict is not None:
        torch.save(best_state_dict, ckpt_path)

    print("=" * 80)
    print("FINAL SUMMARY")
    print(f"gate_update      : {args.gate_update}")
    print(f"final_mcc        : {final_eval['mcc']:.4f}")
    print(f"final_eval_loss  : {final_eval['eval_loss']:.4f}")
    print(f"trainable_params : {trainable_params}")
    print(f"training_time_s  : {total_train_time:.2f}")
    print(f"rank_info        : {rank_info}")
    print(f"saved_metrics    : {metrics_path}")
    print(f"saved_checkpoint : {ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()