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
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from xlstm_modules import (
    XLSTMClassifier,
    replace_selected_linear_layers,
    xlstm_l1_penalty,
    xlstm_rank_info,
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(model_name: str, batch_size: int, max_length: int = 128):
    dataset = load_dataset("glue", "cola")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

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

    return train_loader, val_loader, tokenizer


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate_cola(model, dataloader, device):
    metric = evaluate.load("glue", "cola")
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)

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


def get_rank_info(method: str, model: nn.Module, configured_rank: int):
    if method == "lora":
        return {"configured_rank": configured_rank, "effective_rank_note": "fixed-rank LoRA"}
    if method == "sora_like":
        return xlstm_rank_info(model)
    return {}


def train(
    model,
    train_loader,
    val_loader,
    device,
    lr,
    weight_decay,
    num_epochs,
    warmup_ratio,
    method,
    configured_rank,
    sora_lambda,
):
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
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
    best_state = None

    total_start = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()

        running_loss = 0.0
        running_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            if method == "sora_like":
                loss = loss + sora_lambda * xlstm_l1_penalty(model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_batches += 1

        epoch_time = time.time() - epoch_start
        train_loss = running_loss / max(running_batches, 1)

        val_metrics = evaluate_cola(model, val_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "eval_loss": val_metrics["eval_loss"],
            "mcc": val_metrics["mcc"],
            "epoch_time_sec": epoch_time,
        }
        row.update(get_rank_info(method, model, configured_rank))
        history.append(row)

        print(
            f"epoch={epoch+1} "
            f"train_loss={train_loss:.4f} "
            f"eval_loss={val_metrics['eval_loss']:.4f} "
            f"mcc={val_metrics['mcc']:.4f} "
            f"time={epoch_time:.2f}s"
        )

        if val_metrics["mcc"] > best_mcc:
            best_mcc = val_metrics["mcc"]
            best_state = {
                "model": model.state_dict(),
                "history": history,
            }

    total_time = time.time() - total_start
    return history, best_state, total_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["lora", "sora_like"])
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--sora_lambda", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs_recurrent_ssm")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, tokenizer = build_dataloaders(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    model = XLSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_classes=2,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )

    # freeze everything before inserting adapters
    for p in model.parameters():
        p.requires_grad = False

    target_modules = [
        "x2h_i", "x2h_f", "x2h_o", "x2h_g",
        "h2h_i", "h2h_f", "h2h_o", "h2h_g",
    ]

    model = replace_selected_linear_layers(
        model=model,
        target_substrings=target_modules,
        method=args.method,
        rank=args.rank,
        alpha=args.alpha / args.rank,
    )

    # classifier should be trainable
    for name, p in model.named_parameters():
        if "classifier" in name or "embedding" in name:
            p.requires_grad = True

    trainable_params = count_trainable_parameters(model)

    print("=" * 80)
    print("xLSTM recurrent PEFT run")
    print(f"method           : {args.method}")
    print(f"device           : {device}")
    print(f"trainable_params : {trainable_params}")
    print("=" * 80)

    history, best_state, total_time = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        method=args.method,
        configured_rank=args.rank,
        sora_lambda=args.sora_lambda,
    )

    final_eval = evaluate_cola(model, val_loader, device)
    rank_info = get_rank_info(args.method, model, args.rank)

    metrics = {
        "architecture": "xlstm_like",
        "method": args.method,
        "trainable_params": trainable_params,
        "configured_rank": args.rank,
        "final_eval_loss": final_eval["eval_loss"],
        "final_mcc": final_eval["mcc"],
        "total_train_time_sec": total_time,
        "rank_info": rank_info,
        "history": history,
    }

    metrics_path = os.path.join(args.output_dir, f"xlstm_{args.method}_metrics.json")
    ckpt_path = os.path.join(args.output_dir, f"xlstm_{args.method}_best.pt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if best_state is not None:
        torch.save(best_state, ckpt_path)

    print("=" * 80)
    print("FINAL SUMMARY")
    print(f"architecture      : xlstm_like")
    print(f"method            : {args.method}")
    print(f"final_mcc         : {final_eval['mcc']:.4f}")
    print(f"trainable_params  : {trainable_params}")
    print(f"training_time_s   : {total_time:.2f}")
    print(f"rank_info         : {rank_info}")
    print(f"saved_metrics     : {metrics_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()