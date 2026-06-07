import argparse
import json
import os
import time

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model
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
    sora_l1_penalty,
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


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


def get_effective_rank_info(model: nn.Module, method: str, configured_rank: int):
    if method == "lora":
        return {
            "effective_rank_note": "fixed-rank LoRA",
            "configured_rank": configured_rank,
        }

    if method == "adalora":
        return {
            "effective_rank_note": "adaptive rank (inspect PEFT internals for per-layer details)",
            "configured_rank": configured_rank,
        }

    if method == "sora_like":
        modules = get_sora_like_modules(model)
        rank_list = [m.effective_rank() for m in modules]
        return {
            "effective_rank_sum": int(sum(rank_list)),
            "effective_rank_list": rank_list,
        }

    return {}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    num_epochs: int,
    warmup_ratio: float,
    method: str,
    configured_rank: int,
    sora_lambda: float,
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
    best_state_dict = None

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
                loss = loss + sora_lambda * sora_l1_penalty(model)

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
        row.update(get_effective_rank_info(model, method, configured_rank))
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
            best_state_dict = {
                "model": model.state_dict(),
                "history": history,
            }

    total_train_time = time.time() - total_start
    return history, best_state_dict, total_train_time


def inspect_linear_names(model: nn.Module, limit: int = 200):
    print("=" * 80)
    print("Linear module names:")
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(name)
            count += 1
            if count >= limit:
                break
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["lora", "adalora", "sora_like"])
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--sora_lambda", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs_cola")
    parser.add_argument("--inspect_names_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    if args.inspect_names_only:
        inspect_linear_names(model)
        return

    train_loader, val_loader = build_dataloaders(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Adjust these after running with --inspect_names_only if needed.
    target_modules = ["query_proj", "key_proj", "value_proj", "dense"]

    if args.method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    elif args.method == "adalora":
        peft_config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            init_r=args.rank,
            target_r=max(1, args.rank // 2),
            lora_alpha=args.alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            beta1=0.85,
            beta2=0.85,
            tinit=100,
            tfinal=500,
            deltaT=10,
            orth_reg_weight=0.5,
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    elif args.method == "sora_like":
        for p in model.parameters():
            p.requires_grad = False

        model = replace_linear_with_sora_like(
            model,
            target_substrings=target_modules,
            rank=args.rank,
            alpha=args.alpha / args.rank,
        )

        # Keep classifier trainable
        for name, p in model.named_parameters():
            if "classifier" in name:
                p.requires_grad = True

    else:
        raise ValueError("Unknown method")

    trainable_params = count_trainable_parameters(model)
    total_params = count_total_parameters(model)

    print("=" * 80)
    print(f"method           : {args.method}")
    print(f"device           : {device}")
    print(f"model_name       : {args.model_name}")
    print(f"trainable_params : {trainable_params}")
    print(f"total_params     : {total_params}")
    print(f"rank             : {args.rank}")
    print("=" * 80)

    history, best_state_dict, total_train_time = train(
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
    rank_info = get_effective_rank_info(model, args.method, args.rank)

    metrics = {
        "method": args.method,
        "model_name": args.model_name,
        "device": str(device),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "configured_rank": args.rank,
        "alpha": args.alpha,
        "sora_lambda": args.sora_lambda,
        "num_epochs": args.num_epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "final_eval_loss": final_eval["eval_loss"],
        "final_mcc": final_eval["mcc"],
        "total_train_time_sec": total_train_time,
        "history": history,
        "rank_info": rank_info,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f"{args.method}_metrics.json")
    ckpt_path = os.path.join(args.output_dir, f"{args.method}_best.pt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if best_state_dict is not None:
        torch.save(best_state_dict, ckpt_path)

    print("=" * 80)
    print("FINAL SUMMARY")
    print(f"method           : {args.method}")
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