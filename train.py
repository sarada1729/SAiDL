import time
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data.dataset import build_dataloaders
from models.transformer_lm import SimpleTransformerLM


def evaluate_loss_and_ppl(model, data_loader, criterion, device, max_steps=None):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for step, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            B, L, V = logits.shape

            loss = criterion(
                logits.reshape(B * L, V),
                y.reshape(B * L),
            )

            total_loss += loss.item()
            total_batches += 1

            if max_steps is not None and (step + 1) >= max_steps:
                break

    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def measure_inference_throughput(model, data_loader, device, max_steps=100):
    model.eval()

    total_tokens = 0
    start_time = time.time()

    with torch.no_grad():
        for step, (x, _) in enumerate(data_loader):
            x = x.to(device)
            _ = model(x)

            B, L = x.shape
            total_tokens += B * L

            if (step + 1) >= max_steps:
                break

    elapsed = time.time() - start_time
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    return throughput


def measure_peak_gpu_memory(model, data_loader, device, max_steps=20):
    if device.type != "cuda":
        return None

    model.eval()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for step, (x, _) in enumerate(data_loader):
            x = x.to(device)
            _ = model(x)

            if (step + 1) >= max_steps:
                break

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = peak_bytes / (1024 ** 2)
    return peak_mb


def save_plots(
    train_steps,
    train_losses,
    val_steps,
    val_losses,
    val_ppls,
    train_throughput_steps,
    train_throughputs,
    inference_throughput_steps,
    inference_throughputs,
    output_prefix,
):
    plt.figure()
    plt.plot(train_steps, train_losses, marker="o")
    plt.xlabel("Training Step")
    plt.ylabel("Batch Loss")
    plt.title("Training Loss vs Step")
    plt.savefig(f"{output_prefix}_train_loss.png")
    plt.close()

    plt.figure()
    plt.plot(val_steps, val_losses, marker="o")
    plt.xlabel("Training Step")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Step")
    plt.savefig(f"{output_prefix}_val_loss.png")
    plt.close()

    plt.figure()
    plt.plot(val_steps, val_ppls, marker="o")
    plt.xlabel("Training Step")
    plt.ylabel("Validation Perplexity")
    plt.title("Validation Perplexity vs Step")
    plt.savefig(f"{output_prefix}_val_ppl.png")
    plt.close()

    plt.figure()
    plt.plot(train_throughput_steps, train_throughputs, marker="o", label="Train throughput")
    plt.plot(inference_throughput_steps, inference_throughputs, marker="o", label="Inference throughput")
    plt.xlabel("Training Step")
    plt.ylabel("Tokens / sec")
    plt.title("Throughput vs Step")
    plt.legend()
    plt.savefig(f"{output_prefix}_throughput.png")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============================================================
    # CONFIG
    # ============================================================
    attention_type = "sliding_window"
    positional_encoding_type = "alibi"      # "rope" | "alibi" | "relative"
    block_type = "standard"         # "standard" | "conv_before_attn" | "conv_ffn"

    context_length = 1024
    batch_size = 2
    d_model = 128
    n_heads = 4
    n_layers = 2
    d_ff = 256
    dropout = 0.1

    window_size = 128
    relative_max_distance = 128
    kernel_size = 3

    learning_rate = 3e-4
    num_epochs = 1
    max_train_steps = 1000
    validate_every = 100

    max_val_steps = 200
    max_infer_steps = 100
    max_mem_steps = 20

    output_prefix = (
        f"{attention_type}_{positional_encoding_type}_{block_type}_"
        f"ctx{context_length}_benchmark"
    )
    checkpoint_path = (
        f"{attention_type}_{positional_encoding_type}_{block_type}_"
        f"ctx{context_length}_best_checkpoint.pt"
    )
    metrics_path = (
        f"{attention_type}_{positional_encoding_type}_{block_type}_"
        f"ctx{context_length}_metrics.json"
    )
    # ============================================================

    train_loader, valid_loader, test_loader, tokenizer = build_dataloaders(
        context_length=context_length,
        batch_size=batch_size,
        tokenizer_name="gpt2",
    )

    vocab_size = len(tokenizer)

    model = SimpleTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=context_length,
        dropout=dropout,
        attention_type=attention_type,
        positional_encoding_type=positional_encoding_type,
        block_type=block_type,
        window_size=window_size,
        relative_max_distance=relative_max_distance,
        kernel_size=kernel_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    train_steps = []
    train_losses = []

    val_steps = []
    val_losses = []
    val_ppls = []

    train_throughput_steps = []
    train_throughputs = []

    inference_throughput_steps = []
    inference_throughputs = []

    nan_detected = False
    inf_detected = False
    max_grad_norm_seen = 0.0

    print("=" * 80)
    print("Starting benchmark")
    print(f"Device                   : {device}")
    print(f"Attention type           : {attention_type}")
    print(f"Positional encoding type : {positional_encoding_type}")
    print(f"Block type               : {block_type}")
    print(f"Context length           : {context_length}")
    print(f"Batch size               : {batch_size}")
    print(f"Vocab size               : {vocab_size}")
    print(f"d_model                  : {d_model}")
    print(f"n_heads                  : {n_heads}")
    print(f"n_layers                 : {n_layers}")
    print(f"d_ff                     : {d_ff}")
    print(f"Window size              : {window_size}")
    print(f"Relative max distance    : {relative_max_distance}")
    print(f"Kernel size              : {kernel_size}")
    print(f"Max train steps          : {max_train_steps}")
    print(f"Validate every           : {validate_every}")
    print("=" * 80)

    global_step = 0
    epoch_times = []

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()

        train_window_start_time = time.time()
        train_tokens_since_last_eval = 0

        for _, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            B, L, V = logits.shape

            loss = criterion(
                logits.reshape(B * L, V),
                y.reshape(B * L),
            )

            if torch.isnan(loss):
                nan_detected = True
                print("NaN loss detected. Stopping.")
                break

            if torch.isinf(loss):
                inf_detected = True
                print("Inf loss detected. Stopping.")
                break

            optimizer.zero_grad()
            loss.backward()

            total_sq_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm = p.grad.detach().norm(2).item()
                    total_sq_norm += grad_norm ** 2
            total_grad_norm = total_sq_norm ** 0.5
            max_grad_norm_seen = max(max_grad_norm_seen, total_grad_norm)

            optimizer.step()

            global_step += 1
            train_tokens_since_last_eval += B * L

            train_steps.append(global_step)
            train_losses.append(loss.item())

            if global_step % validate_every == 0:
                elapsed_train = time.time() - train_window_start_time
                train_throughput = (
                    train_tokens_since_last_eval / elapsed_train if elapsed_train > 0 else 0.0
                )

                val_loss, val_ppl = evaluate_loss_and_ppl(
                    model=model,
                    data_loader=valid_loader,
                    criterion=criterion,
                    device=device,
                    max_steps=max_val_steps,
                )

                infer_throughput = measure_inference_throughput(
                    model=model,
                    data_loader=valid_loader,
                    device=device,
                    max_steps=max_infer_steps,
                )

                peak_gpu_memory_mb = measure_peak_gpu_memory(
                    model=model,
                    data_loader=valid_loader,
                    device=device,
                    max_steps=max_mem_steps,
                )

                print("-" * 80)
                print(f"Epoch {epoch + 1}/{num_epochs} | Step {global_step}")
                print(f"Train batch loss        : {loss.item():.4f}")
                print(f"Validation loss         : {val_loss:.4f}")
                print(f"Validation perplexity   : {val_ppl:.4f}")
                print(f"Train throughput        : {train_throughput:.2f} tokens/sec")
                print(f"Inference throughput    : {infer_throughput:.2f} tokens/sec")
                if peak_gpu_memory_mb is None:
                    print("Peak GPU memory         : not measured (CPU run)")
                else:
                    print(f"Peak GPU memory         : {peak_gpu_memory_mb:.2f} MB")
                print(f"Max grad norm so far    : {max_grad_norm_seen:.4f}")

                val_steps.append(global_step)
                val_losses.append(val_loss)
                val_ppls.append(val_ppl)

                train_throughput_steps.append(global_step)
                train_throughputs.append(train_throughput)

                inference_throughput_steps.append(global_step)
                inference_throughputs.append(infer_throughput)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": best_val_loss,
                            "config": {
                                "attention_type": attention_type,
                                "positional_encoding_type": positional_encoding_type,
                                "block_type": block_type,
                                "context_length": context_length,
                                "batch_size": batch_size,
                                "d_model": d_model,
                                "n_heads": n_heads,
                                "n_layers": n_layers,
                                "d_ff": d_ff,
                                "dropout": dropout,
                                "window_size": window_size,
                                "relative_max_distance": relative_max_distance,
                                "kernel_size": kernel_size,
                                "learning_rate": learning_rate,
                            },
                        },
                        checkpoint_path,
                    )
                    print(f"Saved best checkpoint   : {checkpoint_path}")

                train_window_start_time = time.time()
                train_tokens_since_last_eval = 0
                model.train()

            if global_step >= max_train_steps:
                print(f"Reached max_train_steps = {max_train_steps}")
                break

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        if nan_detected or inf_detected or global_step >= max_train_steps:
            break

    final_val_loss, final_val_ppl = evaluate_loss_and_ppl(
        model=model,
        data_loader=valid_loader,
        criterion=criterion,
        device=device,
        max_steps=max_val_steps,
    )

    final_infer_throughput = measure_inference_throughput(
        model=model,
        data_loader=valid_loader,
        device=device,
        max_steps=max_infer_steps,
    )

    final_peak_gpu_memory_mb = measure_peak_gpu_memory(
        model=model,
        data_loader=valid_loader,
        device=device,
        max_steps=max_mem_steps,
    )

    test_loss, test_ppl = evaluate_loss_and_ppl(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        max_steps=max_val_steps,
    )

    save_plots(
        train_steps=train_steps,
        train_losses=train_losses,
        val_steps=val_steps,
        val_losses=val_losses,
        val_ppls=val_ppls,
        train_throughput_steps=train_throughput_steps,
        train_throughputs=train_throughputs,
        inference_throughput_steps=inference_throughput_steps,
        inference_throughputs=inference_throughputs,
        output_prefix=output_prefix,
    )

    stability_summary = {
        "nan_detected": nan_detected,
        "inf_detected": inf_detected,
        "max_grad_norm_seen": max_grad_norm_seen,
        "loss_remained_finite": (not nan_detected) and (not inf_detected),
    }

    metrics = {
        "attention_type": attention_type,
        "positional_encoding_type": positional_encoding_type,
        "block_type": block_type,
        "device": str(device),
        "context_length": context_length,
        "batch_size": batch_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "dropout": dropout,
        "window_size": window_size,
        "relative_max_distance": relative_max_distance,
        "kernel_size": kernel_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "max_train_steps": max_train_steps,
        "validate_every": validate_every,
        "final_validation_loss": final_val_loss,
        "final_validation_perplexity": final_val_ppl,
        "final_inference_throughput_tokens_per_sec": final_infer_throughput,
        "peak_gpu_memory_mb": final_peak_gpu_memory_mb,
        "epoch_times_sec": epoch_times,
        "final_test_loss": test_loss,
        "final_test_perplexity": test_ppl,
        "stability": stability_summary,
        "plots": {
            "train_loss": f"{output_prefix}_train_loss.png",
            "val_loss": f"{output_prefix}_val_loss.png",
            "val_ppl": f"{output_prefix}_val_ppl.png",
            "throughput": f"{output_prefix}_throughput.png",
        },
        "checkpoint": checkpoint_path,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 80)
    print("FINAL BENCHMARK SUMMARY")
    print(f"Attention type         : {attention_type}")
    print(f"Positional encoding    : {positional_encoding_type}")
    print(f"Block type             : {block_type}")
    print(f"Validation loss        : {final_val_loss:.4f}")
    print(f"Validation perplexity  : {final_val_ppl:.4f}")
    print(f"Inference throughput   : {final_infer_throughput:.2f} tokens/sec")
    if final_peak_gpu_memory_mb is None:
        print("Peak GPU memory        : not measured (CPU run)")
    else:
        print(f"Peak GPU memory        : {final_peak_gpu_memory_mb:.2f} MB")
    print(f"Test loss              : {test_loss:.4f}")
    print(f"Test perplexity        : {test_ppl:.4f}")
    print(f"Training stability     : {stability_summary}")
    print(f"Saved checkpoint       : {checkpoint_path}")
    print(f"Saved metrics JSON     : {metrics_path}")
    print(f"Saved plots prefix     : {output_prefix}_*.png")
    print("=" * 80)


if __name__ == "__main__":
    main()