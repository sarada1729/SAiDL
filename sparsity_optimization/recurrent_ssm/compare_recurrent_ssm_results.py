import json
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rank_summary(metrics):
    info = metrics.get("rank_info", {})
    if "effective_rank_sum" in info:
        return str(info["effective_rank_sum"])
    if "configured_rank" in info:
        return str(info["configured_rank"])
    return "NA"


def main():
    base = "outputs_recurrent_ssm"

    files = [
        ("xLSTM + LoRA", os.path.join(base, "xlstm_lora_metrics.json")),
        ("xLSTM + SoRA-like", os.path.join(base, "xlstm_sora_like_metrics.json")),
        ("Mamba-like + LoRA", os.path.join(base, "mamba_lora_metrics.json")),
        ("Mamba-like + SoRA-like", os.path.join(base, "mamba_sora_like_metrics.json")),
    ]

    rows = []
    for name, path in files:
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue
        m = load_json(path)
        rows.append(
            {
                "Method": name,
                "MCC": f"{m['final_mcc']:.4f}",
                "Trainable Params": str(m["trainable_params"]),
                "Effective Rank": rank_summary(m),
                "Training Time (s)": f"{m['total_train_time_sec']:.2f}",
            }
        )

    if not rows:
        print("No result files found.")
        return

    headers = list(rows[0].keys())
    widths = {h: max(len(h), *(len(r[h]) for r in rows)) for h in headers}

    def fmt_row(r):
        return " | ".join(r[h].ljust(widths[h]) for h in headers)

    print(fmt_row({h: h for h in headers}))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(fmt_row(r))


if __name__ == "__main__":
    main()