import json
import os

import matplotlib.pyplot as plt
import torch


def soft_threshold(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)


def l1_subgradient(x: torch.Tensor) -> torch.Tensor:
    g = torch.sign(x)
    g = torch.where(torch.abs(x) < 1e-12, torch.zeros_like(g), g)
    return g


def smooth_objective(g: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum((g - a) ** 2)


def smooth_grad(g: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return g - a


def full_objective(g: torch.Tensor, a: torch.Tensor, lam: float) -> torch.Tensor:
    return smooth_objective(g, a) + lam * torch.sum(torch.abs(g))


def run_subgradient_descent(
    a: torch.Tensor,
    lam: float,
    lr: float,
    num_steps: int,
    g0: torch.Tensor,
):
    g = g0.clone()

    history = {
        "step": [],
        "objective": [],
        "nonzero_count": [],
        "g": [],
    }

    for t in range(num_steps + 1):
        history["step"].append(t)
        history["objective"].append(float(full_objective(g, a, lam).item()))
        history["nonzero_count"].append(int((torch.abs(g) > 1e-8).sum().item()))
        history["g"].append(g.detach().cpu().clone())

        if t == num_steps:
            break

        grad = smooth_grad(g, a) + lam * l1_subgradient(g)
        g = g - lr * grad

    return history


def run_proximal_gradient(
    a: torch.Tensor,
    lam: float,
    lr: float,
    num_steps: int,
    g0: torch.Tensor,
):
    g = g0.clone()

    history = {
        "step": [],
        "objective": [],
        "nonzero_count": [],
        "g": [],
    }

    for t in range(num_steps + 1):
        history["step"].append(t)
        history["objective"].append(float(full_objective(g, a, lam).item()))
        history["nonzero_count"].append(int((torch.abs(g) > 1e-8).sum().item()))
        history["g"].append(g.detach().cpu().clone())

        if t == num_steps:
            break

        grad_smooth = smooth_grad(g, a)
        u = g - lr * grad_smooth
        g = soft_threshold(u, lr * lam)

    return history


def plot_histories(sub_hist, prox_hist, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    steps = sub_hist["step"]

    plt.figure()
    plt.plot(steps, sub_hist["objective"], marker="o", label="SGD + l1 subgradient")
    plt.plot(steps, prox_hist["objective"], marker="o", label="Proximal gradient")
    plt.xlabel("Step")
    plt.ylabel("Objective")
    plt.title("Objective vs Step (Torch)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "torch_objective_vs_step.png"))
    plt.close()

    plt.figure()
    plt.plot(steps, sub_hist["nonzero_count"], marker="o", label="SGD + l1 subgradient")
    plt.plot(steps, prox_hist["nonzero_count"], marker="o", label="Proximal gradient")
    plt.xlabel("Step")
    plt.ylabel("Number of nonzero coordinates")
    plt.title("Sparsity pattern vs Step (Torch)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "torch_nonzero_vs_step.png"))
    plt.close()

    g_sub = torch.stack(sub_hist["g"], dim=0).numpy()
    g_prox = torch.stack(prox_hist["g"], dim=0).numpy()

    for name, gs in [("subgrad", g_sub), ("prox", g_prox)]:
        plt.figure()
        for i in range(gs.shape[1]):
            plt.plot(steps, gs[:, i], marker="o", label=f"g[{i}]")
        plt.xlabel("Step")
        plt.ylabel("Coordinate value")
        plt.title(f"Coordinate trajectories ({name}, Torch)")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"torch_{name}_coords_vs_step.png"))
        plt.close()


def main():
    out_dir = "outputs_l1_compare_torch"
    os.makedirs(out_dir, exist_ok=True)

    a = torch.tensor([2.0, 0.7, 0.25, -0.15, -1.8, 0.02], dtype=torch.float64)
    lam = 0.5
    lr = 0.2
    num_steps = 50
    g0 = torch.tensor([3.0, 2.0, 1.0, -1.0, -2.0, 0.5], dtype=torch.float64)

    sub_hist = run_subgradient_descent(a=a, lam=lam, lr=lr, num_steps=num_steps, g0=g0)
    prox_hist = run_proximal_gradient(a=a, lam=lam, lr=lr, num_steps=num_steps, g0=g0)

    closed_form = soft_threshold(a, lam)

    summary = {
        "a": a.tolist(),
        "lambda": lam,
        "learning_rate": lr,
        "num_steps": num_steps,
        "initial_g": g0.tolist(),
        "closed_form_optimum": closed_form.tolist(),
        "subgradient_final_g": sub_hist["g"][-1].tolist(),
        "proximal_final_g": prox_hist["g"][-1].tolist(),
        "subgradient_final_objective": sub_hist["objective"][-1],
        "proximal_final_objective": prox_hist["objective"][-1],
        "subgradient_final_nonzero_count": sub_hist["nonzero_count"][-1],
        "proximal_final_nonzero_count": prox_hist["nonzero_count"][-1],
    }

    with open(os.path.join(out_dir, "torch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_histories(sub_hist, prox_hist, out_dir)

    print("=" * 80)
    print("TORCH COMPARISON DONE")
    print("Closed-form optimum         :", closed_form)
    print("Subgradient final g         :", sub_hist["g"][-1])
    print("Proximal final g            :", prox_hist["g"][-1])
    print("Subgradient final objective :", sub_hist["objective"][-1])
    print("Proximal final objective    :", prox_hist["objective"][-1])
    print(f"Saved outputs in            : {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()