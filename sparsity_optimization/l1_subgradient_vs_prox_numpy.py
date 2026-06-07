import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def l1_subgradient(x: np.ndarray) -> np.ndarray:
    """
    Choose the canonical subgradient:
        sign(x_i) for x_i != 0
        0         for x_i == 0
    """
    g = np.sign(x)
    g[np.abs(x) < 1e-12] = 0.0
    return g


def smooth_objective(g: np.ndarray, a: np.ndarray) -> float:
    return 0.5 * np.sum((g - a) ** 2)


def smooth_grad(g: np.ndarray, a: np.ndarray) -> np.ndarray:
    return g - a


def full_objective(g: np.ndarray, a: np.ndarray, lam: float) -> float:
    return smooth_objective(g, a) + lam * np.sum(np.abs(g))


def run_subgradient_descent(
    a: np.ndarray,
    lam: float,
    lr: float,
    num_steps: int,
    g0: np.ndarray,
):
    g = g0.copy()

    history = {
        "step": [],
        "objective": [],
        "smooth_objective": [],
        "l1_norm": [],
        "nonzero_count": [],
        "g": [],
    }

    for t in range(num_steps + 1):
        history["step"].append(t)
        history["objective"].append(full_objective(g, a, lam))
        history["smooth_objective"].append(smooth_objective(g, a))
        history["l1_norm"].append(float(np.sum(np.abs(g))))
        history["nonzero_count"].append(int(np.sum(np.abs(g) > 1e-8)))
        history["g"].append(g.copy())

        if t == num_steps:
            break

        grad = smooth_grad(g, a) + lam * l1_subgradient(g)
        g = g - lr * grad

    return history


def run_proximal_gradient(
    a: np.ndarray,
    lam: float,
    lr: float,
    num_steps: int,
    g0: np.ndarray,
):
    g = g0.copy()

    history = {
        "step": [],
        "objective": [],
        "smooth_objective": [],
        "l1_norm": [],
        "nonzero_count": [],
        "g": [],
    }

    for t in range(num_steps + 1):
        history["step"].append(t)
        history["objective"].append(full_objective(g, a, lam))
        history["smooth_objective"].append(smooth_objective(g, a))
        history["l1_norm"].append(float(np.sum(np.abs(g))))
        history["nonzero_count"].append(int(np.sum(np.abs(g) > 1e-8)))
        history["g"].append(g.copy())

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
    plt.title("Objective vs Step")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "numpy_objective_vs_step.png"))
    plt.close()

    plt.figure()
    plt.plot(steps, sub_hist["nonzero_count"], marker="o", label="SGD + l1 subgradient")
    plt.plot(steps, prox_hist["nonzero_count"], marker="o", label="Proximal gradient")
    plt.xlabel("Step")
    plt.ylabel("Number of nonzero coordinates")
    plt.title("Sparsity pattern vs Step")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "numpy_nonzero_vs_step.png"))
    plt.close()

    g_sub = np.stack(sub_hist["g"], axis=0)
    g_prox = np.stack(prox_hist["g"], axis=0)

    for name, gs in [("subgrad", g_sub), ("prox", g_prox)]:
        plt.figure()
        for i in range(gs.shape[1]):
            plt.plot(steps, gs[:, i], marker="o", label=f"g[{i}]")
        plt.xlabel("Step")
        plt.ylabel("Coordinate value")
        plt.title(f"Coordinate trajectories ({name})")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"numpy_{name}_coords_vs_step.png"))
        plt.close()


def main():
    out_dir = "outputs_l1_compare_numpy"
    os.makedirs(out_dir, exist_ok=True)

    # Toy problem:
    #   minimize 0.5 ||g - a||_2^2 + lam ||g||_1
    #
    # Closed-form optimum is soft_threshold(a, lam)
    a = np.array([2.0, 0.7, 0.25, -0.15, -1.8, 0.02], dtype=np.float64)
    lam = 0.5
    lr = 0.2
    num_steps = 50
    g0 = np.array([3.0, 2.0, 1.0, -1.0, -2.0, 0.5], dtype=np.float64)

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
        "subgradient_final_objective": float(sub_hist["objective"][-1]),
        "proximal_final_objective": float(prox_hist["objective"][-1]),
        "subgradient_final_nonzero_count": int(sub_hist["nonzero_count"][-1]),
        "proximal_final_nonzero_count": int(prox_hist["nonzero_count"][-1]),
    }

    with open(os.path.join(out_dir, "numpy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_histories(sub_hist, prox_hist, out_dir)

    print("=" * 80)
    print("NUMPY COMPARISON DONE")
    print("Closed-form optimum         :", closed_form)
    print("Subgradient final g         :", sub_hist["g"][-1])
    print("Proximal final g            :", prox_hist["g"][-1])
    print("Subgradient final objective :", sub_hist["objective"][-1])
    print("Proximal final objective    :", prox_hist["objective"][-1])
    print(f"Saved outputs in            : {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()