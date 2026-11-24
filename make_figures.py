"""Create the analysis figures used in the README.

Run after evaluate.py (final experiment) and after the multi-seed runs:
    python3 make_figures.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np

import aggregate_seeds
from aggregate_seeds import load_runs, mean_std
from config import FIGURES_DIR, FINAL_EXPERIMENT, OUTPUTS_DIR

MODEL_STYLES = {
    "Persistence": ("#9ca3af", "-"),
    "Seasonal persistence (24h)": ("#a78bfa", "-"),
    "Climatology (month x hour)": ("#34d399", "-"),
    "Ridge regression": ("#2e86de", "--"),
    "LSTM": ("#e4572e", "-"),
}

ABLATION_ORDER = ["baseline", "time", "time_residual", "full"]
ABLATION_LABELS = {
    "baseline": "Baseline\nLSTM",
    "time": "+ time\nfeatures",
    "time_residual": "+ time\n+ residual",
    "full": "+ time + residual\n+ regularization",
}


def plot_horizon_errors():
    path = OUTPUTS_DIR / f"metrics_{FINAL_EXPERIMENT}.json"

    with open(path, encoding="utf-8") as f:
        models = json.load(f)["models"]

    hours = np.arange(1, 13)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, (color, style) in MODEL_STYLES.items():
        if name not in models:
            continue

        lw = 3 if name == "LSTM" else 1.8
        ax.plot(hours, models[name]["mae_per_horizon"], color=color,
                linestyle=style, linewidth=lw, marker="o", markersize=4,
                label=name)

    ax.set_xticks(hours)
    ax.set_xticklabels([f"h+{h}" for h in hours])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Mean absolute error (°C)")
    ax.set_title("Test error by forecast horizon (2024–2025)",
                 fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    out = FIGURES_DIR / "error_by_horizon.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_ablation():
    names, mae_m, mae_s, h1_m, h1_s = [], [], [], [], []

    for name in ABLATION_ORDER:
        runs = load_runs(name)

        if len(runs) < 2:
            continue

        m, s = mean_std([r["mae"] for r in runs])
        a, b = mean_std([r["mae_per_horizon"][0] for r in runs])

        names.append(name)
        mae_m.append(m); mae_s.append(s)
        h1_m.append(a); h1_s.append(b)

    if not names:
        print("Skipping ablation figure: need >= 2 seeds per experiment.")
        return

    labels = [ABLATION_LABELS[n] for n in names]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, means, stds, title in (
        (axes[0], mae_m, mae_s, "Overall test MAE"),
        (axes[1], h1_m, h1_s, "MAE at h+1"),
    ):
        ax.bar(x, means, yerr=stds, capsize=5, color="#e4572e", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("MAE (°C)")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(min(means) * 0.8, max(means) * 1.08)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Ablation study (mean ± std over seeds)",
                 fontsize=14, fontweight="bold")

    out = FIGURES_DIR / "ablation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_horizon_errors()
    plot_ablation()