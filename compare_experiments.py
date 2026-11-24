"""Print the ablation table from the saved metrics_<experiment>.json files."""

import json

from config import EXPERIMENTS, OUTPUTS_DIR


def main():
    rows = []

    for name in EXPERIMENTS:
        path = OUTPUTS_DIR / f"metrics_{name}.json"

        if not path.exists():
            continue

        with open(path, encoding="utf-8") as f:
            lstm = json.load(f)["models"]["LSTM"]

        rows.append((name, lstm))

    if not rows:
        raise SystemExit("No metrics files found. Run evaluate.py first.")

    reference = rows[0][1]["mae"]

    print(f"{'Experiment':<14}{'MAE':>8}{'RMSE':>8}{'h+1':>8}{'h+6':>8}"
          f"{'h+12':>8}{'dMAE vs first':>16}")

    for name, m in rows:
        per_h = m["mae_per_horizon"]
        delta = (m["mae"] / reference - 1) * 100
        print(
            f"{name:<14}{m['mae']:>8.3f}{m['rmse']:>8.3f}"
            f"{per_h[0]:>8.3f}{per_h[5]:>8.3f}{per_h[11]:>8.3f}"
            f"{delta:>+15.1f}%"
        )


if __name__ == "__main__":
    main()