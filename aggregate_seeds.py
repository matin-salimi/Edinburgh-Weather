"""Report mean +/- std of the LSTM metrics over all available seeds."""

import json
import statistics

from config import EXPERIMENTS, OUTPUTS_DIR


def load_runs(name):
    files = [OUTPUTS_DIR / f"metrics_{name}.json"]
    files += sorted(OUTPUTS_DIR.glob(f"metrics_{name}_s*.json"))

    runs = []

    for path in files:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f)["models"]["LSTM"])

    return runs


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values)


def main():
    print(f"{'Experiment':<15}{'n':>3}{'MAE':>16}{'RMSE':>16}"
          f"{'h+1 MAE':>16}{'h+12 MAE':>16}")

    found = False

    for name in EXPERIMENTS:
        runs = load_runs(name)

        if len(runs) < 2:
            continue

        found = True

        columns = [
            [r["mae"] for r in runs],
            [r["rmse"] for r in runs],
            [r["mae_per_horizon"][0] for r in runs],
            [r["mae_per_horizon"][11] for r in runs],
        ]

        cells = "".join(
            f"{m:>9.3f} ± {s:.3f}" for m, s in map(mean_std, columns)
        )

        print(f"{name:<15}{len(runs):>3}{cells}")

    if not found:
        raise SystemExit("Need at least 2 seeds per experiment. Run more seeds.")


if __name__ == "__main__":
    main()