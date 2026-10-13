"""Build the paper-style summary table and bar chart: mean ± std accuracy per method and dataset.

Reads ``results/<method>/dataset_N/accuracies.csv`` (written by generate_results.py)
and writes ``results/summary_table.csv``, ``results/summary_table.md`` and the
Figure 7-style grouped bar chart ``plots/summary_accuracy.png``.
The std is the population std (ddof=0), which reproduces the paper's tables.
"All data" pools every subject (and every Dataset 2 class pair), like Figure 7.

Run from the project root:
    python src/results_generation/summary_table.py
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

METHODS = {
    "CSP": "csp",
    "Wavelet-CSP": "wavelet_csp",
    "WPD-CSP": "wpd_csp",
    "SFBCSP": "sfbcsp",
    "SBLFB": "sblfb",
    "CSP-Wavelet+LOG": "csp_wavelet_log",
    "CSP-WPD+LOG": "csp_wpd_log",
    "CSP-FB+LOG": "csp_fb_log",
}
# folder -> (column label, rows expected under the paper's protocol)
DATASETS = {
    "dataset_1": ("Dataset 1", 7),
    "dataset_2": ("Dataset 2", 54),
    "dataset_3": ("Dataset 3", 9),
    "dataset_4": ("Dataset 4", 14),
}

# Categorical palette in fixed method order (validated for colour-vision deficiencies).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _accuracies(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        return np.array([float(row["accuracy_percent"]) for row in csv.DictReader(handle)])


def _cell(scores: np.ndarray) -> str:
    return f"{scores.mean():.2f} ± {scores.std():.2f}" if scores.size else "n/a"


def _stat(scores: np.ndarray):
    return (scores.mean(), scores.std()) if scores.size else (np.nan, np.nan)


def build_table(results_dir: Path):
    """Return (rows of formatted cells, {method: [(mean, std) per column]}, warnings)."""
    rows, stats, warnings = [], {}, []
    for method, folder in METHODS.items():
        row, pooled, stats[method] = {"Method": method}, [], []
        for dataset, (label, expected) in DATASETS.items():
            path = results_dir / folder / dataset / "accuracies.csv"
            scores = _accuracies(path) if path.exists() else np.array([])
            if not path.exists():
                warnings.append(f"{method} / {label}: {path} missing")
            elif len(scores) != expected:
                warnings.append(f"{method} / {label}: {len(scores)} rows, expected {expected}")
            row[label] = _cell(scores)
            stats[method].append(_stat(scores))
            pooled.append(scores)
        everything = np.concatenate(pooled)
        row["All data"] = _cell(everything)
        stats[method].append(_stat(everything))
        rows.append(row)
    return rows, stats, warnings


def plot_summary(stats, groups, output_path: Path) -> None:
    """Grouped bars (one group per dataset + All data), one bar per method, ±1 std whiskers."""
    methods = list(stats)
    width = 0.8 / len(methods)
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(12, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, method in enumerate(methods):
        means = np.array([mean for mean, _ in stats[method]])
        stds = np.array([std for _, std in stats[method]])
        ax.bar(x - 0.4 + (i + 0.5) * width, means, width * 0.92, yerr=stds, label=method,
               color=SERIES_COLORS[i % len(SERIES_COLORS)], edgecolor=SURFACE, linewidth=0.6,
               error_kw=dict(ecolor=TEXT_SECONDARY, elinewidth=0.8, capsize=2, capthick=0.8))
    ax.set_xticks(x, groups, color=TEXT)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%), mean ± std", color=TEXT_SECONDARY)
    ax.set_title("Average classification accuracy by dataset", color=TEXT, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--plot-path", type=Path, default=ROOT / "plots" / "summary_accuracy.png")
    args = parser.parse_args()

    rows, stats, warnings = build_table(args.results_dir)
    columns = ["Method", *(label for label, _ in DATASETS.values()), "All data"]

    csv_path = args.results_dir / "summary_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(row[c] for c in columns) + " |" for row in rows]
    md_path = args.results_dir / "summary_table.md"
    md_path.write_text(
        "Mean ± std classification accuracy (%), population std as in the paper.\n\n"
        + "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    plot_summary(stats, columns[1:], args.plot_path)

    print("\n".join(lines))
    for warning in warnings:
        print(f"[warn] {warning}")
    print(f"\nSaved: {csv_path}\nSaved: {md_path}\nSaved: {args.plot_path}")


if __name__ == "__main__":
    main()
