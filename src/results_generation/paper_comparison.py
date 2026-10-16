"""Compare this repository's results with the published results of Zhang et al. (2020).

Inputs: per-subject accuracies in ``results/<method>/dataset_N/accuracies.csv``, the
ablation cache of ablation_tables.py, and the paper's numbers in paper_values.py.

Outputs:
  results/paper_comparison/summary_comparison.{md,csv}     mean ± std, paper vs ours, per dataset
  results/paper_comparison/ablation_comparison.{md,csv}    Tables 6-9, paper vs ours
  results/paper_comparison/per_subject_comparison.csv      every subject / class pair
  plots/paper_comparison/accuracy_paper_vs_ours.png         paper vs ours per method, by dataset
  plots/paper_comparison/difference_heatmap.png             ours - paper, methods x datasets
  plots/paper_comparison/per_subject_scatter.png            ours vs paper per subject (as Fig. 6)
  plots/paper_comparison/ablation_paper_vs_ours.png         Tables 6-9 "All data", paper vs ours

Run from the project root:
    python src/results_generation/paper_comparison.py
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.results_generation import paper_values as paper
from src.results_generation.ablation_tables import load_results

RESULTS = PROJECT_ROOT / "results"
OUT = RESULTS / "paper_comparison"
PLOTS = PROJECT_ROOT / "plots" / "paper_comparison"

FOLDERS = {
    "CSP": "csp", "Wavelet-CSP": "wavelet_csp", "WPD-CSP": "wpd_csp", "SFBCSP": "sfbcsp",
    "SBLFB": "sblfb", "CSP-Wavelet+LOG": "csp_wavelet_log", "CSP-WPD+LOG": "csp_wpd_log",
    "CSP-FB+LOG": "csp_fb_log",
}
DATASETS = {"dataset_1": "Dataset 1", "dataset_2": "Dataset 2", "dataset_3": "Dataset 3", "dataset_4": "Dataset 4"}
ABLATION_KEYS = {"dataset_1": "1", "dataset_2": "2a", "dataset_3": "2b", "dataset_4": "4"}

PAPER_COLOR, OURS_COLOR = "#8a8984", "#2a78d6"
DATASET_STYLE = {"dataset_1": ("#2a78d6", "o"), "dataset_2": ("#eb6834", "s"),
                 "dataset_3": ("#1baf7a", "^"), "dataset_4": ("#eda100", "D")}
TEXT, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------
def load_ours(method, dataset):
    """{unit: accuracy} from results/<method>/<dataset>/accuracies.csv."""
    path = RESULTS / FOLDERS[method] / dataset / "accuracies.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["subject"] + (f"_{row['pair']}" if row["pair"] else ""): float(row["accuracy_percent"])
                for row in csv.DictReader(handle)}


def paired_scores():
    """{(method, dataset): (units, paper array, ours array)} on the units both share."""
    pairs = {}
    for m_index, method in enumerate(paper.METHODS):
        for dataset in DATASETS:
            published = {unit: values[m_index] for unit, values in paper.PER_SUBJECT[dataset].items()}
            ours = load_ours(method, dataset)
            units = [u for u in published if u in ours]
            missing = sorted(set(published) - set(ours))
            if missing:
                print(f"[warn] {method} / {dataset}: no result for {', '.join(missing)}")
            pairs[(method, dataset)] = (units, np.array([published[u] for u in units]),
                                        np.array([ours[u] for u in units]))
    return pairs


def _pool(pairs, method):
    units, p, o = [], [], []
    for dataset in DATASETS:
        u, a, b = pairs[(method, dataset)]
        units += u
        p.append(a)
        o.append(b)
    return units, np.concatenate(p), np.concatenate(o)


def ablation_ours():
    """Tables 6-9 values from the ablation cache, keyed like paper_values.TABLE_6..9."""
    collected = load_results(list(ABLATION_KEYS.values()))

    def mean(extractor, metric):
        per_ds = [np.mean(collected[(k, extractor)][metric]) for k in ABLATION_KEYS.values()]
        pooled = np.mean(sum((collected[(k, extractor)][metric] for k in ABLATION_KEYS.values()), []))
        return per_ds + [pooled]

    ours = {}
    for table, selector in ((paper.TABLE_6, "LASSO"), (paper.TABLE_7, "LOG")):
        for (method, mode) in table:
            ours[(method, mode)] = mean(method.split("+")[0], f"{selector}_{mode}")
    metric_8 = {"F-score": "F-score", "GA": "GA", "BPSO": "BPSO", "LASSO": "LASSO_with", "LOG": "LOG_with"}
    for (extractor, selector) in paper.TABLE_8:
        ours[(extractor, selector)] = mean(extractor, metric_8[selector])
    metric_9 = {"SVM": "LOG_SVM", "BLDA": "LOG_BLDA", "FLDA": "LOG_with"}
    for (method, classifier) in paper.TABLE_9:
        ours[(method, classifier)] = mean(method.split("+")[0], metric_9[classifier])
    return ours


# ----------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------
def _fmt(values):
    return f"{np.mean(values):.2f} ± {np.std(values):.2f}"


def write_summary(pairs):
    columns = [*DATASETS.values(), "All data"]
    rows_md, rows_csv = [], []
    for method in paper.METHODS:
        cells, record = [method], {"Method": method}
        groups = [pairs[(method, d)] for d in DATASETS] + [_pool(pairs, method)]
        for label, (_, p, o) in zip(columns, groups):
            delta = o.mean() - p.mean()
            cells.append(f"{_fmt(o)} / {_fmt(p)} ({delta:+.2f})")
            record.update({f"{label} paper mean": round(p.mean(), 2), f"{label} paper std": round(p.std(), 2),
                           f"{label} ours mean": round(o.mean(), 2), f"{label} ours std": round(o.std(), 2),
                           f"{label} difference": round(delta, 2)})
        _, p, o = _pool(pairs, method)
        r = float(np.corrcoef(p, o)[0, 1])
        mad = float(np.mean(np.abs(o - p)))
        better = int(np.sum(o > p))
        cells += [f"{r:.2f}", f"{mad:.2f}", f"{better} / {len(o)}"]
        record.update({"subject-level r": round(r, 3), "mean |difference|": round(mad, 2),
                       "units where ours > paper": better, "units": len(o)})
        rows_md.append(cells)
        rows_csv.append(record)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "summary_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_csv[0]))
        writer.writeheader()
        writer.writerows(rows_csv)
    header = ["Method", *columns, "Subject-level r", "Mean abs. Δ", "Ours > paper"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows_md]
    text = ("### Mean ± std accuracy (%): ours / paper (ours − paper)\n\n"
            "Subject-level r, mean absolute difference and the count of units where ours is higher are computed over all "
            "84 units (subjects and Dataset 2 class pairs).\n\n" + "\n".join(lines) + "\n")
    (OUT / "summary_comparison.md").write_text(text, encoding="utf-8")
    return text


def write_ablation(ours):
    sections, records = [], []
    tables = (("Table 6 (LASSO, secondary selection)", paper.TABLE_6, lambda k: f"{k[0]} ({k[1]})"),
              ("Table 7 (LOG, secondary selection)", paper.TABLE_7, lambda k: f"{k[0]} ({k[1]})"),
              ("Table 8 (feature selection)", paper.TABLE_8, lambda k: f"{k[0]} + {k[1]}"),
              ("Table 9 (classifier)", paper.TABLE_9, lambda k: f"{k[0]} {k[1]}"))
    header = ["Setting", *paper.ROWS]
    for title, table, name in tables:
        lines = [f"### {title}: ours / paper (ours − paper)", "",
                 "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for key, published in table.items():
            mine = ours[key]
            cells = [f"{o:.2f} / {p:.2f} ({o - p:+.2f})" for o, p in zip(mine, published)]
            lines.append("| " + " | ".join([name(key), *cells]) + " |")
            for row, o, p in zip(paper.ROWS, mine, published):
                records.append({"table": title, "setting": name(key), "row": row,
                                "paper": p, "ours": round(o, 2), "difference": round(o - p, 2)})
        sections.append("\n".join(lines))
    with (OUT / "ablation_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    text = "\n\n".join(sections) + "\n"
    (OUT / "ablation_comparison.md").write_text(text, encoding="utf-8")
    return text


def write_per_subject(pairs):
    with (OUT / "per_subject_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "unit", "method", "paper", "ours", "difference"])
        for (method, dataset), (units, p, o) in pairs.items():
            for unit, a, b in zip(units, p, o):
                writer.writerow([DATASETS[dataset], unit, method, a, round(b, 2), round(b - a, 2)])


# ----------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_dots(pairs):
    """Paper (grey) and ours (blue) mean per method, joined by a line, one panel per dataset."""
    groups = [*DATASETS, "all"]
    titles = [*DATASETS.values(), "All data"]
    fig, axes = plt.subplots(1, 5, figsize=(17, 5.2), sharey=True, facecolor="white")
    y = np.arange(len(paper.METHODS))[::-1]
    for ax, group, title in zip(axes, groups, titles):
        for yi, method in zip(y, paper.METHODS):
            _, p, o = _pool(pairs, method) if group == "all" else pairs[(method, group)]
            ax.plot([p.mean(), o.mean()], [yi, yi], color=GRID, lw=2.5, zorder=1)
            ax.scatter(p.mean(), yi, s=60, color=PAPER_COLOR, zorder=2, label="Paper")
            ax.scatter(o.mean(), yi, s=60, color=OURS_COLOR, zorder=3, label="Ours")
        ax.set_title(title, color=TEXT, fontsize=12)
        ax.set_xlim(60, 92)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xlabel("Mean accuracy (%)", color=TEXT_SECONDARY)
    axes[0].set_yticks(y, paper.METHODS, color=TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Mean accuracy per method: paper vs this implementation", x=0.01, ha="left",
                 color=TEXT, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / "accuracy_paper_vs_ours.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_heatmap(pairs):
    """Ours − paper mean accuracy; blue = ours higher, red = ours lower."""
    cols = [*DATASETS.values(), "All data"]
    delta = np.array([[pairs[(m, d)][2].mean() - pairs[(m, d)][1].mean() for d in DATASETS]
                      + [_pool(pairs, m)[2].mean() - _pool(pairs, m)[1].mean()] for m in paper.METHODS])
    limit = np.ceil(np.abs(delta).max())
    fig, ax = plt.subplots(figsize=(8.6, 5.4), facecolor="white")
    image = ax.imshow(delta, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            value = delta[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(value) > 0.6 * limit else TEXT)
    ax.set_xticks(range(len(cols)), cols, color=TEXT)
    ax.set_yticks(range(len(paper.METHODS)), paper.METHODS, color=TEXT)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(paper.METHODS)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("Ours − paper (percentage points)", color=TEXT_SECONDARY)
    bar.outline.set_visible(False)
    ax.set_title("Difference in mean accuracy (ours − paper)", loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "difference_heatmap.png", dpi=200, facecolor="white")
    plt.close(fig)


def plot_scatter(pairs):
    """Per-subject accuracy, ours (y) against paper (x), one panel per method (cf. paper Fig. 6)."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.2), facecolor="white")
    for ax, method in zip(axes.ravel(), paper.METHODS):
        for dataset, (color, marker) in DATASET_STYLE.items():
            _, p, o = pairs[(method, dataset)]
            ax.scatter(p, o, s=22, color=color, marker=marker, alpha=0.85, linewidths=0,
                       label=DATASETS[dataset])
        ax.plot([40, 100], [40, 100], color=TEXT_SECONDARY, lw=0.9, ls="--", zorder=0)
        _, p, o = _pool(pairs, method)
        ax.text(0.04, 0.96, f"r = {np.corrcoef(p, o)[0, 1]:.2f}\nmean |Δ| = {np.mean(np.abs(o - p)):.1f}",
                transform=ax.transAxes, va="top", fontsize=9, color=TEXT_SECONDARY)
        ax.set_xlim(40, 101)
        ax.set_ylim(40, 101)
        ax.set_aspect("equal")
        ax.set_title(method, color=TEXT, fontsize=11)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    for ax in axes[1]:
        ax.set_xlabel("Paper accuracy (%)", color=TEXT_SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("Our accuracy (%)", color=TEXT_SECONDARY)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Per-subject accuracy: points above the dashed line are higher in this implementation",
                 x=0.01, ha="left", color=TEXT, fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(PLOTS / "per_subject_scatter.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ablation(ours):
    """Tables 6-9 "All data": paper (grey) vs ours (blue), one panel per table."""
    tables = (("Table 6: LASSO", paper.TABLE_6, lambda k: f"{k[0].split('+')[0]} ({k[1]})"),
              ("Table 7: LOG", paper.TABLE_7, lambda k: f"{k[0].split('+')[0]} ({k[1]})"),
              ("Table 8: feature selection", paper.TABLE_8, lambda k: f"{k[0]} + {k[1]}"),
              ("Table 9: classifier", paper.TABLE_9, lambda k: f"{k[0].split('+')[0]} {k[1]}"))
    heights = [len(t[1]) for t in tables]
    fig, axes = plt.subplots(len(tables), 1, figsize=(8.5, 0.34 * sum(heights) + 3.2), facecolor="white",
                             gridspec_kw={"height_ratios": heights}, sharex=True)
    for ax, (title, table, name) in zip(axes, tables):
        keys = list(table)
        y = np.arange(len(keys))[::-1]
        for yi, key in zip(y, keys):
            p, o = table[key][-1], ours[key][-1]
            ax.plot([p, o], [yi, yi], color=GRID, lw=2.5, zorder=1)
            ax.scatter(p, yi, s=45, color=PAPER_COLOR, zorder=2, label="Paper")
            ax.scatter(o, yi, s=45, color=OURS_COLOR, zorder=3, label="Ours")
        ax.set_yticks(y, [name(k) for k in keys], color=TEXT, fontsize=9)
        ax.set_title(title, loc="left", color=TEXT, fontsize=11)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[-1].set_xlabel("All data mean accuracy (%)", color=TEXT_SECONDARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper right", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "ablation_paper_vs_ours.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    sys.stdout.reconfigure(errors="replace")  # Windows consoles cannot print "−" or "Δ"
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    pairs = paired_scores()
    ours_ablation = ablation_ours()
    print(write_summary(pairs))
    print(write_ablation(ours_ablation))
    write_per_subject(pairs)
    plot_dots(pairs)
    plot_heatmap(pairs)
    plot_scatter(pairs)
    plot_ablation(ours_ablation)
    print(f"Saved tables to {OUT}\nSaved plots to {PLOTS}")


if __name__ == "__main__":
    main()
