"""Generate classification results for every method or a selected method."""

import argparse
import csv
from pathlib import Path

if __package__:
    from .result_runner import DATASET_KEYS, DATASETS, METHODS, ROOT, run_method
else:
    from result_runner import DATASET_KEYS, DATASETS, METHODS, ROOT, run_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("all", *METHODS), default="all",
                        help="Method to run (default: all methods).")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        help="Override the preprocessed-data directory for the selected dataset.",
    )
    parser.add_argument(
        "--subject-limit",
        type=int,
        help="Run only the first N subjects per dataset for a short trial run.",
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    methods = tuple(METHODS) if args.method == "all" else (args.method,)
    summaries = []
    total = len(methods) * len(datasets)
    completed = 0
    for dataset in datasets:
        for method in methods:
            completed += 1
            print(f"\n[{completed}/{total}] Running {method} on {dataset}", flush=True)
            _, summary = run_method(
                method,
                dataset,
                output_dir=args.output_dir,
                preprocessed_dir=args.preprocessed_dir,
                subject_limit=args.subject_limit,
            )
            summaries.append(summary)

    if summaries:
        summary_path = args.output_dir / "all_methods_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "method",
            "dataset",
            "n_subjects",
            "mean_accuracy_percent",
            "std_accuracy_percent",
        ]
        # Combined summary uses the short dataset keys (1, 2a, 2b, 4).
        for summary in summaries:
            summary["dataset"] = DATASET_KEYS.get(summary["dataset"], summary["dataset"])
        # Replace only the (method, dataset) rows run this time; keep the rest.
        run_keys = {(summary["method"], summary["dataset"]) for summary in summaries}
        kept = []
        if summary_path.exists():
            with summary_path.open(newline="", encoding="utf-8") as handle:
                kept = [row for row in csv.DictReader(handle)
                        if (row["method"], row["dataset"]) not in run_keys]
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(kept + summaries)
        print(f"\nSaved combined summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
