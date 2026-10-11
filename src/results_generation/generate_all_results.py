"""Generate classification results for every method and selected dataset."""

import argparse
import csv
from pathlib import Path

if __package__:
    from .result_runner import DATASETS, METHODS, ROOT, run_method
else:
    from result_runner import DATASETS, METHODS, ROOT, run_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    summaries = []
    total = len(METHODS) * len(datasets)
    completed = 0
    for dataset in datasets:
        for method in METHODS:
            completed += 1
            print(f"\n[{completed}/{total}] Running {method} on Dataset {dataset}", flush=True)
            _, summary = run_method(
                method,
                dataset,
                output_dir=args.output_dir,
                preprocessed_dir=args.preprocessed_dir,
                subject_limit=args.subject_limit,
            )
            summaries.append(summary)

    summary_path = args.output_dir / "all_methods_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "method",
        "dataset",
        "n_subjects",
        "mean_accuracy_percent",
        "std_accuracy_percent",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nSaved combined summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
