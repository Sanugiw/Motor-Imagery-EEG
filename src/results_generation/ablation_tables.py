"""Reproduce the paper's Tables 6-9 (secondary selection, selector and classifier comparisons).

For every subject (and Dataset 2 class pair) and each proposed feature extractor
(CSP-Wavelet, CSP-WPD, CSP-FB) this fits LASSO and LOG once and evaluates:
  * Table 6/7: FLDA on all LASSO/LOG-selected features (|w| > 0, "without secondary
    feature selection") vs. the threshold ensemble's best model ("with").
  * Table 8:   F-score, GA and BPSO selection + FLDA, next to LASSO and LOG.
  * Table 9:   the LOG threshold ensemble with linear SVM, BLDA or FLDA.
Per-subject results are cached in ``results/ablation/cache`` so an interrupted run
resumes; tables go to ``results/ablation/``. "All data" pools every subject.

Run from the project root:
    python src/results_generation/ablation_tables.py            # all datasets
    python src/results_generation/ablation_tables.py --jobs 4 --dataset 4
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifiers.blda import BLDA
from src.classifiers.ensemble_flda import ThresholdEnsembleFLDA
from src.data.data_loader import iter_dataset_trials
from src.features.csp_fb import extract_train_test as csp_fb
from src.features.csp_wavelet import extract_train_test as csp_wavelet
from src.features.csp_wpd import extract_train_test as csp_wpd
from src.feature_selection.lasso_selection import LassoFeatureSelector
from src.feature_selection.log_selection import LogFeatureSelector
from src.feature_selection.wrapper_selection import BPSOSelector, FScoreSelector, GASelector

EXTRACTORS = {"CSP-Wavelet": csp_wavelet, "CSP-WPD": csp_wpd, "CSP-FB": csp_fb}
DATASETS = {"1": "Dataset 1", "2a": "Dataset 2", "2b": "Dataset 3", "4": "Dataset 4"}
OUT = PROJECT_ROOT / "results" / "ablation"


def _flda_on(support, x_train, y_train, x_test, y_test):
    if not np.any(support):
        return 0.0
    model = LinearDiscriminantAnalysis(solver="svd").fit(x_train[:, support], y_train)
    return 100.0 * accuracy_score(y_test, model.predict(x_test[:, support]))


def _ensemble(selector, x_train, y_train, x_test, y_test, make_model=None):
    ensemble = ThresholdEnsembleFLDA(make_model=make_model).fit(
        selector.transform(x_train), y_train, selector.coef_)
    return ensemble.evaluate(selector.transform(x_test), y_test)


def evaluate_unit(trial, extractor_name):
    """All Table 6-9 accuracies (%) for one subject/pair and one feature extractor."""
    y_tr, y_te = trial.y_train, trial.y_test
    x_tr, x_te, _ = EXTRACTORS[extractor_name](trial.X_train, y_tr, trial.X_test, trial.fs)
    out = {}
    for name, selector in (("LASSO", LassoFeatureSelector()), ("LOG", LogFeatureSelector())):
        selector.fit(x_tr, y_tr)
        result = _ensemble(selector, x_tr, y_tr, x_te, y_te)
        out[f"{name}_without"] = 100.0 * result["threshold_results"][0]["accuracy"]  # |w| > 0 only
        out[f"{name}_with"] = 100.0 * result["paper_score"]
        if name == "LOG":
            out["LOG_SVM"] = 100.0 * _ensemble(selector, x_tr, y_tr, x_te, y_te,
                                               lambda: SVC(kernel="linear", C=1.0))["paper_score"]
            out["LOG_BLDA"] = 100.0 * _ensemble(selector, x_tr, y_tr, x_te, y_te, BLDA)["paper_score"]
    for name, selector in (("F-score", FScoreSelector()), ("GA", GASelector()), ("BPSO", BPSOSelector())):
        out[name] = _flda_on(selector.fit(x_tr, y_tr).get_support(), x_tr, y_tr, x_te, y_te)
    return out


def _cache_path(dataset, trial, extractor_name):
    unit = trial.subject + (f"_{trial.pair}" if trial.pair else "")
    return OUT / "cache" / f"dataset_{dataset}" / extractor_name / f"{unit}.json"


def _run_task(dataset, trial, extractor_name):
    path = _cache_path(dataset, trial, extractor_name)
    if path.exists():
        return
    result = evaluate_unit(trial, extractor_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    print(f"done {path.relative_to(OUT / 'cache')}", flush=True)


def load_results(datasets):
    """{(dataset, extractor): {metric: [per-unit accuracies]}} from the cache."""
    collected = {}
    for dataset in datasets:
        for extractor_name in EXTRACTORS:
            folder = OUT / "cache" / f"dataset_{dataset}" / extractor_name
            per_metric = {}
            for path in sorted(folder.glob("*.json")):
                for metric, value in json.loads(path.read_text(encoding="utf-8")).items():
                    per_metric.setdefault(metric, []).append(value)
            collected[(dataset, extractor_name)] = per_metric
    return collected


def _mean(collected, datasets, extractor_name, metric, dataset=None):
    keys = [dataset] if dataset else datasets  # None -> pool all datasets ("All data")
    values = sum((collected[(d, extractor_name)].get(metric, []) for d in keys), [])
    return f"{np.mean(values):.2f}" if values else "n/a"


def _write(name, title, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)


def build_tables(datasets):
    collected = load_results(datasets)
    labels = [DATASETS[d] for d in datasets]
    row_keys = [*datasets, None]
    row_labels = [*labels, "All data"]
    sections = []

    for table, selector in (("table6_lasso_secondary", "LASSO"), ("table7_log_secondary", "LOG")):
        header = ["Datasets", *(f"{e}+{selector} (without)" for e in EXTRACTORS),
                  *(f"{e}+{selector} (with)" for e in EXTRACTORS)]
        rows = [[label, *(_mean(collected, datasets, e, f"{selector}_{mode}", d)
                          for mode in ("without", "with") for e in EXTRACTORS)]
                for label, d in zip(row_labels, row_keys)]
        number = 6 if selector == "LASSO" else 7
        sections.append(_write(table, f"Table {number}. Effect of secondary feature selection ({selector})",
                               header, rows))

    header = ["Feature extraction", "Feature selection", *labels, "All data"]
    rows = []
    for e in EXTRACTORS:
        for selector, metric in (("F-score", "F-score"), ("GA", "GA"), ("BPSO", "BPSO"),
                                 ("LASSO", "LASSO_with"), ("LOG", "LOG_with")):
            rows.append([e, selector, *(_mean(collected, datasets, e, metric, d) for d in row_keys)])
    sections.append(_write("table8_feature_selection", "Table 8. Effect of feature selection method",
                           header, rows))

    header = ["Datasets", *(f"{e}+LOG {c}" for e in EXTRACTORS for c in ("SVM", "BLDA", "FLDA"))]
    rows = [[label, *(_mean(collected, datasets, e, metric, d) for e in EXTRACTORS
                      for metric in ("LOG_SVM", "LOG_BLDA", "LOG_with"))]
            for label, d in zip(row_labels, row_keys)]
    sections.append(_write("table9_classifiers", "Table 9. Effect of classifier (LOG selection)",
                           header, rows))

    text = "Average classification accuracy (%), as in the paper's Tables 6-9.\n\n" + "\n\n".join(sections) + "\n"
    (OUT / "ablation_tables.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="*", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--jobs", type=int, default=-2, help="Parallel workers (joblib; default all but one core).")
    parser.add_argument("--tables-only", action="store_true", help="Only rebuild tables from the cache.")
    args = parser.parse_args()

    if not args.tables_only:
        tasks = [(d, trial, e) for d in args.dataset for trial in iter_dataset_trials(d) for e in EXTRACTORS]
        todo = [t for t in tasks if not _cache_path(*t).exists()]
        print(f"{len(tasks)} subject x extractor tasks, {len(todo)} to compute", flush=True)
        Parallel(n_jobs=args.jobs, verbose=5)(delayed(_run_task)(*t) for t in todo)
    build_tables(args.dataset)


if __name__ == "__main__":
    main()
