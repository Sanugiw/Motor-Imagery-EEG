"""Shared execution and output helpers for the per-method result scripts."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifiers.ensemble_flda import ThresholdEnsembleFLDA
from src.data.data_loader import ROOT, iter_dataset_trials
from src.features.csp_features import CSPFeatures
from src.features.csp_fb import extract_train_test as csp_fb
from src.features.csp_wavelet import extract_train_test as csp_wavelet
from src.features.csp_wpd import extract_train_test as csp_wpd
from src.features.sfbcsp import extract_filterbank_csp_features
from src.features.wavelet_csp import extract_train_test as wavelet_csp
from src.features.wpd_csp import extract_train_test as wpd_csp
from src.feature_selection.lasso_selection import LassoFeatureSelector
from src.feature_selection.log_selection import LogFeatureSelector
from src.feature_selection.sbl_selection import SparseBayesianSelector


METHODS = {
    "CSP": "csp",
    "Wavelet-CSP": "wavelet_csp",
    "WPD-CSP": "wpd_csp",
    "SFBCSP": "sfbcsp",
    "SBLFB": "sblfb",
    "CSP-Wavelet+LOG": "csp_wavelet_log",
    "CSP-WPD+LOG": "csp_wpd_log",
    "CSP-FB+LOG": "csp_fb_log",
    "CSP-Wavelet+LASSO": "csp_wavelet_lasso",
    "CSP-WPD+LASSO": "csp_wpd_lasso",
    "CSP-FB+LASSO": "csp_fb_lasso",
}

DATASETS = ("1", "2a", "2b", "4")
PAIR_ORDER = {name: i for i, name in enumerate(("LR", "LF", "LT", "RF", "RT", "FT"))}


def _flda_accuracy(X_train, y_train, X_test, y_test):
    if X_train.shape[1] == 0:
        return 0.0
    model = LinearDiscriminantAnalysis(solver="svd")
    model.fit(X_train, y_train)
    return float(accuracy_score(y_test, model.predict(X_test)))


def _selected_accuracy(features_train, y_train, features_test, y_test, selector):
    selector.fit(features_train, y_train)
    selected = selector.get_support()
    if not np.any(selected):
        return 0.0
    x_train = selector.transform(features_train)[:, selected]
    x_test = selector.transform(features_test)[:, selected]
    return _flda_accuracy(x_train, y_train, x_test, y_test)


def _proposed_accuracy(trials, extractor, selection):
    x_train, x_test, _ = extractor(
        trials.X_train, trials.y_train, trials.X_test, trials.fs
    )
    selector = (LassoFeatureSelector() if selection == "LASSO"
                else LogFeatureSelector())
    selector.fit(x_train, trials.y_train)
    scaled_train = selector.transform(x_train)
    scaled_test = selector.transform(x_test)
    ensemble = ThresholdEnsembleFLDA().fit(
        scaled_train, trials.y_train, selector.coef_
    )
    return ensemble.evaluate(scaled_test, trials.y_test)["paper_score"]


def evaluate(method, trials):
    if method == "CSP":
        csp = CSPFeatures(3).fit(trials.X_train, trials.y_train)
        return _flda_accuracy(
            csp.transform(trials.X_train), trials.y_train,
            csp.transform(trials.X_test), trials.y_test,
        )
    if method == "Wavelet-CSP":
        train, test, _ = wavelet_csp(
            trials.X_train, trials.y_train, trials.X_test, trials.fs
        )
        return _flda_accuracy(train, trials.y_train, test, trials.y_test)
    if method == "WPD-CSP":
        train, test, _ = wpd_csp(
            trials.X_train, trials.y_train, trials.X_test, trials.fs
        )
        return _flda_accuracy(train, trials.y_train, test, trials.y_test)
    if method in ("SFBCSP", "SBLFB"):
        train, test = extract_filterbank_csp_features(
            trials.X_train_raw, trials.y_train, trials.X_test_raw, trials.fs
        )
        selector = (LassoFeatureSelector() if method == "SFBCSP"
                    else SparseBayesianSelector())
        return _selected_accuracy(
            train, trials.y_train, test, trials.y_test, selector
        )

    parts = method.split("+")
    if len(parts) == 2 and parts[0] in ("CSP-Wavelet", "CSP-WPD", "CSP-FB"):
        extractors = {
            "CSP-Wavelet": csp_wavelet,
            "CSP-WPD": csp_wpd,
            "CSP-FB": csp_fb,
        }
        return _proposed_accuracy(trials, extractors[parts[0]], parts[1])
    raise ValueError(f"Unknown classification method: {method}")


def _sort_key(row):
    pair, subject = row["pair"], row["subject"]
    return (PAIR_ORDER.get(pair, len(PAIR_ORDER)), subject) if pair else (subject, "")


def _write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_method(method, dataset, output_dir=None, preprocessed_dir=None, subject_limit=None):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {tuple(METHODS)}")
    trials = list(iter_dataset_trials(dataset, preprocessed_dir))
    if subject_limit:
        subjects = list(dict.fromkeys(trial.subject for trial in trials))[:subject_limit]
        trials = [trial for trial in trials if trial.subject in subjects]
    if not trials:
        raise FileNotFoundError(f"No trials found for dataset {dataset}.")

    rows = []
    for index, trial in enumerate(trials, 1):
        name = trial.subject + (f"-{trial.pair}" if trial.pair else "")
        print(f"[{method} | dataset {dataset}] {index}/{len(trials)} {name}", flush=True)
        accuracy = 100.0 * evaluate(method, trial)
        rows.append({"subject": trial.subject, "pair": trial.pair,
                     "accuracy_percent": accuracy})
    rows.sort(key=_sort_key)
    scores = np.asarray([row["accuracy_percent"] for row in rows], dtype=float)
    summary = [{"method": method, "dataset": dataset, "n_subjects": len(rows),
                "mean_accuracy_percent": float(np.mean(scores)),
                "std_accuracy_percent": float(np.std(scores, ddof=0))}]

    root = Path(output_dir) if output_dir else ROOT / "results"
    folder = root / METHODS[method] / f"dataset_{dataset}"
    detail_path = folder / "accuracies.csv"
    summary_path = folder / "summary.csv"
    _write_csv(detail_path, rows, ["subject", "pair", "accuracy_percent"])
    _write_csv(summary_path, summary, ["method", "dataset", "n_subjects",
                                      "mean_accuracy_percent", "std_accuracy_percent"])
    print(f"Mean +/- Std accuracy: {np.mean(scores):.2f}% +/- {np.std(scores):.2f}%")
    print(f"Saved: {detail_path.resolve()}")
    print(f"Saved: {summary_path.resolve()}")
    return rows, summary[0]


def main(method):
    parser = argparse.ArgumentParser(description=f"Generate {method} classification accuracy results.")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--preprocessed-dir", type=Path,
                        help="Override the preprocessed-data directory for the selected dataset.")
    parser.add_argument("--subject-limit", type=int,
                        help="Run only the first N subjects for a short trial run.")
    args = parser.parse_args()
    chosen = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in chosen:
        run_method(method, dataset, args.output_dir,
                   args.preprocessed_dir, args.subject_limit)
