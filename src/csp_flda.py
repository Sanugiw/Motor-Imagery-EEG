"""
Common Spatial Pattern (CSP) Transformation and FLDA Classification.
Paper: "The CSP-Based New Features Plus Non-Convex Log Sparse Feature Selection
    for Motor Imagery EEG Classification", Sensors 2020, 20(17), 4749.

Supports:
- Dataset 1: BCI Competition IV Dataset 1 (7 subjects, binary motor imagery)
- Dataset 2: BCI Competition IV Dataset 2a (9 subjects, 6 binary pairs: LR, LF, LT, RF, RT, FT = 54 subsets)
- Dataset 3: BCI Competition IV Dataset 2b (9 subjects, 3 channels C3/Cz/C4, left vs. right hand).
             The paper uses session 3 only (B0103T ... B0903T); the other sessions are optional.
- Dataset 4: Steyrl et al. (2016), 14 subjects, 15 channels, right hand vs. feet.

CSP Transformation:
- Normalized sample covariance matrices:
    C_bar_k = (1 / N_k) * sum_{n=1}^{N_k} [ D(k,n) D(k,n)^T / trace(D(k,n) D(k,n)^T) ],  k = 1, 2   (Eq. 2)
- Generalized eigenvalue problem:
    C_bar_2^{-1} C_bar_1 w = lambda w                                                                 (Eq. 3)
- Spatial filter matrix W:
    Eigenvectors sorted by descending eigenvalues, taking the first m and last m columns (m = 3).
    If 2m exceeds the number of channels (Dataset 3 has only 3 channels), the first-m and
    last-m selections overlap; every spatial filter is then used exactly once (see note below).
- Spatial projection:
    Z = W^T D                                                                                         (Eq. 4)
- Traditional CSP feature extraction (normalized log-variance):
    f_p = log( var(Z_p) / sum_{i=1}^{2m} var(Z_i) ),  p = 1, 2, ..., 2m                              (Eq. 5)

Classification:
- Fisher Linear Discriminant Analysis (FLDA) classifier trained on the training set
  and evaluated on the test set.

Note on Dataset 3 (3 channels, m = 3): taking "the first 3 and the last 3 columns" of a
3-column matrix selects every column twice. Duplicated features add no information (FLDA
on duplicated columns is equivalent to FLDA on the unique ones) and the doubled denominator
in Eq. 5 only shifts every feature by the same constant (-log 2). We therefore use the
3 distinct filters once each.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import scipy.linalg as la
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- Configuration ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if (PROJECT_ROOT / "preprocessed_data").exists():
    DEFAULT_BASE_DATA = PROJECT_ROOT / "preprocessed_data"
else:
    DEFAULT_BASE_DATA = PROJECT_ROOT / "processed_data"

DEFAULT_M = 3  # m = 3 pairs of spatial filters (total 2m = 6 filters)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

# CSP + FLDA test accuracies (%) reported in the paper (Table 4 / Table 5, "CSP" column).
# Printed next to our results as a sanity check; small deviations are expected.
PAPER_CSP_DS3 = {
    "B01": 77.50, "B02": 60.00, "B03": 46.25, "B04": 98.75, "B05": 88.75,
    "B06": 81.25, "B07": 81.25, "B08": 93.75, "B09": 81.25,
}
PAPER_CSP_DS4 = {
    "S01": 56.67, "S02": 85.00, "S03": 100.00, "S04": 85.00, "S05": 60.00,
    "S06": 66.67, "S07": 90.00, "S08": 81.67, "S09": 98.33, "S10": 53.33,
    "S11": 76.67, "S12": 81.67, "S13": 60.00, "S14": 51.67,
}
PAPER_CSP_DS3_MEAN_STD = (78.75, 15.47)
PAPER_CSP_DS4_MEAN_STD = (74.76, 15.93)


# ---------------- CSP Transformation Class ----------------

class CommonSpatialPattern:
    """
    Common Spatial Pattern (CSP) spatial filter design and feature extraction.

    Parameters
    ----------
    n_components : int, default=3
        The parameter 'm' from the paper. The final spatial filter W contains
        the first m and last m columns (total 2*m filters). If 2*m exceeds the
        number of channels, all channels' filters are used once (see module docstring).
    reg : float, default=1e-5
        Small diagonal shrinkage/regularization parameter to prevent numerical
        singularity during matrix inversion.
    """

    def __init__(self, n_components: int = 3, reg: float = 1e-5):
        self.m = n_components
        self.reg = reg
        self.W: Optional[np.ndarray] = None  # Spatial filter matrix of shape (C, n_filters)
        self.eigenvalues: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def _compute_normalized_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute average normalized trial covariance matrix (Equation 2 in paper):
            C_bar = (1 / N) * sum_{n=1}^N [ D_n @ D_n.T / trace(D_n @ D_n.T) ]
        """
        n_trials, n_channels, _ = X.shape
        cov_sum = np.zeros((n_channels, n_channels), dtype=np.float64)

        for d in X:
            ddt = d @ d.T
            tr = np.trace(ddt)
            cov_sum += ddt / tr if tr > 0 else ddt
        return cov_sum / n_trials

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CommonSpatialPattern":
        """
        Fit CSP spatial filters W on training trials.
        """
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes, but found {len(unique_classes)}: {unique_classes}")

        self.classes_ = unique_classes
        class1, class2 = unique_classes[0], unique_classes[1]

        X1 = X[y == class1]
        X2 = X[y == class2]

        # 1. Compute normalized class-average covariance matrices (Eq. 2)
        C1 = self._compute_normalized_covariance(X1)
        C2 = self._compute_normalized_covariance(X2)

        n_channels = C1.shape[0]

        # Regularization (diagonal loading) for numerical stability
        if self.reg > 0:
            C1 += self.reg * np.eye(n_channels)
            C2 += self.reg * np.eye(n_channels)

        # 2. Solve generalized eigenvalue problem: C2^{-1} C1 w = lambda w (Eq. 3)
        eigvals, eigvecs = la.eig(C1, C2)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        # 3. Sort eigenvalues in descending order to obtain M_tilde
        sort_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_indices]
        eigvecs = eigvecs[:, sort_indices]

        self.eigenvalues = eigvals

        # 4. Form final spatial filter W from first m and last m columns
        m = self.m
        if 2 * m > n_channels:
            # e.g. Dataset 3 (3 channels, m = 3): first-m and last-m overlap -> use each filter once
            self.W = eigvecs.copy()  # Shape: (C, C)
        else:
            W_first_m = eigvecs[:, :m]
            W_last_m = eigvecs[:, -m:]
            self.W = np.concatenate([W_first_m, W_last_m], axis=1)  # Shape: (C, 2*m)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply spatial projection and extract normalized log-variance features.
        Equations 4 & 5:
            Z = W^T D
            f_p = log( var(Z_p) / sum_{i=1}^{2m} var(Z_i) )
        """
        if self.W is None:
            raise RuntimeError("The CSP model has not been fitted yet. Call fit() first.")

        features = []
        eps = 1e-12

        for d in X:
            Z = self.W.T @ d
            var_Z = np.var(Z, axis=1, ddof=1)
            var_sum = np.sum(var_Z)
            if var_sum <= 0:
                var_sum = eps
            f = np.log((var_Z / var_sum) + eps)
            features.append(f)

        return np.asarray(features, dtype=np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)


# ---------------- Fisher Linear Discriminant Analysis (FLDA) ----------------

class FLDAClassifier:
    """Fisher Linear Discriminant Analysis (FLDA) classifier."""

    def __init__(self, solver: str = "svd", shrinkage: Optional[Union[str, float]] = None):
        self.clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FLDAClassifier":
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.clf.score(X, y))


# ---------------- Evaluation Helper ----------------

def evaluate_single_npz(
    npz_path: Path,
    m: int = DEFAULT_M,
    reg: float = 1e-5,
    save_features: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Evaluate CSP (m) + FLDA on single .npz preprocessed dataset file."""
    data = np.load(npz_path, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    subject = str(data["subject"]) if "subject" in data else npz_path.stem.split("_")[0]
    classes = [str(c) for c in data["classes"]] if "classes" in data else ["Class 1", "Class 2"]
    pair_label = str(data["pair"]) if "pair" in data else ""
    session = str(data["session"]) if "session" in data else ""  # Dataset 3 only

    csp = CommonSpatialPattern(n_components=m, reg=reg)
    csp.fit(X_train, y_train)

    train_features = csp.transform(X_train)
    test_features = csp.transform(X_test)

    flda = FLDAClassifier()
    flda.fit(train_features, y_train)

    train_acc = accuracy_score(y_train, flda.predict(train_features))
    test_acc = accuracy_score(y_test, flda.predict(test_features))

    if save_features and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        feat_name = f"{npz_path.stem.replace('_preprocessed', '')}_csp_features.npz"
        feat_path = output_dir / feat_name
        np.savez_compressed(
            feat_path,
            train_features=train_features,
            y_train=y_train,
            test_features=test_features,
            y_test=y_test,
            W=csp.W,
            eigenvalues=csp.eigenvalues,
            m=m,
            subject=subject,
            classes=np.asarray(classes),
            train_acc=train_acc,
            test_acc=test_acc,
        )

    return {
        "file": npz_path.name,
        "subject": subject,
        "session": session,
        "pair": pair_label,
        "classes": classes,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "W_shape": csp.W.shape,
    }


def save_results_csv(results: List[Dict], output_path: Path) -> None:
    """Save per-file CSP + FLDA metrics as a CSV results table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file", "subject", "session", "pair", "classes", "train_acc", "test_acc",
        "n_train", "n_test", "W_shape",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = result.copy()
            row["classes"] = " vs. ".join(row["classes"])
            row["W_shape"] = "x".join(str(size) for size in row["W_shape"])
            writer.writerow(row)


def _print_mean_std(label: str, values: np.ndarray, paper: Optional[Sequence[float]] = None) -> None:
    line = f"{label}: {np.mean(values):.2f}% +/- {np.std(values):.2f}%"
    if paper is not None:
        line += f"   (paper, CSP column: {paper[0]:.2f}% +/- {paper[1]:.2f}%)"
    print(line)


def evaluate_dataset1(
    data_dir: Path,
    output_dir: Path,
    m: int = 3,
    reg: float = 1e-5,
    summary_path: Optional[Path] = None,
):
    """Run CSP + FLDA on all 7 subjects of Dataset 1."""
    npz_files = sorted(data_dir.glob("*_preprocessed.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No Dataset 1 files found in {data_dir.resolve()}.")

    print(f"\n{'=' * 75}")
    print(f"EVALUATING DATASET 1: BCI Competition IV Dataset 1 (CSP m={m}, FLDA)")
    print(f"{'=' * 75}")

    results = []
    for f in npz_files:
        res = evaluate_single_npz(f, m=m, reg=reg, save_features=True, output_dir=output_dir)
        results.append(res)
        class_str = " vs. ".join(res["classes"])
        print(f"Subject {res['subject']:<4} | Tasks: {class_str:<20} | Train: {res['train_acc'] * 100:6.2f}% | Test: {res['test_acc'] * 100:6.2f}%")

    train_accuracies = np.asarray([r["train_acc"] for r in results]) * 100
    test_accuracies = np.asarray([r["test_acc"] for r in results]) * 100
    print("-" * 75)
    print("DATASET 1 SUMMARY (7 subjects; mean +/- population standard deviation)")
    print(
        f"Train accuracy: {np.mean(train_accuracies):.2f}% +/- "
        f"{np.std(train_accuracies):.2f}%"
    )
    print(
        f"Test accuracy : {np.mean(test_accuracies):.2f}% +/- "
        f"{np.std(test_accuracies):.2f}%"
    )
    print("=" * 75)
    if summary_path is not None:
        save_results_csv(results, summary_path)
        print(f"Per-subject results saved to: {summary_path.resolve()}")
    return results


def evaluate_dataset2(
    data_dir: Path,
    output_dir: Path,
    m: int = 3,
    reg: float = 1e-5,
    summary_path: Optional[Path] = None,
):
    """Run CSP + FLDA on the 54 binary classification subsets of Dataset 2a."""
    # Exclude full 4-class file, focus on the 6 binary pairs per subject
    npz_files = sorted([f for f in data_dir.glob("A0*_*_preprocessed.npz") if "full" not in f.name])
    if not npz_files:
        raise FileNotFoundError(f"No binary subset files found in {data_dir.resolve()}.")

    print(f"\n{'=' * 75}")
    print(
        "EVALUATING DATASET 2a: 54 Binary Classification Subsets (C_4^2 = 6 pairs x 9 subjects)"
    )
    print(f"{'=' * 75}")

    results = []
    for f in npz_files:
        res = evaluate_single_npz(f, m=m, reg=reg, save_features=True, output_dir=output_dir)
        results.append(res)

    # Group by subject
    subjects = sorted(list({r["subject"] for r in results}))
    print(f"\n{'Subject':<10} {'LR':<9} {'LF':<9} {'LT':<9} {'RF':<9} {'RT':<9} {'FT':<9} {'Mean Test':<10}")
    print("-" * 75)

    sub_means = []
    for sub in subjects:
        sub_res = {r["pair"]: r["test_acc"] * 100 for r in results if r["subject"] == sub}
        lr = sub_res.get("LR", 0)
        lf = sub_res.get("LF", 0)
        lt = sub_res.get("LT", 0)
        rf = sub_res.get("RF", 0)
        rt = sub_res.get("RT", 0)
        ft = sub_res.get("FT", 0)
        s_mean = np.mean([lr, lf, lt, rf, rt, ft])
        sub_means.append(s_mean)
        print(f"{sub:<10} {lr:>6.1f}%  {lf:>6.1f}%  {lt:>6.1f}%  {rf:>6.1f}%  {rt:>6.1f}%  {ft:>6.1f}%  {s_mean:>8.2f}%")

    train_accuracies = np.asarray([r["train_acc"] for r in results]) * 100
    test_accuracies = np.asarray([r["test_acc"] for r in results]) * 100
    overall_mean = np.mean(sub_means)
    print("-" * 75)
    print(f"{'OVERALL MEAN':<58} {overall_mean:>8.2f}%")
    print("\nDATASET 2a SUMMARY (54 subject/pair evaluations; mean +/- population standard deviation)")
    print(
        f"Train accuracy: {np.mean(train_accuracies):.2f}% +/- "
        f"{np.std(train_accuracies):.2f}%"
    )
    print(
        f"Test accuracy : {np.mean(test_accuracies):.2f}% +/- "
        f"{np.std(test_accuracies):.2f}%"
    )
    print("=" * 75)
    if summary_path is not None:
        save_results_csv(results, summary_path)
        print(f"Per-subject/pair results saved to: {summary_path.resolve()}")
    return results


def evaluate_dataset3(
    data_dir: Path,
    output_dir: Path,
    m: int = 3,
    reg: float = 1e-5,
    summary_path: Optional[Path] = None,
    sessions: Sequence[int] = (3,),
):
    """Run CSP + FLDA on Dataset 3 (BCI IV 2b; 9 subjects, 3 channels, left vs. right hand).

    sessions: which session files to evaluate. The paper uses session 3 only
    (B0103T ... B0903T); pass e.g. (1, 2, 3, 4, 5) for every preprocessed file.
    """
    npz_files = sorted(data_dir.glob("B0?0?[TE]_preprocessed.npz"))
    if sessions:
        npz_files = [f for f in npz_files if int(f.stem[3:5]) in set(sessions)]
    if not npz_files:
        raise FileNotFoundError(
            f"No Dataset 3 files for session(s) {list(sessions) or 'any'} found in {data_dir.resolve()}."
        )

    print(f"\n{'=' * 75}")
    print(f"EVALUATING DATASET 3: BCI Competition IV Dataset 2b (CSP m={m}, FLDA)")
    print(f"Sessions: {sorted({int(f.stem[3:5]) for f in npz_files})}  ({len(npz_files)} files)")
    print("Note: only 3 channels -> all 3 spatial filters are used once (2m = 6 > 3 channels).")
    print(f"{'=' * 75}")

    results, failed = [], []
    for f in npz_files:
        try:
            res = evaluate_single_npz(f, m=m, reg=reg, save_features=True, output_dir=output_dir)
        except Exception as exc:  # keep going with the remaining files
            failed.append((f.name, repr(exc)))
            print(f"{f.stem.replace('_preprocessed', ''):<8} | FAILED: {exc!r}")
            continue
        results.append(res)

    # Table: subjects x sessions (test accuracy, %)
    sess_present = sorted({int(r["session"]) for r in results})
    subjects = sorted({r["subject"] for r in results})
    lookup = {(r["subject"], int(r["session"])): r for r in results}
    header = f"{'Subject':<8}" + "".join(f"{'Sess ' + str(s):>10}" for s in sess_present)
    print("\nTest accuracy (%) per subject and session")
    print(header)
    print("-" * len(header))
    for sub in subjects:
        cells = []
        for s in sess_present:
            r = lookup.get((sub, s))
            cells.append(f"{r['test_acc'] * 100:>10.2f}" if r else f"{'-':>10}")
        print(f"{sub:<8}" + "".join(cells))
    print("-" * len(header))
    for s in sess_present:
        accs = np.asarray([r["test_acc"] for r in results if int(r["session"]) == s]) * 100
        print(f"Session {s}: test mean {np.mean(accs):.2f}% +/- {np.std(accs):.2f}% (n={len(accs)})")

    # Paper protocol: session 3 (80 train / 80 test), compared with the paper's CSP column
    if 3 in sess_present:
        r3 = sorted([r for r in results if int(r["session"]) == 3], key=lambda r: r["subject"])
        print(f"\n{'Subject':<8} {'Train':>8} {'Test':>8} {'Paper CSP':>10} {'Diff':>8}   (session 3 = paper protocol)")
        print("-" * 52)
        for r in r3:
            paper = PAPER_CSP_DS3.get(r["subject"])
            test = r["test_acc"] * 100
            ref = f"{paper:>10.2f}" if paper is not None else f"{'-':>10}"
            diff = f"{test - paper:>+8.2f}" if paper is not None else f"{'-':>8}"
            print(f"{r['subject']:<8} {r['train_acc'] * 100:>7.2f}% {test:>7.2f}% {ref} {diff}")
        print("-" * 52)
        print("DATASET 3 SUMMARY, session 3 (mean +/- population standard deviation)")
        _print_mean_std("Train accuracy", np.asarray([r["train_acc"] for r in r3]) * 100)
        _print_mean_std("Test accuracy ", np.asarray([r["test_acc"] for r in r3]) * 100, PAPER_CSP_DS3_MEAN_STD)
    print("=" * 75)
    for name, err in failed:
        print(f"Failed: {name}: {err}")
    if summary_path is not None:
        save_results_csv(results, summary_path)
        print(f"Per-file results saved to: {summary_path.resolve()}")
    return results


def evaluate_dataset4(
    data_dir: Path,
    output_dir: Path,
    m: int = 3,
    reg: float = 1e-5,
    summary_path: Optional[Path] = None,
):
    """Run CSP + FLDA on Dataset 4 (Steyrl et al.; 14 subjects, 15 channels, right hand vs. feet)."""
    npz_files = sorted(data_dir.glob("S??_preprocessed.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No Dataset 4 files found in {data_dir.resolve()}.")

    print(f"\n{'=' * 75}")
    print(f"EVALUATING DATASET 4: Steyrl et al. 2016, right hand vs. feet (CSP m={m}, FLDA)")
    print(f"{'=' * 75}")
    print(f"{'Subject':<8} {'Tasks':<20} {'Train':>8} {'Test':>8} {'Paper CSP':>10} {'Diff':>8}")
    print("-" * 68)

    results, failed = [], []
    for f in npz_files:
        try:
            res = evaluate_single_npz(f, m=m, reg=reg, save_features=True, output_dir=output_dir)
        except Exception as exc:  # keep going with the remaining subjects
            failed.append((f.name, repr(exc)))
            print(f"{f.stem.replace('_preprocessed', ''):<8} | FAILED: {exc!r}")
            continue
        results.append(res)
        class_str = " vs. ".join(res["classes"])
        paper = PAPER_CSP_DS4.get(res["subject"])
        test = res["test_acc"] * 100
        ref = f"{paper:>10.2f}" if paper is not None else f"{'-':>10}"
        diff = f"{test - paper:>+8.2f}" if paper is not None else f"{'-':>8}"
        print(f"{res['subject']:<8} {class_str:<20} {res['train_acc'] * 100:>7.2f}% {test:>7.2f}% {ref} {diff}")

    if results:
        print("-" * 68)
        print(f"DATASET 4 SUMMARY ({len(results)} subjects; mean +/- population standard deviation)")
        _print_mean_std("Train accuracy", np.asarray([r["train_acc"] for r in results]) * 100)
        _print_mean_std("Test accuracy ", np.asarray([r["test_acc"] for r in results]) * 100, PAPER_CSP_DS4_MEAN_STD)
    print("=" * 68)
    for name, err in failed:
        print(f"Failed: {name}: {err}")
    if summary_path is not None and results:
        save_results_csv(results, summary_path)
        print(f"Per-subject results saved to: {summary_path.resolve()}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run CSP Transformation (m=3) & FLDA on Datasets 1, 2a, 3 and/or 4."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["1", "2a", "3", "4", "all"],
        help="Dataset to evaluate: '1', '2a', '3', '4', or 'all' (default: 'all')",
    )
    parser.add_argument("-m", type=int, default=DEFAULT_M, help="Number of filter pairs (m, default: 3).")
    parser.add_argument("--reg", type=float, default=1e-5, help="Regularization parameter (default: 1e-5).")
    parser.add_argument("--ds3-sessions", type=int, nargs="*", default=[3],
                        help="Dataset 3 sessions to evaluate (default: 3 = B0103T..B0903T, as in the paper).")
    parser.add_argument("--ds3-all-sessions", action="store_true",
                        help="Dataset 3: evaluate every preprocessed session file (overrides --ds3-sessions).")
    args = parser.parse_args()

    base_dir = DEFAULT_BASE_DATA
    ds3_sessions = () if args.ds3_all_sessions else tuple(args.ds3_sessions)

    # (key, evaluation function, preprocessed folder, results folder, extra kwargs)
    jobs = [
        ("1", evaluate_dataset1, "dataset_1", "dataset_1", {}),
        ("2a", evaluate_dataset2, "dataset_2", "dataset_2", {}),
        ("3", evaluate_dataset3, "dataset_3", "dataset_3", {"sessions": ds3_sessions}),
        ("4", evaluate_dataset4, "dataset_4", "dataset_4", {}),
    ]
    for key, func, in_name, out_name, extra in jobs:
        if args.dataset not in (key, "all"):
            continue
        results_dir = DEFAULT_RESULTS_DIR / out_name
        try:
            func(
                base_dir / in_name,
                results_dir / "csp_features",
                m=args.m,
                reg=args.reg,
                summary_path=results_dir / "csp_flda_results.csv",
                **extra,
            )
        except FileNotFoundError as exc:
            if args.dataset != "all":
                raise
            print(f"\n[skip] dataset {key}: {exc}")


if __name__ == "__main__":
    main()
