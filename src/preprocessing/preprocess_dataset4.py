"""Preprocess the Steyrl et al. motor-imagery MAT dataset (paper Dataset 4)."""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = ROOT / "Data" / "dataset_4"
DEFAULT_OUTPUT_DIR = ROOT / "preprocessed_data" / "dataset_4"
TARGET_FS = 256.0
LOW, HIGH = 8.0, 30.0
START, END = 0.5, 2.5


def report_saved_trials(subject, X_train, X_test, output_file: Path, task=None):
    """Print one consistent summary line for a saved trial set."""
    name = f"{subject}-{task}" if task else str(subject)
    print(
        f"Subject {name:<8} | Train: {X_train.shape} | Test: {X_test.shape} "
        f"| Saved: {output_file.resolve()}"
    )


def _field(run, key):
    if hasattr(run, key):
        return getattr(run, key)
    if isinstance(run, dict):
        return run[key]
    raise KeyError(f"MAT run has no '{key}' field.")


def _runs(path: Path):
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    for run in np.atleast_1d(mat["data"]):
        signal = np.asarray(_field(run, "X"), dtype=np.float64)
        if signal.ndim != 2:
            raise ValueError(f"Expected 2D continuous EEG in {path}; got {signal.shape}")
        # These recordings store samples by channels. Correct if a file is transposed.
        if signal.shape[0] < signal.shape[1] and signal.shape[0] <= 64:
            signal = signal.T
        trial_samples = np.asarray(_field(run, "trial"), dtype=np.int64).reshape(-1) - 1
        labels = np.asarray(_field(run, "y"), dtype=np.int64).reshape(-1)
        fs = float(np.asarray(_field(run, "fs")).squeeze())
        if fs == 512.0:
            signal = resample_poly(signal, up=1, down=2, axis=0)
            trial_samples = np.rint(trial_samples / 2).astype(np.int64)
            fs = TARGET_FS
        if fs != TARGET_FS:
            raise ValueError(f"Expected 512 Hz or 256 Hz input in {path}; found {fs} Hz")
        yield signal, trial_samples, labels, fs


def _epoch_run(signal, cues, labels, sos):
    # Filter continuously before cutting the [0.5, 2.5] s motor-imagery interval.
    filtered = sosfiltfilt(sos, signal, axis=0)
    a, b = int(round(START * TARGET_FS)), int(round(END * TARGET_FS))
    raw_epochs, filtered_epochs, kept = [], [], []
    for onset, label in zip(cues, labels):
        left, right = int(onset) + a, int(onset) + b
        if left >= 0 and right <= signal.shape[0]:
            raw_epochs.append(signal[left:right].T)
            filtered_epochs.append(filtered[left:right].T)
            kept.append(label)
    return raw_epochs, filtered_epochs, kept


def preprocess_subject(subject_id: str, data_dir: Path, output_dir: Path):
    train_path, test_path = data_dir / f"{subject_id}T.mat", data_dir / f"{subject_id}E.mat"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing training or evaluation MAT file for {subject_id} in {data_dir}")
    sos = butter(6, [LOW, HIGH], btype="bandpass", fs=TARGET_FS, output="sos")
    arrays = {"train_raw": [], "train_filt": [], "train_y": [],
              "test_raw": [], "test_filt": [], "test_y": []}
    for path, prefix in ((train_path, "train"), (test_path, "test")):
        for signal, cues, labels, _ in _runs(path):
            xr, xf, y = _epoch_run(signal, cues, labels, sos)
            arrays[f"{prefix}_raw"].extend(xr)
            arrays[f"{prefix}_filt"].extend(xf)
            arrays[f"{prefix}_y"].extend(y)
    X_train_raw = np.asarray(arrays["train_raw"], dtype=np.float32)
    X_train_filt = np.asarray(arrays["train_filt"], dtype=np.float32)
    y_train = np.asarray(arrays["train_y"], dtype=np.int64)
    X_test_raw = np.asarray(arrays["test_raw"], dtype=np.float32)
    X_test_filt = np.asarray(arrays["test_filt"], dtype=np.float32)
    y_test = np.asarray(arrays["test_y"], dtype=np.int64)
    if len(y_train) != 100 or len(y_test) != 60:
        raise ValueError(
            f"{subject_id}: expected 100 train and 60 test trials, "
            f"found {len(y_train)} and {len(y_test)}. Check the MAT files and trial parsing."
        )
    class_values = np.unique(np.concatenate((y_train, y_test)))
    if len(class_values) != 2:
        raise ValueError(f"Expected two classes for {subject_id}; found {class_values.tolist()}.")
    y_train_binary = np.searchsorted(class_values, y_train).astype(np.int64)
    y_test_binary = np.searchsorted(class_values, y_test).astype(np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{subject_id}_preprocessed.npz"
    np.savez_compressed(
        out,
        X_train_raw=X_train_raw, X_train=X_train_filt, y_train=y_train,
        X_test_raw=X_test_raw, X_test=X_test_filt, y_test=y_test,
        y_train_binary=y_train_binary, y_test_binary=y_test_binary,
        fs=TARGET_FS, subject=subject_id, pair="",
        channels=np.asarray([f"channel_{i + 1}" for i in range(X_train_raw.shape[1])]),
        classes=np.asarray([f"class_{value}" for value in class_values]),
        time_window=np.array([START, END]), freq_band=np.array([LOW, HIGH]),
        filter_order=6,
    )
    report_saved_trials(subject_id, X_train_raw, X_test_raw, out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", nargs="+", default=[f"S{i:02d}" for i in range(1, 15)])
    args = parser.parse_args()
    for subject in args.subjects:
        preprocess_subject(subject, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
