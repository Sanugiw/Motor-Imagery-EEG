"""Preprocess BCI Competition IV Dataset 1 (Berlin BCI)."""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "Data" / "dataset_1"
)
DEFAULT_BASE_OUTPUT = PROJECT_ROOT / "preprocessed_data"

FS = 100
EPOCH_START = 0.5
EPOCH_END = 3.5
N_TRAIN = 100
N_TEST = 100
LOWCUT = 8.0
HIGHCUT = 30.0
FILTER_ORDER = 6


def report_saved_trials(subject, X_train, X_test, output_file: Path, task=None):
    """Print one consistent summary line for a saved trial set."""
    name = f"{subject}-{task}" if task else str(subject)
    print(
        f"Subject {name:<8} | Train: {X_train.shape} | Test: {X_test.shape} "
        f"| Saved: {output_file.resolve()}"
    )


def apply_butterworth_bandpass(
    cnt: np.ndarray,
    fs: float = FS,
    lowcut: float = LOWCUT,
    highcut: float = HIGHCUT,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter along the time axis."""
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, cnt, axis=1)


def get_field(struct, name: str):
    """Safely retrieve a field from a MATLAB struct or dict."""
    if hasattr(struct, name):
        return getattr(struct, name)
    if isinstance(struct, dict) and name in struct:
        return struct[name]
    raise KeyError(f"Field '{name}' not found.")


def load_dataset1_mat(mat_path: Path) -> Dict:
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    cnt = np.asarray(mat["cnt"], dtype=np.float64)
    if cnt.shape[0] > cnt.shape[1]:
        cnt = cnt.T
    cnt *= 0.1  # Convert to microvolts.

    mrk = mat["mrk"]
    nfo = mat["nfo"]
    positions = np.asarray(get_field(mrk, "pos"), dtype=int).reshape(-1) - 1
    labels = np.asarray(get_field(mrk, "y"), dtype=int).reshape(-1)
    fs = float(get_field(nfo, "fs"))
    channels = [str(ch) for ch in get_field(nfo, "clab")]
    classes = [str(c) for c in get_field(nfo, "classes")]
    subject = mat_path.stem.replace("BCICIV_calib_ds", "").replace("BCICIV_eval_ds", "")

    return {
        "cnt": cnt,
        "positions": positions,
        "labels": labels,
        "fs": fs,
        "channels": channels,
        "classes": classes,
        "subject": subject,
    }


def preprocess_dataset1_subject(mat_path: Path, output_dir: Path) -> Dict:
    """Preprocess one Dataset 1 subject and save its training/test epochs."""
    data = load_dataset1_mat(mat_path)
    raw_cnt = data["cnt"]
    filtered_cnt = apply_butterworth_bandpass(raw_cnt, fs=FS)

    start_offset = int(round(EPOCH_START * FS))
    end_offset = int(round(EPOCH_END * FS))
    expected_len = end_offset - start_offset
    epochs = []
    raw_epochs = []
    valid_labels = []
    for position, label in zip(data["positions"], data["labels"]):
        start = position + start_offset
        end = position + end_offset
        if start >= 0 and end <= filtered_cnt.shape[1]:
            epoch = filtered_cnt[:, start:end]
            raw_epoch = raw_cnt[:, start:end]
            if epoch.shape[1] == expected_len and raw_epoch.shape[1] == expected_len:
                epochs.append(epoch)
                raw_epochs.append(raw_epoch)
                valid_labels.append(label)

    X = np.asarray(epochs, dtype=np.float32)
    X_raw = np.asarray(raw_epochs, dtype=np.float32)
    y = np.asarray(valid_labels, dtype=np.int64)
    X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
    X_test = X[N_TRAIN:N_TRAIN + N_TEST]
    y_test = y[N_TRAIN:N_TRAIN + N_TEST]
    X_train_raw = X_raw[:N_TRAIN]
    X_test_raw = X_raw[N_TRAIN:N_TRAIN + N_TEST]
    y_train_binary = np.where(y_train == -1, 0, 1).astype(np.int64)
    y_test_binary = np.where(y_test == -1, 0, 1).astype(np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"BCICIV_calib_ds{data['subject']}_preprocessed.npz"
    np.savez_compressed(
        out_file,
        X_train=X_train,
        X_train_raw=X_train_raw,
        y_train=y_train,
        y_train_binary=y_train_binary,
        X_test=X_test,
        X_test_raw=X_test_raw,
        y_test=y_test,
        y_test_binary=y_test_binary,
        fs=FS,
        classes=np.asarray(data["classes"]),
        channels=np.asarray(data["channels"]),
        subject=data["subject"],
        pair="",
        time_window=np.array([EPOCH_START, EPOCH_END]),
        freq_band=np.array([LOWCUT, HIGHCUT]),
        filter_order=FILTER_ORDER,
    )
    report_saved_trials(data["subject"], X_train, X_test, out_file)
    return {
        "dataset": "Dataset 1",
        "subject": data["subject"],
        "classes": data["classes"],
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "out_file": str(out_file),
    }


def run_preprocessing(data_dir: Path, output_dir: Path) -> List[Dict]:
    """Preprocess all calibration files for Dataset 1."""
    calib_files = sorted(data_dir.glob("BCICIV_calib_ds1*.mat"))
    if not calib_files:
        raise FileNotFoundError(f"No BCICIV_calib_ds1*.mat files found in {data_dir.resolve()}.")

    print(f"\n{'=' * 75}")
    print("PREPROCESSING DATASET 1: BCI Competition IV Dataset 1 (59 ch, 100 Hz, [0.5-3.5 s])")
    print(f"Data directory  : {data_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"{'=' * 75}\n")

    results = []
    for mat_path in calib_files:
        result = preprocess_dataset1_subject(mat_path, output_dir)
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BCI Competition IV Dataset 1.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"Raw Dataset 1 directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_OUTPUT / "dataset_1",
                        help="Output directory for preprocessed Dataset 1 files.")
    args = parser.parse_args()
    run_preprocessing(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
