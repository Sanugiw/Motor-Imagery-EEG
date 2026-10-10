"""
Preprocessing script for BCI Competition IV (2008) Dataset 1.
Paper: "The CSP-Based New Features Plus Non-Convex Log Sparse Feature Selection
       for Motor Imagery EEG Classification", Sensors 2020, 20(17), 4749.

Dataset specifications:
- 59 EEG channels, recorded at 100 Hz.
- Motor imagery tasks: left hand, right hand, foot (each subject performs 2 classes).
- 7 subjects: 1a, 1b, 1c, 1d, 1e, 1f, 1g.
- Calibration data: 2 runs with 100 single trials each (total 200 trials).
  - Run 1 (trials 0..99) -> Training set (100 trials: 50 per class).
  - Run 2 (trials 100..199) -> Test set (100 trials: 50 per class).

Preprocessing pipeline:
1. Voltage conversion: cnt = 0.1 * cnt (µV).
2. 6th-order Butterworth bandpass filter (8–30 Hz) on continuous EEG channels (sosfiltfilt).
3. Trial epoching: [0.5 s, 3.5 s] relative to cue onset (t = 0 s), yielding 300 samples per trial.
4. Train/Test chronological run split (Run 1: 100 trials, Run 2: 100 trials).
5. Saving preprocessed trials (X_train, y_train, X_test, y_test, metadata) to:
   processed_data/<dataset_name>/BCICIV_calib_ds<subject>_preprocessed.npz
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

# ---------------- Default Configuration ----------------
DATASET_NAME = "BCICIV_1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data" / "BCICIV_1_mat"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed_data" / DATASET_NAME

FS = 100  # Sampling rate (Hz)
LOWCUT = 8.0  # Bandpass low cutoff (Hz) - mu rhythm lower bound
HIGHCUT = 30.0  # Bandpass high cutoff (Hz) - beta rhythm upper bound
FILTER_ORDER = 6  # 6th-order Butterworth filter

EPOCH_START = 0.5  # Start of motor imagery window (seconds)
EPOCH_END = 3.5  # End of motor imagery window (seconds)
EXPECTED_SAMPLES = int(round((EPOCH_END - EPOCH_START) * FS))  # 300 samples

N_TRAIN = 100  # Run 1 single trials
N_TEST = 100  # Run 2 single trials


# ---------------- MATLAB Helper Functions ----------------

def get_field(struct, name: str):
    """Safely retrieve a field from a MATLAB struct or dict loaded by scipy.io."""
    if hasattr(struct, name):
        return getattr(struct, name)
    if isinstance(struct, dict) and name in struct:
        return struct[name]
    raise KeyError(f"Field '{name}' not found in MATLAB struct/dict.")


def load_bciciv_dataset(mat_path: Path) -> Dict:
    """
    Load continuous EEG signals, cue markers, and metadata from a BCI Competition IV Dataset 1 .mat file.

    Parameters
    ----------
    mat_path : Path
        Path to the calibration .mat file (e.g., BCICIV_calib_ds1a.mat).

    Returns
    -------
    data_dict : dict
        Dictionary containing:
        - 'cnt': continuous EEG array of shape (n_channels, n_samples) in µV
        - 'positions': cue onset positions (0-based sample indices)
        - 'labels': target labels for each cue (-1 or 1)
        - 'fs': sampling rate in Hz (100 Hz)
        - 'clab': list of 59 electrode names
        - 'classes': list of class names (e.g., ['left', 'foot'])
        - 'subject': subject ID (e.g., '1a')
    """
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    if "cnt" not in mat or "mrk" not in mat:
        raise KeyError(
            f"File {mat_path.name} is missing 'cnt' or 'mrk'. "
            "Ensure this is a calibration file (BCICIV_calib_ds1*.mat)."
        )

    cnt = np.asarray(mat["cnt"], dtype=np.float64)
    mrk = mat["mrk"]
    nfo = mat["nfo"]

    # Official format: samples x channels. Transpose to channels x samples.
    if cnt.shape[0] > cnt.shape[1]:
        cnt = cnt.T

    # Convert stored integer values to microvolts (µV) per official documentation
    cnt *= 0.1

    # Extract cue positions and labels
    raw_positions = np.asarray(get_field(mrk, "pos"), dtype=int).reshape(-1)
    labels = np.asarray(get_field(mrk, "y"), dtype=int).reshape(-1)

    # MATLAB 1-based indexing -> Python 0-based indexing
    positions = raw_positions - 1

    # Metadata
    fs = float(get_field(nfo, "fs"))
    if not np.isclose(fs, FS):
        raise ValueError(f"Expected {FS} Hz sampling rate, but found {fs} Hz in {mat_path.name}.")

    raw_clab = get_field(nfo, "clab")
    clab = [str(ch) for ch in raw_clab] if hasattr(raw_clab, "__iter__") else []

    raw_classes = get_field(nfo, "classes")
    classes = [str(c) for c in raw_classes] if hasattr(raw_classes, "__iter__") else []

    # Extract subject identifier from filename (e.g., '1a' from 'BCICIV_calib_ds1a.mat')
    stem = mat_path.stem
    subject = stem.replace("BCICIV_calib_ds", "").replace("BCICIV_eval_ds", "")

    return {
        "cnt": cnt,
        "positions": positions,
        "labels": labels,
        "fs": fs,
        "clab": clab,
        "classes": classes,
        "subject": subject,
    }


# ---------------- Digital Signal Processing ----------------

def apply_butterworth_bandpass(
    cnt: np.ndarray,
    lowcut: float = LOWCUT,
    highcut: float = HIGHCUT,
    fs: float = FS,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter using Second-Order Sections (SOS).

    Parameters
    ----------
    cnt : ndarray of shape (n_channels, n_samples)
        Continuous raw EEG data.
    lowcut : float
        Lower cutoff frequency in Hz (default: 8.0 Hz).
    highcut : float
        Upper cutoff frequency in Hz (default: 30.0 Hz).
    fs : float
        Sampling frequency in Hz (default: 100.0 Hz).
    order : int
        Filter order (default: 6).

    Returns
    -------
    filtered : ndarray of shape (n_channels, n_samples)
        Zero-phase bandpass-filtered EEG signal.
    """
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, cnt, axis=1)
    return filtered


def extract_epochs(
    filtered_cnt: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    epoch_start: float = EPOCH_START,
    epoch_end: float = EPOCH_END,
    fs: float = FS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract cue-aligned single-trial epochs.

    Parameters
    ----------
    filtered_cnt : ndarray of shape (n_channels, n_samples)
        Continuous filtered EEG data.
    positions : ndarray of shape (n_cues,)
        0-based cue onset sample indices.
    labels : ndarray of shape (n_cues,)
        Target class labels (-1 or 1).
    epoch_start : float
        Trial window start time in seconds relative to cue (default: 0.5 s).
    epoch_end : float
        Trial window end time in seconds relative to cue (default: 3.5 s).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    X : ndarray of shape (n_trials, n_channels, n_samples)
        Extracted trial epochs (e.g., shape (200, 59, 300)).
    y : ndarray of shape (n_trials,)
        Corresponding target class labels.
    """
    start_offset = int(round(epoch_start * fs))
    end_offset = int(round(epoch_end * fs))
    expected_len = end_offset - start_offset

    epochs = []
    valid_labels = []
    total_samples = filtered_cnt.shape[1]

    for idx, (cue_pos, label) in enumerate(zip(positions, labels)):
        start = cue_pos + start_offset
        end = cue_pos + end_offset

        if start < 0 or end > total_samples:
            print(f"Warning: Trial {idx} at cue sample {cue_pos} is out of bounds [{start}, {end}]. Skipping.")
            continue

        epoch = filtered_cnt[:, start:end]
        if epoch.shape[1] != expected_len:
            print(f"Warning: Trial {idx} length {epoch.shape[1]} != {expected_len}. Skipping.")
            continue

        epochs.append(epoch)
        valid_labels.append(label)

    X = np.asarray(epochs, dtype=np.float32)
    y = np.asarray(valid_labels, dtype=np.int64)
    return X, y


def split_runs(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronologically split trials into Run 1 (Training set) and Run 2 (Testing set)
    as defined in the BCI Competition IV experimental protocol.

    Parameters
    ----------
    X : ndarray of shape (n_trials, n_channels, n_samples)
    y : ndarray of shape (n_trials,)
    n_train : int
        Number of trials in Run 1 (default: 100).
    n_test : int
        Number of trials in Run 2 (default: 100).

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    total_required = n_train + n_test
    if len(y) < total_required:
        raise ValueError(
            f"Expected at least {total_required} trials for Run 1 + Run 2 split, but got {len(y)}."
        )

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_test = X[n_train:total_required]
    y_test = y[n_train:total_required]

    return X_train, y_train, X_test, y_test


# ---------------- Processing Pipeline ----------------

def preprocess_subject(
    mat_path: Path,
    output_dir: Path,
) -> Dict:
    """
    Run full preprocessing pipeline for a single subject calibration file.
    """
    print(f"\n{'=' * 65}")
    print(f"Processing calibration file: {mat_path.name}")
    print(f"{'=' * 65}")

    # 1. Load data
    data = load_bciciv_dataset(mat_path)
    cnt = data["cnt"]
    positions = data["positions"]
    labels = data["labels"]
    clab = data["clab"]
    classes = data["classes"]
    subject = data["subject"]

    print(f"Subject: {subject}")
    print(f"Classes: {classes}")
    print(f"Raw continuous EEG shape: {cnt.shape} (channels x samples)")
    print(f"Total cue markers: {len(positions)}")

    # 2. Bandpass filtering (6th-order Butterworth 8-30 Hz)
    print(f"Applying 6th-order Butterworth bandpass filter ({LOWCUT}–{HIGHCUT} Hz)...")
    filtered_cnt = apply_butterworth_bandpass(cnt, lowcut=LOWCUT, highcut=HIGHCUT, fs=FS, order=FILTER_ORDER)

    # 3. Epoch extraction (0.5 s to 3.5 s relative to cue onset)
    print(f"Extracting epochs [{EPOCH_START} s, {EPOCH_END} s] -> {EXPECTED_SAMPLES} samples/trial...")
    X, y = extract_epochs(
        filtered_cnt,
        positions,
        labels,
        epoch_start=EPOCH_START,
        epoch_end=EPOCH_END,
        fs=FS,
    )
    print(f"Extracted epochs shape: {X.shape} (trials x channels x samples)")

    unique_labels, counts = np.unique(y, return_counts=True)
    label_dist = dict(zip(unique_labels.tolist(), counts.tolist()))
    print(f"Trial label counts (overall): {label_dist}")

    # 4. Split Run 1 (training) and Run 2 (testing)
    X_train, y_train, X_test, y_test = split_runs(X, y, n_train=N_TRAIN, n_test=N_TEST)

    # Standard binary label mapping: -1 -> 0, +1 -> 1
    y_train_binary = np.where(y_train == -1, 0, 1).astype(np.int64)
    y_test_binary = np.where(y_test == -1, 0, 1).astype(np.int64)

    print(f"Run 1 (Training set): X_train={X_train.shape}, classes={dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Run 2 (Testing set) : X_test={X_test.shape}, classes={dict(zip(*np.unique(y_test, return_counts=True)))}")

    # 5. Save preprocessed dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"BCICIV_calib_ds{subject}_preprocessed.npz"

    np.savez_compressed(
        out_file,
        X_train=X_train,
        y_train=y_train,
        y_train_binary=y_train_binary,
        X_test=X_test,
        y_test=y_test,
        y_test_binary=y_test_binary,
        fs=FS,
        classes=np.asarray(classes),
        clab=np.asarray(clab),
        subject=subject,
        time_window=np.array([EPOCH_START, EPOCH_END]),
        freq_band=np.array([LOWCUT, HIGHCUT]),
        filter_order=FILTER_ORDER,
    )
    print(f"Saved processed data to: {out_file}")

    return {
        "subject": subject,
        "classes": classes,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_dist": dict(zip(np.unique(y_train).tolist(), np.bincount(y_train_binary).tolist())),
        "y_test_dist": dict(zip(np.unique(y_test).tolist(), np.bincount(y_test_binary).tolist())),
        "file": str(out_file),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BCI Competition IV Dataset 1 (Sensors 2020 paper protocol)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path to raw dataset folder containing .mat files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save preprocessed .npz files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory '{data_dir.resolve()}' does not exist. "
            "Please check the path or provide --data-dir."
        )

    # Calibration files: BCICIV_calib_ds1[a-g].mat
    calib_files = sorted(data_dir.glob("BCICIV_calib_ds1*.mat"))

    if not calib_files:
        raise FileNotFoundError(
            f"No calibration files matching 'BCICIV_calib_ds1*.mat' found in {data_dir.resolve()}."
        )

    print(f"Found {len(calib_files)} calibration dataset files in {data_dir.resolve()}:")
    for f in calib_files:
        print(f"  - {f.name}")
    print(f"Output directory: {output_dir.resolve()}")

    summaries = []
    for mat_path in calib_files:
        summary = preprocess_subject(
            mat_path=mat_path,
            output_dir=output_dir,
        )
        summaries.append(summary)

    print("\n" + "=" * 65)
    print("PREPROCESSING COMPLETE SUMMARY")
    print("=" * 65)
    print(f"{'Subject':<10} {'Classes':<20} {'Train Trials':<15} {'Test Trials':<15} {'Saved File'}")
    print("-" * 80)
    for s in summaries:
        class_str = ", ".join(s["classes"])
        train_shape = f"{s['X_train_shape'][0]} trials"
        test_shape = f"{s['X_test_shape'][0]} trials"
        file_name = Path(s["file"]).name
        print(f"{s['subject']:<10} {class_str:<20} {train_shape:<15} {test_shape:<15} {file_name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
