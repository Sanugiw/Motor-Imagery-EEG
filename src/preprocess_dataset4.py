"""Preprocess Dataset 4: Steyrl et al. (2016) right hand vs. feet MI (BNCI Horizon 2020).

Follows Zhang et al., Sensors 2020, 20, 4749 (Sections 2.1 and 2.3):
  * 15 EEG channels, 14 subjects (S01 ... S14), right hand vs. feet.
  * Recorded at 512 Hz and downsampled to 256 Hz.
  * Runs 1-5 (file S??T.mat, 100 trials) are the training set,
    runs 6-8 (file S??E.mat, 60 trials) are the test set.
  * 6th-order Butterworth band-pass, 8-30 Hz (zero-phase).
  * Single-trial segmentation: 0.5-2.5 s after the motor-imagery onset
    (the `trial` marker of the .mat files).

Each .mat file holds a struct array `data`; every element is one run with
X (samples x 15), trial (onset sample, 1-based MATLAB index), y (1 = right hand,
2 = feet), fs and classes.  Runs are processed separately so that filtering and
resampling never run across the gap between two recordings.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.io as sio
from scipy.signal import butter, decimate, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_DATA_DIR = _first_existing(
    PROJECT_ROOT / "data" / "dataset_4",
    PROJECT_ROOT / "Data" / "dataset_4",
)
DEFAULT_BASE_OUTPUT = (
    PROJECT_ROOT / "preprocessed_data"
    if (PROJECT_ROOT / "preprocessed_data").exists()
    else PROJECT_ROOT / "processed_data"
)

ORIGINAL_FS = 512
FS = 256  # sampling rate after downsampling (as in the paper)
EPOCH_START = 0.5
EPOCH_END = 2.5
N_EEG_CHANNELS = 15
LOWCUT = 8.0
HIGHCUT = 30.0
FILTER_ORDER = 6
CLASS_NAMES = {1: "right hand", 2: "feet"}


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


def load_mat_session(mat_path: Path) -> List[Dict]:
    """Load one S??T.mat / S??E.mat file as a list of runs (channels x samples)."""
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    runs = []
    for run_idx, run in enumerate(np.atleast_1d(mat["data"])):
        X = np.asarray(run.X, dtype=np.float64)
        if X.shape[0] > X.shape[1]:  # (samples, channels) -> (channels, samples)
            X = X.T
        if X.shape[0] != N_EEG_CHANNELS:
            raise ValueError(f"{mat_path.name} run {run_idx + 1}: expected {N_EEG_CHANNELS} channels, got {X.shape[0]}")
        runs.append(
            {
                "X": X,
                "trial": np.atleast_1d(run.trial).astype(int) - 1,  # MATLAB 1-based -> 0-based
                "y": np.atleast_1d(run.y).astype(np.int64),
                "fs": float(run.fs),
                "classes": [str(c) for c in np.atleast_1d(run.classes)],
            }
        )
    return runs


def downsample_run(X: np.ndarray, trial: np.ndarray, fs: float, target_fs: float = FS):
    """Anti-aliased, zero-phase decimation of one run; marker positions are rescaled."""
    factor = fs / target_fs
    if abs(factor - round(factor)) > 1e-9:
        raise ValueError(f"fs={fs} is not an integer multiple of target fs={target_fs}")
    factor = int(round(factor))
    if factor == 1:
        return X, trial
    X_ds = decimate(X, factor, ftype="fir", zero_phase=True, axis=1)
    return X_ds, np.round(trial / factor).astype(int)


def extract_epochs(
    data: np.ndarray,
    onsets: np.ndarray,
    labels: np.ndarray,
    fs: float = FS,
    epoch_start: float = EPOCH_START,
    epoch_end: float = EPOCH_END,
):
    """Cut [onset + epoch_start, onset + epoch_end) windows out of continuous data."""
    start_offset = int(round(epoch_start * fs))
    end_offset = int(round(epoch_end * fs))
    expected_len = end_offset - start_offset
    epochs, valid_labels = [], []
    for onset, label in zip(onsets, labels):
        start, end = onset + start_offset, onset + end_offset
        if start >= 0 and end <= data.shape[1]:
            epoch = data[:, start:end]
            if epoch.shape[1] == expected_len:
                epochs.append(epoch)
                valid_labels.append(label)
    return epochs, valid_labels


def process_session(mat_path: Path):
    """Downsample -> band-pass -> epoch every run of a session file."""
    X_filt, X_raw, labels, run_ids = [], [], [], []
    for run_idx, run in enumerate(load_mat_session(mat_path), start=1):
        X_ds, trial_ds = downsample_run(run["X"], run["trial"], run["fs"])
        X_bp = apply_butterworth_bandpass(X_ds, fs=FS)
        ep_f, lab = extract_epochs(X_bp, trial_ds, run["y"])
        ep_r, _ = extract_epochs(X_ds, trial_ds, run["y"])
        X_filt += ep_f
        X_raw += ep_r
        labels += lab
        run_ids += [run_idx] * len(lab)
    return (
        np.asarray(X_filt, dtype=np.float32),
        np.asarray(X_raw, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(run_ids, dtype=np.int64),
    )


def preprocess_subject(
    subject_id: str,
    data_dir: Path,
    output_dir: Path,
    channel_names: Optional[Sequence[str]] = None,
) -> Dict:
    """Preprocess the training (runs 1-5) and test (runs 6-8) files of one subject."""
    train_mat = data_dir / f"{subject_id}T.mat"
    test_mat = data_dir / f"{subject_id}E.mat"
    if not train_mat.exists() or not test_mat.exists():
        raise FileNotFoundError(f"Missing {train_mat.name} or {test_mat.name} in {data_dir.resolve()}.")

    X_train, X_train_raw, y_train, run_train = process_session(train_mat)
    X_test, X_test_raw, y_test, run_test = process_session(test_mat)

    y_train_binary = np.where(y_train == 1, 0, 1).astype(np.int64)
    y_test_binary = np.where(y_test == 1, 0, 1).astype(np.int64)
    y_train_sign = np.where(y_train == 1, -1, 1).astype(np.int64)
    y_test_sign = np.where(y_test == 1, -1, 1).astype(np.int64)

    if channel_names is None:
        channel_names = [f"Ch{i:02d}" for i in range(1, N_EEG_CHANNELS + 1)]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{subject_id}_preprocessed.npz"
    np.savez_compressed(
        out_file,
        X_train=X_train,
        y_train=y_train_sign,
        y_train_binary=y_train_binary,
        y_train_raw=y_train,
        run_train=run_train,
        X_test=X_test,
        y_test=y_test_sign,
        y_test_binary=y_test_binary,
        y_test_raw=y_test,
        run_test=run_test,
        # unfiltered (but downsampled) epochs of the same windows, for visualisation
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        fs=FS,
        original_fs=ORIGINAL_FS,
        channels=np.asarray(list(channel_names)),
        subject=subject_id,
        classes=np.array([CLASS_NAMES[1], CLASS_NAMES[2]]),
        time_window=np.array([EPOCH_START, EPOCH_END]),
        freq_band=np.array([LOWCUT, HIGHCUT]),
        filter_order=FILTER_ORDER,
    )
    return {
        "dataset": "Dataset 4",
        "subject": subject_id,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "train_class_counts": np.bincount(y_train, minlength=3)[1:].tolist(),
        "test_class_counts": np.bincount(y_test, minlength=3)[1:].tolist(),
        "out_file": str(out_file),
    }


def discover_subjects(data_dir: Path):
    """Return (subjects with both S??T.mat and S??E.mat, list of unpaired file names)."""
    train_ids = {p.stem[:-1] for p in data_dir.glob("S??T.mat")}
    test_ids = {p.stem[:-1] for p in data_dir.glob("S??E.mat")}
    unpaired = [f"{i}T.mat" for i in sorted(train_ids - test_ids)] + [f"{i}E.mat" for i in sorted(test_ids - train_ids)]
    return sorted(train_ids & test_ids), unpaired


def run_preprocessing(
    data_dir: Path,
    output_dir: Path,
    subjects: Optional[Sequence[str]] = None,
    channel_names: Optional[Sequence[str]] = None,
) -> List[Dict]:
    found, unpaired = discover_subjects(data_dir)
    subject_ids = list(subjects) if subjects else found
    if not subject_ids:
        raise FileNotFoundError(f"No S??T.mat / S??E.mat pairs found in {data_dir.resolve()}.")

    print(f"\n{'=' * 75}")
    print("PREPROCESSING DATASET 4: Steyrl et al. 2016 (15 ch, 512 -> 256 Hz, [0.5-2.5 s])")
    print(f"Data directory  : {data_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Subjects        : {len(subject_ids)} -> {', '.join(subject_ids)}")
    print(f"{'=' * 75}\n")
    if unpaired:
        print(f"[warn] files without a T/E partner (ignored): {', '.join(unpaired)}\n")

    results, failed = [], []
    for subject_id in subject_ids:
        try:
            result = preprocess_subject(subject_id, data_dir, output_dir, channel_names)
        except Exception as exc:  # keep going with the remaining subjects
            failed.append((subject_id, repr(exc)))
            print(f"Subject {subject_id:<4} | FAILED: {exc!r}")
            continue
        results.append(result)
        print(
            f"Subject {subject_id:<4} | Train: {result['X_train_shape']} {result['train_class_counts']} | "
            f"Test: {result['X_test_shape']} {result['test_class_counts']}"
        )
    print(f"\nDone: {len(results)}/{len(subject_ids)} subjects preprocessed -> {output_dir.resolve()}")
    for sid, err in failed:
        print(f"Failed: {sid}: {err}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Dataset 4 (Steyrl et al. 2016, right hand vs. feet).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"Raw Dataset 4 directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_OUTPUT / "dataset_4",
                        help="Output directory for preprocessed Dataset 4 files.")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subjects to process, e.g. S01 S02 (default: every subject found).")
    parser.add_argument("--channel-names", type=str, default=None,
                        help="Optional comma-separated list of 15 electrode names in file order. "
                             "The .mat files carry no labels, so channels are named Ch01..Ch15 by default.")
    args = parser.parse_args()
    names = [n.strip() for n in args.channel_names.split(",")] if args.channel_names else None
    if names is not None and len(names) != N_EEG_CHANNELS:
        parser.error(f"--channel-names needs exactly {N_EEG_CHANNELS} names, got {len(names)}")
    run_preprocessing(args.data_dir, args.output_dir, args.subjects, names)


if __name__ == "__main__":
    main()