"""Preprocess Dataset 3: BCI Competition IV Dataset 2b (Graz BCI, B01-B09).

Follows Zhang et al., Sensors 2020, 20, 4749 (Sections 2.1 and 2.3):
  * 3 EEG channels (C3, Cz, C4) at 250 Hz; the 3 EOG channels are dropped.
  * The paper uses only the third training session of each subject (B0103T ...
    B0903T): 160 trials (80 left / 80 right), the first 80 trials = training set,
    the remaining 80 = test set.  This script processes EVERY session file it
    finds (45 files = 9 subjects x 5 sessions: 01T, 02T, 03T, 04E, 05E) with that
    same rule (first 80 trials -> train, remaining trials -> test), so the paper's
    protocol is simply the *03T files (use --sessions 3 to process only those).
  * Sessions 01T/02T have 120 trials (-> 80 train / 40 test), 03T-05E have 160.
  * The evaluation files (04E, 05E) usually carry no class labels in the GDF.
    Put the competition's "true labels" files next to them
    (data/dataset_3/B0104E.mat or data/dataset_3/true_labels/B0104E.mat, variable
    `classlabel`).  Files without labels are skipped and reported at the end.
  * 6th-order Butterworth band-pass, 8-30 Hz (zero-phase).
  * Single-trial segmentation: 0.5-2.5 s after the motor-imagery cue.
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# NumPy 2.x removed binary-input support from np.fromstring. Older MNE GDF
# readers still rely on it, so route binary values through np.frombuffer.
_orig_fromstring = np.fromstring


def _compat_fromstring(string, dtype=float, count=-1, sep="", *args, **kwargs):
    if sep == "":
        if isinstance(string, str):
            string = string.encode("latin-1")
        if isinstance(string, (bytes, bytearray, memoryview)):
            return np.frombuffer(string, dtype=dtype, count=count)
    return _orig_fromstring(string, dtype=dtype, count=count, sep=sep, *args, **kwargs)


np.fromstring = _compat_fromstring

import mne
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt

warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_DATA_DIR = _first_existing(
    PROJECT_ROOT / "data" / "dataset_3",
    PROJECT_ROOT / "Data" / "dataset_3",
)
DEFAULT_BASE_OUTPUT = (
    PROJECT_ROOT / "preprocessed_data"
    if (PROJECT_ROOT / "preprocessed_data").exists()
    else PROJECT_ROOT / "processed_data"
)

FS = 250
EPOCH_START = 0.5
EPOCH_END = 2.5
N_EEG_CHANNELS = 3
N_TRAIN = 80
N_TEST = 80
LOWCUT = 8.0
HIGHCUT = 30.0
FILTER_ORDER = 6
DEFAULT_PATTERN = "B0?0?[TE].gdf"  # every session (01T,02T,03T,04E,05E) of every subject

# GDF event codes of BCI Competition IV 2b
CODE_TRIAL_START = "768"
CODE_ARTIFACT = "1023"  # trial rejected by the data providers (artifact)
CODE_CUE_UNKNOWN = "783"  # cue with unknown class (evaluation files)
CUE_DELAY_S = 3.0  # the cue appears 3 s after the trial start (768)
CUE_CODES = {"769": 1, "770": 2}  # 1 = left hand, 2 = right hand
CLASS_NAMES = {1: "left hand", 2: "right hand"}


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


def _interpolate_nans(data: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN gaps (if any) channel by channel."""
    if not np.isnan(data).any():
        return data
    x = np.arange(data.shape[1])
    for ch in range(data.shape[0]):
        bad = np.isnan(data[ch])
        if bad.any():
            data[ch, bad] = np.interp(x[bad], x[~bad], data[ch, ~bad])
    return data


class MissingLabelsError(Exception):
    """Raised when a file has cues but no class labels (evaluation files)."""


def find_label_file(gdf_path: Path) -> Optional[Path]:
    """Look for B0104E.mat next to the GDF or in a true_labels sub-folder."""
    for candidate in (
        gdf_path.with_suffix(".mat"),
        gdf_path.parent / "true_labels" / f"{gdf_path.stem}.mat",
    ):
        if candidate.exists():
            return candidate
    return None


def extract_cues(
    samples: np.ndarray,
    desc: np.ndarray,
    fs: float,
    true_labels: Optional[np.ndarray] = None,
):
    """Return (cue samples, labels 1/2, artifact flags, label source) from GDF annotations.

    * Training files: cue = event 769 (left) / 770 (right); labels come from the events.
    * Evaluation files: no 769/770.  Cue = event 783 if present, else trial start
      (768) + 3 s; labels must come from the competition's true-label file.
    """
    trial_starts = np.sort(samples[desc == CODE_TRIAL_START])
    artifact_samples = samples[desc == CODE_ARTIFACT]

    cue_mask = np.isin(desc, list(CUE_CODES))
    if cue_mask.any():
        order = np.argsort(samples[cue_mask], kind="stable")
        cues = samples[cue_mask][order]
        labels = np.array([CUE_CODES[d] for d in desc[cue_mask][order]], dtype=np.int64)
        source = "events"
    else:
        unknown = np.sort(samples[desc == CODE_CUE_UNKNOWN])
        if len(unknown):
            cues = unknown
        elif len(trial_starts):
            cues = trial_starts + int(round(CUE_DELAY_S * fs))
        else:
            raise ValueError("no cue (769/770/783) or trial-start (768) events found")
        if true_labels is None:
            raise MissingLabelsError("no class labels in the GDF and no true-label .mat file found")
        labels = np.asarray(true_labels).reshape(-1).astype(np.int64)
        if len(labels) != len(cues):
            raise ValueError(f"{len(cues)} cues but {len(labels)} labels in the label file")
        source = "label file"

    # A trial is flagged as artifact if a 1023 marker sits between its
    # trial-start (768) and its cue.
    artifact = np.zeros(len(cues), dtype=bool)
    for i, cue in enumerate(cues):
        prior = trial_starts[trial_starts <= cue]
        start = prior[-1] if len(prior) else cue
        artifact[i] = bool(np.any((artifact_samples >= start) & (artifact_samples <= cue)))
    return cues, labels, artifact, source


def load_gdf_session(gdf_path: Path) -> Dict:
    """Load one 2b GDF file: EEG (uV), cue positions, labels, artifact flags."""
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
    fs = float(raw.info["sfreq"])

    eeg_idx = [i for i, name in enumerate(raw.ch_names) if name.upper().startswith("EEG")]
    eeg_idx = eeg_idx[:N_EEG_CHANNELS]
    if len(eeg_idx) != N_EEG_CHANNELS:
        raise ValueError(f"{gdf_path.name}: expected {N_EEG_CHANNELS} EEG channels, got {raw.ch_names}")
    ch_names = [raw.ch_names[i].split(":")[-1] for i in eeg_idx]  # 'EEG:C3' -> 'C3'
    eeg = np.asarray(raw.get_data(picks=eeg_idx), dtype=np.float64) * 1e6  # V -> uV
    eeg = _interpolate_nans(eeg)

    ann = raw.annotations
    samples = np.round(np.asarray(ann.onset) * fs).astype(int)
    desc = np.asarray(ann.description)

    true_labels = None
    label_file = find_label_file(gdf_path)
    if label_file is not None:
        true_labels = sio.loadmat(str(label_file))["classlabel"].flatten()
    cues, labels, artifact, source = extract_cues(samples, desc, fs, true_labels)

    return {
        "eeg": eeg,
        "cues": cues,
        "labels": labels,
        "artifact": artifact,
        "fs": fs,
        "channels": ch_names,
        "label_source": source,
    }


def extract_epochs(
    data: np.ndarray,
    cues: np.ndarray,
    labels: np.ndarray,
    artifact: np.ndarray,
    fs: float = FS,
    epoch_start: float = EPOCH_START,
    epoch_end: float = EPOCH_END,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut [cue + epoch_start, cue + epoch_end) windows out of continuous data."""
    start_offset = int(round(epoch_start * fs))
    end_offset = int(round(epoch_end * fs))
    expected_len = end_offset - start_offset
    epochs, valid_labels, valid_art = [], [], []
    for cue, label, art in zip(cues, labels, artifact):
        start, end = cue + start_offset, cue + end_offset
        if start >= 0 and end <= data.shape[1]:
            epoch = data[:, start:end]
            if epoch.shape[1] == expected_len:
                epochs.append(epoch)
                valid_labels.append(label)
                valid_art.append(art)
    return (
        np.asarray(epochs, dtype=np.float32),
        np.asarray(valid_labels, dtype=np.int64),
        np.asarray(valid_art, dtype=bool),
    )


def preprocess_subject(gdf_path: Path, output_dir: Path, n_train: int = N_TRAIN, n_test: int = N_TEST) -> Dict:
    """Preprocess one 2b session file and save its training/test epochs."""
    session = load_gdf_session(gdf_path)
    fs = session["fs"]
    if int(round(fs)) != FS:
        raise ValueError(f"{gdf_path.name}: expected fs={FS} Hz, found {fs} Hz")

    filtered = apply_butterworth_bandpass(session["eeg"], fs=FS)
    X, y, art = extract_epochs(filtered, session["cues"], session["labels"], session["artifact"])
    X_raw, _, _ = extract_epochs(session["eeg"], session["cues"], session["labels"], session["artifact"])

    if len(X) < n_train:
        raise ValueError(f"{gdf_path.name}: only {len(X)} trials, need at least {n_train} for training")
    sl_train = slice(0, n_train)
    sl_test = slice(n_train, n_train + n_test)
    X_train, y_train, art_train = X[sl_train], y[sl_train], art[sl_train]
    X_test, y_test, art_test = X[sl_test], y[sl_test], art[sl_test]

    # Label conventions identical to the Dataset 2a script:
    # raw 1/2 -> binary 0/1 and sign -1/+1.
    y_train_binary = np.where(y_train == 1, 0, 1).astype(np.int64)
    y_test_binary = np.where(y_test == 1, 0, 1).astype(np.int64)
    y_train_sign = np.where(y_train == 1, -1, 1).astype(np.int64)
    y_test_sign = np.where(y_test == 1, -1, 1).astype(np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{gdf_path.stem}_preprocessed.npz"
    np.savez_compressed(
        out_file,
        X_train=X_train,
        y_train=y_train_sign,
        y_train_binary=y_train_binary,
        y_train_raw=y_train,
        artifact_train=art_train,
        X_test=X_test,
        y_test=y_test_sign,
        y_test_binary=y_test_binary,
        y_test_raw=y_test,
        artifact_test=art_test,
        # unfiltered epochs of the same windows (only used for visualisation)
        X_train_raw=X_raw[sl_train],
        X_test_raw=X_raw[sl_test],
        fs=FS,
        channels=np.asarray(session["channels"]),
        subject=gdf_path.stem[:3],
        session=gdf_path.stem[3:5],
        classes=np.array([CLASS_NAMES[1], CLASS_NAMES[2]]),
        time_window=np.array([EPOCH_START, EPOCH_END]),
        freq_band=np.array([LOWCUT, HIGHCUT]),
        filter_order=FILTER_ORDER,
    )
    return {
        "dataset": "Dataset 3",
        "subject": gdf_path.stem,
        "channels": session["channels"],
        "n_trials_total": len(X),
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "train_class_counts": np.bincount(y_train, minlength=3)[1:].tolist(),
        "test_class_counts": np.bincount(y_test, minlength=3)[1:].tolist(),
        "n_artifact_trials": int(art.sum()),
        "label_source": session["label_source"],
        "out_file": str(out_file),
    }


def run_preprocessing(data_dir: Path, output_dir: Path, pattern: str = DEFAULT_PATTERN) -> List[Dict]:
    """Preprocess every matching session file of Dataset 3 (one bad file never stops the run)."""
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir.resolve()}.")

    print(f"\n{'=' * 75}")
    print("PREPROCESSING DATASET 3: BCI Competition IV Dataset 2b (3 ch, 250 Hz, [0.5-2.5 s])")
    print(f"Data directory  : {data_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"File pattern    : {pattern}  ->  {len(files)} files found")
    print(f"{'=' * 75}\n")

    results, skipped, failed = [], [], []
    for gdf_path in files:
        try:
            result = preprocess_subject(gdf_path, output_dir)
        except MissingLabelsError as exc:
            skipped.append(gdf_path.name)
            print(f"{gdf_path.stem:<8} | SKIPPED: {exc}")
            continue
        except Exception as exc:  # keep going with the remaining files
            failed.append((gdf_path.name, repr(exc)))
            print(f"{gdf_path.stem:<8} | FAILED : {exc!r}")
            continue
        results.append(result)
        print(
            f"{result['subject']:<8} | Train: {result['X_train_shape']} {result['train_class_counts']} | "
            f"Test: {result['X_test_shape']} {result['test_class_counts']} | "
            f"Artifact-flagged (kept): {result['n_artifact_trials']:>2} | labels: {result['label_source']}"
        )

    print(f"\nDone: {len(results)}/{len(files)} files preprocessed -> {output_dir.resolve()}")
    if skipped:
        print(f"Skipped (no class labels): {', '.join(skipped)}")
        print("  -> download the competition's true labels (B0104E.mat, B0105E.mat, ...) and put them in\n"
              f"     {data_dir / 'true_labels'} (or next to the GDF files), then re-run.")
    for name, err in failed:
        print(f"Failed: {name}: {err}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BCI Competition IV Dataset 2b (paper's Dataset 3).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"Raw Dataset 3 directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_OUTPUT / "dataset_3",
                        help="Output directory for preprocessed Dataset 3 files.")
    parser.add_argument("--sessions", type=int, nargs="*", default=None,
                        help="Only these sessions, e.g. --sessions 3 for the paper's B0103T..B0903T "
                             "(default: all five sessions of every subject).")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Custom glob instead of --sessions, e.g. --pattern B0101T.gdf")
    args = parser.parse_args()
    if args.pattern:
        pattern = args.pattern
    elif args.sessions:
        pattern = "B0?0[" + "".join(str(i) for i in args.sessions) + "]?.gdf"
    else:
        pattern = DEFAULT_PATTERN
    run_preprocessing(args.data_dir, args.output_dir, pattern)


if __name__ == "__main__":
    main()