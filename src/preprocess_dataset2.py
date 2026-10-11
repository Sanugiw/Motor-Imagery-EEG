"""Preprocess BCI Competition IV Dataset 2a (Graz BCI)."""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data" / "BCICIV_2a"
DEFAULT_BASE_OUTPUT = (
    PROJECT_ROOT / "preprocessed_data"
    if (PROJECT_ROOT / "preprocessed_data").exists()
    else PROJECT_ROOT / "processed_data"
)

FS = 250
EPOCH_START = 0.5
EPOCH_END = 2.5
N_EEG_CHANNELS = 22
LOWCUT = 8.0
HIGHCUT = 30.0
FILTER_ORDER = 6

CLASS_NAMES = {
    1: "left hand",
    2: "right hand",
    3: "foot",
    4: "tongue",
}
BINARY_PAIRS = [
    ("LR", 1, 2),
    ("LF", 1, 3),
    ("LT", 1, 4),
    ("RF", 2, 3),
    ("RT", 2, 4),
    ("FT", 3, 4),
]


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


def load_gdf_session(
    gdf_path: Path,
    label_mat_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load a GDF session, its cue positions, labels, and EEG channels."""
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
    ch_names = raw.ch_names[:N_EEG_CHANNELS]
    eeg_data = np.asarray(raw.get_data()[:N_EEG_CHANNELS], dtype=np.float64) * 1e6
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    is_training_session = "T" in gdf_path.stem

    if is_training_session:
        cue_event_codes = [
            value for key, value in event_id.items()
            if key in ["769", "770", "771", "772"]
        ]
        cue_mask = np.isin(events[:, 2], cue_event_codes)
        cues = events[cue_mask][:, 0]
    else:
        code_783 = event_id.get("783")
        if code_783 is not None:
            cues = events[events[:, 2] == code_783][:, 0]
        else:
            # Evaluation data may omit the cue annotation; cue follows trial start by 2 s.
            code_768 = event_id.get("768")
            cues = events[events[:, 2] == code_768][:, 0] + 500

    if label_mat_path is not None and label_mat_path.exists():
        labels = sio.loadmat(str(label_mat_path))["classlabel"].flatten().astype(np.int64)
    else:
        inverse_map = {
            value: int(key) - 768
            for key, value in event_id.items()
            if key in ["769", "770", "771", "772"]
        }
        labels = np.asarray(
            [inverse_map[code] for code in events[cue_mask][:, 2]],
            dtype=np.int64,
        )

    return eeg_data, cues, labels, ch_names


def preprocess_subject(
    subject_id: str,
    data_dir: Path,
    output_dir: Path,
    true_labels_dir: Optional[Path] = None,
) -> Dict:
    """Preprocess both sessions for one Dataset 2a subject."""
    train_gdf = data_dir / f"{subject_id}T.gdf"
    test_gdf = data_dir / f"{subject_id}E.gdf"
    if not train_gdf.exists() or not test_gdf.exists():
        raise FileNotFoundError(f"Missing GDF files for {subject_id} in {data_dir.resolve()}.")

    labels_dir = true_labels_dir or (data_dir / "true_labels")
    train_label_mat = labels_dir / f"{subject_id}T.mat"
    test_label_mat = labels_dir / f"{subject_id}E.mat"
    eeg_train, cues_train, y_train_full, ch_names = load_gdf_session(train_gdf, train_label_mat)
    eeg_test, cues_test, y_test_full, _ = load_gdf_session(test_gdf, test_label_mat)

    filtered_train = apply_butterworth_bandpass(eeg_train)
    filtered_test = apply_butterworth_bandpass(eeg_test)
    start_offset = int(round(EPOCH_START * FS))
    end_offset = int(round(EPOCH_END * FS))
    expected_len = end_offset - start_offset

    def extract_epochs(filtered_data, cues, labels):
        epochs = []
        valid_labels = []
        for cue, label in zip(cues, labels):
            start = cue + start_offset
            end = cue + end_offset
            if start >= 0 and end <= filtered_data.shape[1]:
                epoch = filtered_data[:, start:end]
                if epoch.shape[1] == expected_len:
                    epochs.append(epoch)
                    valid_labels.append(label)
        return np.asarray(epochs, dtype=np.float32), np.asarray(valid_labels, dtype=np.int64)

    X_train_full, y_train_full = extract_epochs(filtered_train, cues_train, y_train_full)
    X_test_full, y_test_full = extract_epochs(filtered_test, cues_test, y_test_full)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_out_file = output_dir / f"{subject_id}_full_4class_preprocessed.npz"
    np.savez_compressed(
        full_out_file,
        X_train=X_train_full,
        y_train=y_train_full,
        X_test=X_test_full,
        y_test=y_test_full,
        fs=FS,
        channels=np.asarray(ch_names),
        subject=subject_id,
        classes=np.array(["left hand", "right hand", "foot", "tongue"]),
        time_window=np.array([EPOCH_START, EPOCH_END]),
        freq_band=np.array([LOWCUT, HIGHCUT]),
        filter_order=FILTER_ORDER,
    )

    binary_files = {}
    for pair_name, class1, class2 in BINARY_PAIRS:
        train_mask = np.isin(y_train_full, [class1, class2])
        test_mask = np.isin(y_test_full, [class1, class2])
        X_train, y_train = X_train_full[train_mask], y_train_full[train_mask]
        X_test, y_test = X_test_full[test_mask], y_test_full[test_mask]
        y_train_binary = np.where(y_train == class1, 0, 1).astype(np.int64)
        y_test_binary = np.where(y_test == class1, 0, 1).astype(np.int64)
        y_train_sign = np.where(y_train == class1, -1, 1).astype(np.int64)
        y_test_sign = np.where(y_test == class1, -1, 1).astype(np.int64)

        pair_file = output_dir / f"{subject_id}_{pair_name}_preprocessed.npz"
        np.savez_compressed(
            pair_file,
            X_train=X_train,
            y_train=y_train_sign,
            y_train_binary=y_train_binary,
            y_train_raw=y_train,
            X_test=X_test,
            y_test=y_test_sign,
            y_test_binary=y_test_binary,
            y_test_raw=y_test,
            fs=FS,
            channels=np.asarray(ch_names),
            subject=subject_id,
            pair=pair_name,
            classes=np.array([CLASS_NAMES[class1], CLASS_NAMES[class2]]),
            time_window=np.array([EPOCH_START, EPOCH_END]),
            freq_band=np.array([LOWCUT, HIGHCUT]),
            filter_order=FILTER_ORDER,
        )
        binary_files[pair_name] = str(pair_file)

    return {
        "dataset": "Dataset 2a",
        "subject": subject_id,
        "full_train_shape": X_train_full.shape,
        "full_test_shape": X_test_full.shape,
        "binary_subsets": binary_files,
        "full_file": str(full_out_file),
    }


def run_preprocessing(data_dir: Path, output_dir: Path) -> List[Dict]:
    """Preprocess all nine subjects in Dataset 2a."""
    subject_ids = [f"A0{i}" for i in range(1, 10)]
    print(f"\n{'=' * 75}")
    print("PREPROCESSING DATASET 2: BCI Competition IV Dataset 2a (22 ch, 250 Hz, [0.5-2.5 s])")
    print(f"Data directory  : {data_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"{'=' * 75}\n")

    results = []
    for subject_id in subject_ids:
        result = preprocess_subject(subject_id, data_dir, output_dir)
        results.append(result)
        print(
            f"Subject {subject_id:<4} | Full: Train {result['full_train_shape']} / "
            f"Test {result['full_test_shape']} | 6 Binary Pairs Generated (144 tr / 144 te each)"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BCI Competition IV Dataset 2a.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"Raw Dataset 2a directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE_OUTPUT / "BCICIV_2a",
                        help="Output directory for preprocessed Dataset 2a files.")
    args = parser.parse_args()
    run_preprocessing(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
