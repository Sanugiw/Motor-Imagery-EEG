"""Preprocess BCI Competition IV Dataset IIb (paper Dataset 3)."""

import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

# NumPy 2 removed binary-input support from fromstring. MNE's GDF reader may
# pass binary header fields as Latin-1 strings, so restore their original bytes.
_numpy_fromstring = np.fromstring


def _compatible_fromstring(string, dtype=float, count=-1, sep="", *args, **kwargs):
    if sep == "":
        if isinstance(string, str):
            string = string.encode("latin-1")
        if isinstance(string, (bytes, bytearray, memoryview)):
            return np.frombuffer(string, dtype=dtype, count=count)
    return _numpy_fromstring(string, dtype=dtype, count=count, sep=sep, *args, **kwargs)


np.fromstring = _compatible_fromstring

import mne

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = ROOT / "Data" / "dataset_3"
DEFAULT_OUTPUT_DIR = ROOT / "preprocessed_data" / "dataset_3"
FS = 250.0
LOW, HIGH = 8.0, 30.0
START, END = 0.5, 2.5
TRAIN_TRIALS = 80


def report_saved_trials(subject, X_train, X_test, output_file: Path, task=None):
    """Print one consistent summary line for a saved trial set."""
    name = f"{subject}-{task}" if task else str(subject)
    print(
        f"Subject {name:<8} | Train: {X_train.shape} | Test: {X_test.shape} "
        f"| Saved: {output_file.resolve()}"
    )


def read_gdf(path, preload=True):
    """Read GDF with MNE while filtering known header/channel warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Channel names are not unique, found duplicates",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Highpass cutoff frequency .* is greater than lowpass cutoff frequency .*",
            category=RuntimeWarning,
        )
        return mne.io.read_raw_gdf(str(path), preload=preload, verbose=False)


def events_from_annotations(raw):
    return mne.events_from_annotations(raw, verbose=False)


def butterworth_bandpass(data, fs, low=8.0, high=30.0, order=6):
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(data), axis=1)


def extract_epochs(data, cues, labels, fs, start=0.5, end=2.5):
    start_offset = int(round(start * fs))
    end_offset = int(round(end * fs))
    expected_length = end_offset - start_offset
    epochs, kept_labels = [], []
    for cue, label in zip(cues, labels):
        left, right = int(cue) + start_offset, int(cue) + end_offset
        if left >= 0 and right <= data.shape[1]:
            epoch = data[:, left:right]
            if epoch.shape[1] == expected_length:
                epochs.append(epoch)
                kept_labels.append(label)
    return np.asarray(epochs, dtype=np.float32), np.asarray(kept_labels, dtype=np.int64)


def preprocess_subject(subject_id: str, data_dir: Path, output_dir: Path):
    path = data_dir / f"{subject_id}03T.gdf"
    if not path.exists():
        raise FileNotFoundError(f"Missing Dataset 3 recording: {path}")
    raw = read_gdf(path, preload=True)
    preferred = [name for name in ("EEG:C3", "EEG:Cz", "EEG:C4") if name in raw.ch_names]
    picks = preferred or raw.ch_names[:3]
    raw.pick(picks)
    fs = float(raw.info["sfreq"])
    continuous = np.asarray(raw.get_data(), dtype=np.float64) * 1e6
    events, event_id = events_from_annotations(raw)
    cue_codes = {"769": 1, "770": 2}
    code_ids = {event_id[name]: label for name, label in cue_codes.items() if name in event_id}
    if len(code_ids) != 2:
        raise ValueError(f"Expected left/right cue codes 769 and 770 in {path}; got {list(event_id)}")
    cue_mask = np.isin(events[:, 2], list(code_ids))
    cues = events[cue_mask, 0]
    labels = np.asarray([code_ids[code] for code in events[cue_mask, 2]], dtype=np.int64)
    if len(cues) != 160:
        raise ValueError(f"Expected 160 labeled trials for {subject_id}; found {len(cues)}")

    filtered = butterworth_bandpass(continuous, fs, LOW, HIGH, order=6)
    X_raw, y = extract_epochs(continuous, cues, labels, fs, START, END)
    X_filtered, y_filtered = extract_epochs(filtered, cues, labels, fs, START, END)
    if not np.array_equal(y, y_filtered):
        raise ValueError(f"Raw and filtered epochs do not align for {subject_id}.")
    if len(y) != 160:
        raise ValueError(f"Only {len(y)} complete epochs were extracted for {subject_id}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{subject_id}_preprocessed.npz"
    y_train = y[:TRAIN_TRIALS]
    y_test = y[TRAIN_TRIALS:]
    np.savez_compressed(
        out,
        X_train_raw=X_raw[:TRAIN_TRIALS], X_test_raw=X_raw[TRAIN_TRIALS:],
        X_train=X_filtered[:TRAIN_TRIALS], X_test=X_filtered[TRAIN_TRIALS:],
        y_train=y_train, y_test=y_test,
        y_train_binary=(y_train - 1), y_test_binary=(y_test - 1),
        fs=fs, subject=subject_id, pair="",
        channels=np.asarray([name.replace("EEG:", "") for name in picks]),
        classes=np.asarray(["left hand", "right hand"]),
        time_window=np.array([START, END]), freq_band=np.array([LOW, HIGH]),
        filter_order=6,
    )
    report_saved_trials(subject_id, X_raw[:TRAIN_TRIALS], X_raw[TRAIN_TRIALS:], out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", nargs="+", default=[f"B{i:02d}" for i in range(1, 10)])
    args = parser.parse_args()
    for subject in args.subjects:
        preprocess_subject(subject, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
