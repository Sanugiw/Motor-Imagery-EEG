"""Load saved, standardized preprocessed trials for the results pipeline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
PREPROCESSED = ROOT / "preprocessed_data"


@dataclass
class BinaryTrials:
    subject: str
    pair: str
    fs: float
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


DATASET_PATHS = {
    "1": ("dataset_1", "BCICIV_calib_ds1*_preprocessed.npz"),
    "2a": ("dataset_2", "A0[1-9]_[A-Z][A-Z]_preprocessed.npz"),
    # Paper protocol: session 03T only (B0103T..B0903T), split 80/80.
    "2b": ("dataset_3", "B0?03T_preprocessed.npz"),
    "4": ("dataset_4", "S*_preprocessed.npz"),
}


def _value(data, *keys):
    for key in keys:
        if key in data:
            value = data[key]
            return value.item() if np.asarray(value).ndim == 0 else value
    raise KeyError(f"Missing expected preprocessed field; tried {keys}.")


def _load_one(path: Path) -> BinaryTrials:
    with np.load(path, allow_pickle=True) as data:
        # Dataset 2b's established archive names explicitly mark filtered arrays.
        raw_train = _value(data, "X_train_raw")
        raw_test = _value(data, "X_test_raw")
        filtered_train = _value(data, "X_train", "X_train_filt")
        filtered_test = _value(data, "X_test", "X_test_filt")
        labels_train = _value(data, "y_train_binary", "y_train")
        labels_test = _value(data, "y_test_binary", "y_test")
        return BinaryTrials(
            subject=str(_value(data, "subject", "subject_id")),
            pair=str(_value(data, "pair")) if "pair" in data else "",
            fs=float(_value(data, "fs", "sample_rate")),
            X_train_raw=np.asarray(raw_train),
            X_test_raw=np.asarray(raw_test),
            X_train=np.asarray(filtered_train),
            X_test=np.asarray(filtered_test),
            y_train=np.asarray(labels_train, dtype=np.int64),
            y_test=np.asarray(labels_test, dtype=np.int64),
        )


def iter_dataset_trials(dataset: str, data_dir=None):
    """Yield saved trial sets. ``data_dir`` overrides the dataset output folder."""
    try:
        folder, pattern = DATASET_PATHS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 1, 2a, 2b, or 4.") from exc
    directory = Path(data_dir) if data_dir else PREPROCESSED / folder
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No preprocessed Dataset {dataset} files found in {directory.resolve()}. "
            "Run the matching src.preprocessing.preprocess_dataset* script first."
        )
    for path in paths:
        try:
            yield _load_one(path)
        except KeyError as exc:
            raise ValueError(
                f"{path.name} does not include the required raw and filtered arrays. "
                f"Rerun the matching preprocessing script to regenerate it."
            ) from exc
