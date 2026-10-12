"""Generate raw-versus-filtered EEG inspection plots for both datasets."""

import argparse
import warnings
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR_DS1 = PROJECT_ROOT / "Data" / "dataset_1"
DEFAULT_DATA_DIR_DS2 = PROJECT_ROOT / "Data" / "dataset_2"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "plots"

LOWCUT = 8.0
HIGHCUT = 30.0
FILTER_ORDER = 6
FS_DS1 = 100
FS_DS2 = 250
EPOCH_START = 0.5
EPOCH_END_DS1 = 3.5
EPOCH_END_DS2 = 2.5
N_EEG_CHANNELS_DS2 = 22

DS2_EVENT_LABELS = {
    "769": "Left hand",
    "770": "Right hand",
    "771": "Foot",
    "772": "Tongue",
}


def _filter_signal(data: np.ndarray, fs: float) -> np.ndarray:
    sos = butter(FILTER_ORDER, [LOWCUT, HIGHCUT], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=1)


def _select_channel(ch_names: Sequence[str], requested: str) -> Tuple[int, str]:
    """Match channel names while ignoring common EEG prefixes and casing."""
    target = requested.casefold().removeprefix("eeg-")
    normalized = [name.casefold().removeprefix("eeg-") for name in ch_names]
    if target in normalized:
        index = normalized.index(target)
    elif "cz" in normalized:
        index = normalized.index("cz")
    else:
        index = 0
    return index, ch_names[index]


def _save_inspection_plot(
    raw_segment: np.ndarray,
    filtered_segment: np.ndarray,
    time_sec: np.ndarray,
    output_path: Path,
    subject: str,
    channel_name: str,
    trial_number: int,
    task_label: str,
    epoch_end: float,
    fs: float,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    axes[0].plot(time_sec, raw_segment, color="#1f77b4", linewidth=1.1, label="Raw EEG")
    axes[1].plot(
        time_sec,
        filtered_segment,
        color="#2ca02c",
        linewidth=1.1,
        label=f"Filtered ({LOWCUT:g}–{HIGHCUT:g} Hz)",
    )

    for axis in axes:
        axis.axvline(
            0,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label=f"Cue onset (t=0 s: {task_label})",
        )
        axis.axvspan(
            EPOCH_START,
            epoch_end,
            color="#ff7f0e",
            alpha=0.18,
            label=f"Epoch window [{EPOCH_START:.1f}–{epoch_end:.1f} s]",
        )
        axis.set_ylabel("Amplitude (µV)", fontsize=11)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", framealpha=0.9)

    axes[0].set_title(
        f"Subject {subject} — Raw EEG (Electrode {channel_name}, "
        f"Trial {trial_number}: {task_label})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_title(
        f"Subject {subject} — Preprocessed EEG "
        f"(6th-order zero-phase Butterworth {LOWCUT:g}–{HIGHCUT:g} Hz)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Time relative to cue onset (seconds)", fontsize=11)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved inspection plot: {output_path}")
    return output_path


def plot_dataset1_subject(
    mat_path: Path,
    output_dir: Path,
    channel_name: str = "C3",
    trial_index: int = 0,
    pre_cue_sec: float = 1.0,
    post_cue_sec: float = 4.0,
) -> Path:
    """Plot raw and filtered EEG around one Dataset 1 cue."""
    if trial_index < 0:
        raise ValueError("Trial index must be zero or greater.")
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    cnt = np.asarray(mat["cnt"], dtype=np.float64)
    if cnt.shape[0] > cnt.shape[1]:
        cnt = cnt.T
    cnt *= 0.1  # Convert to microvolts.
    mrk, nfo = mat["mrk"], mat["nfo"]
    positions = np.asarray(mrk.pos, dtype=int).reshape(-1) - 1
    labels = np.asarray(mrk.y, dtype=int).reshape(-1)
    ch_names = [str(ch) for ch in nfo.clab]
    classes = [str(label) for label in nfo.classes]
    if len(positions) == 0:
        raise ValueError(f"No cue markers found in {mat_path}.")

    trial_index = min(trial_index, len(positions) - 1)
    cue_sample = positions[trial_index]
    task_label = classes[0] if labels[trial_index] == -1 else classes[1]
    channel_index, channel_name = _select_channel(ch_names, channel_name)
    filtered = _filter_signal(cnt, FS_DS1)

    start = max(0, int(cue_sample - pre_cue_sec * FS_DS1))
    end = min(cnt.shape[1], int(cue_sample + post_cue_sec * FS_DS1))
    time_sec = (np.arange(start, end) - cue_sample) / FS_DS1
    subject = mat_path.stem.replace("BCICIV_calib_ds", "")
    out_file = output_dir / f"subject_{subject}_raw_vs_preprocessed.png"
    return _save_inspection_plot(
        cnt[channel_index, start:end],
        filtered[channel_index, start:end],
        time_sec,
        out_file,
        subject,
        channel_name,
        trial_index + 1,
        task_label,
        EPOCH_END_DS1,
        FS_DS1,
    )


def plot_dataset2_subject(
    data_dir: Path,
    output_dir: Path,
    subject: str,
    channel_name: str = "C3",
    trial_index: int = 0,
    pre_cue_sec: float = 1.0,
    post_cue_sec: float = 4.0,
) -> Path:
    """Plot raw and filtered EEG around one Dataset 2a training cue."""
    if trial_index < 0:
        raise ValueError("Trial index must be zero or greater.")
    gdf_path = data_dir / f"{subject}T.gdf"
    if not gdf_path.exists():
        raise FileNotFoundError(f"Training GDF file not found: {gdf_path.resolve()}.")

    # Some Dataset 2a GDF files use the generic label "EEG" for multiple
    # channels. MNE renames duplicates automatically; hide only that warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Channel names are not unique, found duplicates",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
    ch_names = raw.ch_names[:N_EEG_CHANNELS_DS2]
    data = raw.get_data()[:N_EEG_CHANNELS_DS2] * 1e6  # Convert volts to microvolts.
    fs = float(raw.info["sfreq"])
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    cue_code_to_label = {
        event_id[description]: label
        for description, label in DS2_EVENT_LABELS.items()
        if description in event_id
    }
    cue_events = events[np.isin(events[:, 2], list(cue_code_to_label))]
    if len(cue_events) == 0:
        raise ValueError(f"No motor imagery cue events found in {gdf_path}.")

    trial_index = min(trial_index, len(cue_events) - 1)
    cue_sample = int(cue_events[trial_index, 0])
    event_code = int(cue_events[trial_index, 2])
    task_label = cue_code_to_label[event_code]
    channel_index, channel_name = _select_channel(ch_names, channel_name)
    filtered = _filter_signal(data, fs)

    start = max(0, int(cue_sample - pre_cue_sec * fs))
    end = min(data.shape[1], int(cue_sample + post_cue_sec * fs))
    time_sec = (np.arange(start, end) - cue_sample) / fs
    out_file = output_dir / f"subject_{subject}_raw_vs_preprocessed.png"
    return _save_inspection_plot(
        data[channel_index, start:end],
        filtered[channel_index, start:end],
        time_sec,
        out_file,
        subject,
        channel_name,
        trial_index + 1,
        task_label,
        EPOCH_END_DS2,
        fs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate raw-versus-filtered EEG plots for Dataset 1 and Dataset 2a."
    )
    parser.add_argument(
        "--dataset",
        choices=["1", "2a", "all"],
        default="all",
        help="Dataset to plot: 1, 2a, or all (default: all).",
    )
    parser.add_argument("--data-dir-ds1", type=Path, default=DEFAULT_DATA_DIR_DS1)
    parser.add_argument("--data-dir-ds2", type=Path, default=DEFAULT_DATA_DIR_DS2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR,
                        help="Base directory for plots (dataset-specific folders are created).")
    parser.add_argument("--channel", default="C3", help="Electrode to plot (default: C3).")
    parser.add_argument("--trial", type=int, default=0, help="Zero-based cue index (default: 0).")
    parser.add_argument("--pre-cue-sec", type=float, default=1.0)
    parser.add_argument("--post-cue-sec", type=float, default=4.0)
    args = parser.parse_args()

    if args.dataset in ["1", "all"]:
        files = sorted(args.data_dir_ds1.glob("BCICIV_calib_ds1*.mat"))
        if not files:
            raise FileNotFoundError(f"No Dataset 1 calibration files found in {args.data_dir_ds1.resolve()}.")
        out_dir = args.output_dir / "dataset_1"
        print(f"Generating Dataset 1 inspection plots for {len(files)} subjects...")
        for mat_path in files:
            plot_dataset1_subject(
                mat_path,
                out_dir,
                args.channel,
                args.trial,
                args.pre_cue_sec,
                args.post_cue_sec,
            )

    if args.dataset in ["2a", "all"]:
        subjects = [f"A0{i}" for i in range(1, 10)]
        if not any((args.data_dir_ds2 / f"{subject}T.gdf").exists() for subject in subjects):
            raise FileNotFoundError(f"No Dataset 2a training GDF files found in {args.data_dir_ds2.resolve()}.")
        out_dir = args.output_dir / "dataset_2"
        print(f"Generating Dataset 2a inspection plots for {len(subjects)} subjects...")
        for subject in subjects:
            plot_dataset2_subject(
                args.data_dir_ds2,
                out_dir,
                subject,
                args.channel,
                args.trial,
                args.pre_cue_sec,
                args.post_cue_sec,
            )

    print(f"\nAll requested plots saved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
