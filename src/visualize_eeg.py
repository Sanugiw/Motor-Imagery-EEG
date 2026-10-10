"""
EEG Inspection & Visualization script for BCI Competition IV Dataset 1.
Generates raw vs. preprocessed (8–30 Hz bandpass filtered) EEG signal plots,
one per subject, showing motor imagery cue onset and trial extraction window.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data" / "BCICIV_1_mat"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "plots" / "BCICIV_1"

FS = 100
LOWCUT = 8.0
HIGHCUT = 30.0
FILTER_ORDER = 6
EPOCH_START = 0.5
EPOCH_END = 3.5


def plot_subject_inspection(
    mat_path: Path,
    output_dir: Path,
    channel_name: str = "C3",
    trial_index: int = 0,
    pre_cue_sec: float = 1.0,
    post_cue_sec: float = 4.0,
):
    """
    Generate a 2-panel comparison of Raw vs. Preprocessed EEG for a given subject.
    """
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    cnt = np.asarray(mat["cnt"], dtype=np.float64)
    if cnt.shape[0] > cnt.shape[1]:
        cnt = cnt.T
    cnt *= 0.1  # Convert to µV

    mrk = mat["mrk"]
    nfo = mat["nfo"]
    pos = np.asarray(mrk.pos, dtype=int).reshape(-1) - 1
    y = np.asarray(mrk.y, dtype=int).reshape(-1)
    clab = [str(ch) for ch in nfo.clab]
    classes = [str(c) for c in nfo.classes]

    sub = mat_path.stem.replace("BCICIV_calib_ds", "")

    # 6th-order Butterworth bandpass filter (8-30 Hz)
    sos = butter(FILTER_ORDER, [LOWCUT, HIGHCUT], btype="bandpass", fs=FS, output="sos")
    filtered = sosfiltfilt(sos, cnt, axis=1)

    # Determine channel index
    if channel_name in clab:
        ch_idx = clab.index(channel_name)
    else:
        ch_idx = clab.index("Cz") if "Cz" in clab else 0
        channel_name = clab[ch_idx]

    # Select trial
    trial_idx = min(trial_index, len(pos) - 1)
    cue_sample = pos[trial_idx]
    label_val = y[trial_idx]
    cue_label = classes[0] if label_val == -1 else classes[1]

    # Window
    start_sample = max(0, int(cue_sample - pre_cue_sec * FS))
    end_sample = min(cnt.shape[1], int(cue_sample + post_cue_sec * FS))
    time_sec = (np.arange(start_sample, end_sample) - cue_sample) / FS

    raw_seg = cnt[ch_idx, start_sample:end_sample]
    filt_seg = filtered[ch_idx, start_sample:end_sample]

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)

    # 1. Raw EEG
    axes[0].plot(time_sec, raw_seg, color="#1f77b4", linewidth=1.1, label="Raw EEG")
    axes[0].axvline(0, color="#d62728", linestyle="--", linewidth=1.5, label=f"Cue Onset (t=0s: {cue_label})")
    axes[0].axvspan(EPOCH_START, EPOCH_END, color="#ff7f0e", alpha=0.18, label="Extraction Window [0.5–3.5s]")
    axes[0].set_ylabel("Amplitude (µV)", fontsize=11)
    axes[0].set_title(
        f"Subject {sub} — Raw EEG Signal (Electrode {channel_name}, Trial {trial_idx + 1} '{cue_label}')",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", framealpha=0.9)

    # 2. Preprocessed EEG
    axes[1].plot(time_sec, filt_seg, color="#2ca02c", linewidth=1.1, label="Filtered (8–30 Hz)")
    axes[1].axvline(0, color="#d62728", linestyle="--", linewidth=1.5, label=f"Cue Onset (t=0s: {cue_label})")
    axes[1].axvspan(EPOCH_START, EPOCH_END, color="#ff7f0e", alpha=0.18, label="Extraction Window [0.5–3.5s]")
    axes[1].set_ylabel("Amplitude (µV)", fontsize=11)
    axes[1].set_xlabel("Time Relative to Cue Onset (seconds)", fontsize=11)
    axes[1].set_title(
        f"Subject {sub} — Preprocessed EEG (6th-Order Butterworth 8–30 Hz, Zero-Phase)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"subject_{sub}_raw_vs_preprocessed.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved inspection plot: {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Generate Raw vs Preprocessed EEG inspection plots.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to raw dataset folder.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR, help="Directory to save plot images.")
    parser.add_argument("--channel", type=str, default="C3", help="Electrode channel to plot (default: C3).")
    parser.add_argument("--trial", type=int, default=0, help="Trial index to visualize (default: 0).")
    args = parser.parse_args()

    calib_files = sorted(args.data_dir.glob("BCICIV_calib_ds1*.mat"))
    if not calib_files:
        raise FileNotFoundError(f"No calibration files found in {args.data_dir.resolve()}.")

    print(f"Generating inspection plots for {len(calib_files)} subjects...")
    for f in calib_files:
        plot_subject_inspection(
            mat_path=f,
            output_dir=args.output_dir,
            channel_name=args.channel,
            trial_index=args.trial,
        )
    print(f"\nAll plots saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

