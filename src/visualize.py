"""
Visualization module for the motor-imagery EEG project (Datasets 1, 2a, 3 and 4).

Two families of figures
-----------------------
A. Raw-vs-filtered inspection plot (all four datasets, straight from the raw files):
   one cue with 1 s before / 4 s after, raw EEG on top, the 8-30 Hz zero-phase
   Butterworth-filtered EEG below, cue onset and the extracted epoch window marked.
   Output: plots/BCICIV_1, plots/BCICIV_2a, plots/dataset_3/<file>, plots/dataset_4/<subject>

B. Epoch-level figures from the preprocessed .npz files (Datasets 3 and 4 only):
   1. Butterworth band-pass frequency response (single pass and effective zero-phase)
   2. Segmentation overview: continuous EEG with cue/onset markers and extracted windows
   3. Single-trial raw vs. filtered signals (one example per class)
   4. Power spectral density before and after the 8-30 Hz filtering
   5. Class-wise band power over the epoch
   (the filter-response plot is produced for every dataset)

Units: everything is plotted in microvolts.

Command line (run from the project root)
----------------------------------------
    python src/visualize.py                              # datasets 1, 2a, 3 and 4 (everything)
    python src/visualize.py --dataset 1                  # BCI IV dataset 1, all subjects
    python src/visualize.py --dataset 2a --subject A01
    python src/visualize.py --dataset 3                  # all 45 files (9 subjects x 5 sessions)
    python src/visualize.py --dataset 3 --session 3      # only B0103T ... B0903T (paper)
    python src/visualize.py --dataset 3 --subject B01    # all sessions of B01
    python src/visualize.py --dataset 4                  # all 14 subjects
    python src/visualize.py --dataset 4 --subject S01 --no-segmentation
"""

import argparse
import os
import traceback
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_COLORS = ["#1f77b4", "#ff7f0e"]


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _data_dir(*names: str) -> Path:
    """First existing raw-data folder among data/<name> and Data/<name>."""
    return _first_existing(*[PROJECT_ROOT / base / n for n in names for base in ("data", "Data")])


DEFAULT_DATA_DIR_DS1 = _data_dir("BCICIV_1", "BCICIV_1_mat")
DEFAULT_DATA_DIR_DS2 = _data_dir("BCICIV_2a")
DEFAULT_DATA_DIR_DS3 = _data_dir("dataset_3")
DEFAULT_DATA_DIR_DS4 = _data_dir("dataset_4")
DEFAULT_PREPROCESSED_DIR = _first_existing(PROJECT_ROOT / "preprocessed_data", PROJECT_ROOT / "processed_data")
DEFAULT_PLOT_DIR = PROJECT_ROOT / "plots"

LOWCUT = 8.0
HIGHCUT = 30.0
FILTER_ORDER = 6
FS_DS1 = 100
FS_DS2 = 250
FS_DS3 = 250
FS_DS4 = 256
EPOCH_START = 0.5
EPOCH_END_DS1 = 3.5
EPOCH_END_DS2 = 2.5
EPOCH_END_DS3 = 2.5
EPOCH_END_DS4 = 2.5
N_EEG_CHANNELS_DS2 = 22
DS4_DEFAULT_CHANNEL = 5  # 0-based; guess at the C3-like electrode (names are not stored in the .mat files)

DS2_EVENT_LABELS = {
    "769": "Left hand",
    "770": "Right hand",
    "771": "Foot",
    "772": "Tongue",
}
DS3_CLASS_LABELS = {1: "Left hand", 2: "Right hand"}
DS4_CLASS_LABELS = {1: "Right hand", 2: "Feet"}


def _filter_signal(data: np.ndarray, fs: float) -> np.ndarray:
    sos = butter(FILTER_ORDER, [LOWCUT, HIGHCUT], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=1)


def _select_channel(ch_names: Sequence[str], requested: str, fallback: Optional[int] = None) -> Tuple[int, str]:
    """Match channel names while ignoring common EEG prefixes and casing."""
    target = requested.casefold().removeprefix("eeg-")
    normalized = [name.casefold().removeprefix("eeg-") for name in ch_names]
    if target in normalized:
        index = normalized.index(target)
    elif fallback is not None:
        index = fallback
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
    """Raw (top) vs. filtered (bottom) EEG around one cue, epoch window shaded."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    axes[0].plot(time_sec, raw_segment, color="#1f77b4", linewidth=1.1, label="Raw EEG")
    axes[1].plot(
        time_sec,
        filtered_segment,
        color="#2ca02c",
        linewidth=1.1,
        label=f"Filtered ({LOWCUT:g}\u2013{HIGHCUT:g} Hz)",
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
            label=f"Epoch window [{EPOCH_START:.1f}\u2013{epoch_end:.1f} s]",
        )
        axis.set_ylabel("Amplitude (\u00b5V)", fontsize=11)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", framealpha=0.9)

    axes[0].set_title(
        f"Subject {subject} \u2014 Raw EEG (Electrode {channel_name}, "
        f"Trial {trial_number}: {task_label})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_title(
        f"Subject {subject} \u2014 Preprocessed EEG "
        f"(6th-order zero-phase Butterworth {LOWCUT:g}\u2013{HIGHCUT:g} Hz)",
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


def _inspection_from_continuous(
    data: np.ndarray,
    fs: float,
    cue_samples: np.ndarray,
    task_labels: Sequence[str],
    ch_names: Sequence[str],
    out_file: Path,
    subject: str,
    channel_name: str,
    trial_index: int,
    epoch_end: float,
    pre_cue_sec: float,
    post_cue_sec: float,
    fallback_channel: Optional[int] = None,
) -> Path:
    """Shared body of the per-dataset inspection plots (data: channels x samples, uV)."""
    if trial_index < 0:
        raise ValueError("Trial index must be zero or greater.")
    if len(cue_samples) == 0:
        raise ValueError(f"No cue markers found for {subject}.")
    trial_index = min(trial_index, len(cue_samples) - 1)
    cue_sample = int(cue_samples[trial_index])
    channel_index, channel_name = _select_channel(ch_names, channel_name, fallback_channel)
    filtered = _filter_signal(data[channel_index:channel_index + 1], fs)[0]  # only the plotted channel
    start = max(0, int(cue_sample - pre_cue_sec * fs))
    end = min(data.shape[1], int(cue_sample + post_cue_sec * fs))
    time_sec = (np.arange(start, end) - cue_sample) / fs
    return _save_inspection_plot(
        data[channel_index, start:end], filtered[start:end], time_sec, out_file,
        subject, channel_name, trial_index + 1, task_labels[trial_index], epoch_end, fs,
    )


# ----------------------------------------------------------------------------------
# A. Raw-vs-filtered inspection plots, one function per dataset
# ----------------------------------------------------------------------------------
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
    task_labels = [classes[0] if lab == -1 else classes[1] for lab in labels]
    subject = mat_path.stem.replace("BCICIV_calib_ds", "")
    return _inspection_from_continuous(
        cnt, FS_DS1, positions, task_labels, ch_names,
        output_dir / f"subject_{subject}_raw_vs_preprocessed.png",
        subject, channel_name, trial_index, EPOCH_END_DS1, pre_cue_sec, post_cue_sec,
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

    import mne  # imported lazily (after the np.fromstring shim above)

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
    task_labels = [cue_code_to_label[int(code)] for code in cue_events[:, 2]]
    return _inspection_from_continuous(
        data, fs, cue_events[:, 0], task_labels, ch_names,
        output_dir / f"subject_{subject}_raw_vs_preprocessed.png",
        subject, channel_name, trial_index, EPOCH_END_DS2, pre_cue_sec, post_cue_sec,
    )


def plot_dataset3_inspection(
    session: dict,
    stem: str,
    output_dir: Path,
    channel_name: str = "C3",
    trial_index: int = 0,
    pre_cue_sec: float = 1.0,
    post_cue_sec: float = 4.0,
) -> Path:
    """Raw vs. filtered around one Dataset 3 (BCI IV 2b) cue. `session` comes from
    preprocess_dataset3.load_gdf_session."""
    task_labels = [DS3_CLASS_LABELS[int(lab)] for lab in session["labels"]]
    return _inspection_from_continuous(
        session["eeg"], session["fs"], session["cues"], task_labels, session["channels"],
        output_dir / stem / f"raw_vs_preprocessed_{stem}.png",
        stem, channel_name, trial_index, EPOCH_END_DS3, pre_cue_sec, post_cue_sec,
    )


def plot_dataset4_inspection(
    run: dict,
    subject: str,
    ch_names: Sequence[str],
    output_dir: Path,
    channel_name: str = "C3",
    trial_index: int = 0,
    pre_cue_sec: float = 1.0,
    post_cue_sec: float = 4.0,
    fallback_channel: int = DS4_DEFAULT_CHANNEL,
) -> Path:
    """Raw vs. filtered around one Dataset 4 onset marker of run 1 (original 512 Hz).
    `run` is one element of preprocess_dataset4.load_mat_session."""
    task_labels = [DS4_CLASS_LABELS[int(lab)] for lab in run["y"]]
    return _inspection_from_continuous(
        run["X"], run["fs"], run["trial"], task_labels, ch_names,
        output_dir / subject / f"raw_vs_preprocessed_{subject}_run1.png",
        f"{subject} (run 1)", channel_name, trial_index, EPOCH_END_DS4, pre_cue_sec, post_cue_sec,
        fallback_channel=fallback_channel,
    )


def _finish(fig, save_path: Optional[str], bbox_inches: Optional[str] = None) -> None:
    """Save the figure (creating folders) or show it interactively."""
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches=bbox_inches)
        plt.close(fig)
    else:
        plt.show()


def epoch_time_axis(n_samples: int, fs: float, start: float = 0.5) -> np.ndarray:
    """Time axis (s, relative to the cue / MI onset) of an epoch that starts at `start`."""
    return start + np.arange(n_samples) / fs


# ----------------------------------------------------------------------------------
# 1. Filter response
# ----------------------------------------------------------------------------------
def plot_filter_frequency_response(
    lowcut: float = 8.0,
    highcut: float = 30.0,
    fs: float = 250.0,
    order: int = 6,
    save_path: Optional[str] = None,
):
    """Magnitude/phase response of the 6th-order Butterworth band-pass.

    The pipeline uses sosfiltfilt (forward-backward), so the *effective* magnitude
    is |H|^2 (-6 dB at the cut-offs) and the phase is exactly zero.
    """
    sos = signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    w, h = signal.sosfreqz(sos, worN=4000, fs=fs)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-8))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    ax1.plot(w, mag_db, "b", lw=1.8, label="Single pass")
    ax1.plot(w, 2 * mag_db, "m--", lw=1.4, label="Effective (filtfilt, zero-phase)")
    ax1.set_title(f"{order}th-Order Butterworth Band-pass ({lowcut:g}-{highcut:g} Hz, fs={fs:g} Hz)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Gain (dB)")
    ax1.axvline(lowcut, color="r", ls="--", alpha=0.7, label=f"Low cut-off ({lowcut:g} Hz)")
    ax1.axvline(highcut, color="g", ls="--", alpha=0.7, label=f"High cut-off ({highcut:g} Hz)")
    ax1.axhline(-3, color="k", ls=":", alpha=0.6, label="-3 dB")
    ax1.set_ylim(-80, 5)
    ax1.set_xlim(0, 60)
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8)

    ax2.plot(w, np.degrees(np.unwrap(np.angle(h))), "g", lw=1.5, label="Single pass")
    ax2.axhline(0, color="m", ls="--", lw=1.4, label="Effective (filtfilt): 0 deg")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    _finish(fig, save_path)


# ----------------------------------------------------------------------------------
# 2. Segmentation overview
# ----------------------------------------------------------------------------------
def plot_segmentation_overview(
    signal_1ch: np.ndarray,
    fs: float,
    cue_samples: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    channel_name: str = "C3",
    window: Sequence[float] = (0.5, 2.5),
    t_start: Optional[float] = None,
    duration: float = 60.0,
    artifact: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Continuous single-channel EEG (uV) with cue markers and extracted windows.

    labels: values 1/2 (class_names[0] / class_names[1]).
    """
    cue_samples = np.asarray(cue_samples)
    labels = np.asarray(labels)
    if t_start is None:
        t_start = max(0.0, cue_samples[0] / fs - 5.0)
    t_end = t_start + duration
    i0, i1 = int(t_start * fs), int(min(t_end * fs, len(signal_1ch)))
    t = np.arange(i0, i1) / fs

    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.plot(t, signal_1ch[i0:i1], color="0.35", lw=0.6)

    used_labels = set()
    for k, (cue, lab) in enumerate(zip(cue_samples, labels)):
        tc = cue / fs
        if not (t_start <= tc <= t_end):
            continue
        color = CLASS_COLORS[(int(lab) - 1) % 2]
        name = class_names[(int(lab) - 1) % 2]
        ax.axvline(tc, color=color, lw=1.2, label=None if name in used_labels else f"Cue: {name}")
        used_labels.add(name)
        is_art = artifact is not None and bool(artifact[k])
        ax.axvspan(tc + window[0], tc + window[1], color=color, alpha=0.18,
                   hatch="//" if is_art else None, edgecolor="k" if is_art else None, lw=0)
    ax.plot([], [], color="0.5", alpha=0.3, lw=6, label=f"Extracted window [{window[0]}-{window[1]} s]")
    if artifact is not None and np.any(artifact):
        ax.axvspan(0, 0, hatch="//", facecolor="none", edgecolor="k", label="Artifact-flagged trial (kept)")

    ax.set_xlim(t_start, t_end)
    ax.set_xlabel("Time in recording (s)")
    ax.set_ylabel(f"{channel_name} (uV)")
    ax.set_title(title or "Trial segmentation overview", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    _finish(fig, save_path)


# ----------------------------------------------------------------------------------
# 3. Single-trial raw vs filtered
# ----------------------------------------------------------------------------------
def plot_trial_comparison(
    raw_trial: np.ndarray,
    filt_trial: np.ndarray,
    time_in_sec: np.ndarray,
    channel_names: List[str],
    subject_id: str = "B01",
    trial_idx: int = 0,
    class_label: str = "Left Hand",
    window: Sequence[float] = (0.5, 2.5),
    save_path: Optional[str] = None,
):
    """Raw vs. filtered single trial. Arrays: (n_channels, n_samples), in uV."""
    n_ch = len(channel_names)
    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 2.4 * n_ch + 1), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for i, ch in enumerate(channel_names):
        ax = axes[i]
        ax.plot(time_in_sec, raw_trial[i], color="#1f77b4", lw=1.0, alpha=0.7, label="Raw (uV)")
        ax.plot(time_in_sec, filt_trial[i], color="#d62728", lw=1.4, label="Preprocessed 8-30 Hz (uV)")
        ax.set_ylabel(f"{ch} (uV)", fontweight="bold")
        ax.grid(True, ls="--", alpha=0.4)
        if i == 0:
            ax.set_title(f"Subject {subject_id} - Trial #{trial_idx + 1} ({class_label}) "
                         f"Single-Trial Preprocessing [{window[0]} - {window[1]} s]",
                         fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time relative to motor imagery cue (s)")
    fig.tight_layout()
    _finish(fig, save_path)


# ----------------------------------------------------------------------------------
# 4. PSD
# ----------------------------------------------------------------------------------
def plot_psd(
    raw_trials: np.ndarray,
    filt_trials: np.ndarray,
    fs: float,
    channel_names: List[str],
    subject_id: str = "B01",
    save_path: Optional[str] = None,
):
    """Trial-averaged Welch PSD (uV^2/Hz) before/after filtering. Arrays: (trials, ch, samples)."""
    n_ch = len(channel_names)
    fig, axes = plt.subplots(1, n_ch, figsize=(4.7 * n_ch, 4), sharey=True, squeeze=False)
    axes = axes[0]
    nperseg = min(int(fs), raw_trials.shape[-1])  # 1 Hz resolution
    for i, ch in enumerate(channel_names):
        ax = axes[i]
        f, p_raw = signal.welch(raw_trials[:, i, :], fs=fs, nperseg=nperseg, axis=-1)
        _, p_filt = signal.welch(filt_trials[:, i, :], fs=fs, nperseg=nperseg, axis=-1)
        ax.semilogy(f, p_raw.mean(0), color="#1f77b4", lw=1.6, label="Raw EEG")
        ax.semilogy(f, np.maximum(p_filt.mean(0), 1e-8), color="#d62728", lw=1.8, label="Preprocessed (8-30 Hz)")
        ax.axvspan(8, 30, color="gray", alpha=0.15, label="Passband (8-30 Hz)")
        ax.set_title(f"Channel: {ch}", fontweight="bold")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, min(60, fs / 2))
        ax.grid(True, which="both", ls="--", alpha=0.4)
        if i == 0:
            ax.set_ylabel("PSD (uV$^2$/Hz)")
            ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Subject {subject_id} - Power Spectral Density, before vs. after filtering",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _finish(fig, save_path, bbox_inches="tight")


# ----------------------------------------------------------------------------------
# 5. Class-wise band power
# ----------------------------------------------------------------------------------
def plot_band_power(
    filt_trials: np.ndarray,
    labels: np.ndarray,
    time_in_sec: np.ndarray,
    fs: float,
    channel_names: List[str],
    class_values: Sequence[int] = (1, 2),
    class_names: Sequence[str] = ("Left Hand", "Right Hand"),
    subject_id: str = "B01",
    save_path: Optional[str] = None,
):
    """Trial-averaged 8-30 Hz band power (x^2, 100 ms moving average) +/- s.e.m. per class.

    The epochs only cover 0.5-2.5 s, so no pre-cue baseline is available: this
    shows absolute band power, not ERD% - class differences matter, not the offset.
    """
    n_ch = len(channel_names)
    win = max(1, int(round(0.1 * fs)))
    fig, axes = plt.subplots(1, n_ch, figsize=(4.7 * n_ch, 4), sharey=True, squeeze=False)
    axes = axes[0]
    for i, ch in enumerate(channel_names):
        ax = axes[i]
        for k, (val, name) in enumerate(zip(class_values, class_names)):
            power = filt_trials[labels == val, i, :] ** 2
            power = uniform_filter1d(power, size=win, axis=-1, mode="nearest")
            mean = power.mean(0)
            sem = power.std(0, ddof=1) / np.sqrt(len(power)) if len(power) > 1 else np.zeros_like(mean)
            ax.plot(time_in_sec, mean, color=CLASS_COLORS[k], lw=2.0, label=f"{name} (n={len(power)})")
            ax.fill_between(time_in_sec, mean - sem, mean + sem, color=CLASS_COLORS[k], alpha=0.2)
        ax.set_title(f"Channel: {ch}", fontweight="bold")
        ax.set_xlabel("Time relative to cue (s)")
        ax.grid(True, ls="--", alpha=0.4)
        if i == 0:
            ax.set_ylabel("Band power 8-30 Hz (uV$^2$)")
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Subject {subject_id} - Sensorimotor rhythm power (8-30 Hz) during motor imagery",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _finish(fig, save_path, bbox_inches="tight")


# Backwards-compatible names used by the old Dataset-3-only visualize.py
plot_dataset3_trial_comparison = plot_trial_comparison
plot_dataset3_psd = plot_psd
plot_dataset3_erd_ers = plot_band_power


# ----------------------------------------------------------------------------------
# Command line driver
# ----------------------------------------------------------------------------------
def _import_preprocessor(name: str):
    """Import src/preprocess_datasetN whether run as a script or as a package module."""
    import importlib
    try:
        return importlib.import_module(f"src.{name}")
    except ImportError:
        return importlib.import_module(name)


def epoch_figures(dataset: int, npz_path: Path, out_dir: Path, args, seg_source: Optional[dict] = None) -> None:
    """Family B: all epoch-level figures for one preprocessed .npz (Datasets 3 and 4).

    seg_source: loaded raw session (dataset 3) or first run (dataset 4) used for the
    segmentation overview; None skips that plot.
    """
    d = np.load(npz_path, allow_pickle=False)
    stem = npz_path.name.replace("_preprocessed.npz", "")
    fs = float(d["fs"])
    w0, w1 = [float(v) for v in d["time_window"]]
    all_ch = [str(c) for c in d["channels"]]
    class_names = [str(c).title() for c in d["classes"]]

    X_f, X_r, y = d[f"X_{args.set}"], d[f"X_{args.set}_raw"], d[f"y_{args.set}_raw"]
    if len(X_f) == 0:
        print(f"  [skip] {stem}: no {args.set} trials")
        return
    ch_idx = args.channels if args.channels else (list(range(3)) if dataset == 3 else [5, 7, 9])
    ch_names = [all_ch[i] for i in ch_idx]
    t_axis = epoch_time_axis(X_f.shape[-1], fs, w0)
    tag = f"{stem}_{args.set}"
    save = lambda name: str(out_dir / name)
    print(f"  {stem}: {X_f.shape[0]} {args.set} trials, fs={fs:g} Hz, channels {ch_names}")

    if seg_source is not None:
        if dataset == 3:
            plot_segmentation_overview(
                seg_source["eeg"][ch_idx[0]], seg_source["fs"], seg_source["cues"], seg_source["labels"],
                class_names, all_ch[ch_idx[0]], (w0, w1), artifact=seg_source["artifact"],
                title=f"{stem} - segmentation: cue markers and [{w0}-{w1} s] windows",
                save_path=save(f"segmentation_{stem}.png"))
        else:
            plot_segmentation_overview(
                seg_source["X"][ch_idx[0]], seg_source["fs"], seg_source["trial"], seg_source["y"],
                class_names, all_ch[ch_idx[0]], (w0, w1), t_start=0.0, duration=80.0,
                title=f"{stem} (run 1, {seg_source['fs']:g} Hz) - segmentation: onset markers and [{w0}-{w1} s] windows",
                save_path=save(f"segmentation_{stem}_run1.png"))

    for k in (1, 2):
        idx = np.flatnonzero(y == k)
        if len(idx) == 0:
            continue
        i = int(idx[min(args.example_idx, len(idx) - 1)])
        plot_trial_comparison(X_r[i][ch_idx], X_f[i][ch_idx], t_axis, ch_names, stem, i,
                              class_names[k - 1], (w0, w1),
                              save(f"trial_{tag}_{class_names[k - 1].replace(' ', '_')}.png"))

    plot_psd(X_r[:, ch_idx, :], X_f[:, ch_idx, :], fs, ch_names, stem, save(f"psd_{tag}.png"))
    plot_band_power(X_f[:, ch_idx, :], y, t_axis, fs, ch_names, (1, 2), class_names, stem,
                    save(f"band_power_{tag}.png"))


def _filter_response_once(fs: float, out_dir: Path) -> None:
    plot_filter_frequency_response(LOWCUT, HIGHCUT, fs, FILTER_ORDER, str(out_dir / f"filter_response_{int(fs)}Hz.png"))


# ----------------------------------------------------------------------------------
# Dataset drivers: each returns a list of (name, error) for files that failed
# ----------------------------------------------------------------------------------
def run_dataset1(args) -> List[Tuple[str, str]]:
    pattern = f"BCICIV_calib_ds{args.subject}.mat" if args.subject else "BCICIV_calib_ds1*.mat"
    files = sorted(args.data_dir_ds1.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Dataset 1 calibration files ('{pattern}') in {args.data_dir_ds1.resolve()}.")
    out_dir = args.output_dir / "BCICIV_1"
    print(f"\n== Dataset 1: {len(files)} subject(s) ==")
    _filter_response_once(FS_DS1, out_dir)
    failed = []
    for mat_path in files:
        try:
            plot_dataset1_subject(mat_path, out_dir, args.channel, args.trial, args.pre_cue_sec, args.post_cue_sec)
        except Exception as exc:
            failed.append((mat_path.name, repr(exc)))
            print(f"  [FAILED] {mat_path.name}: {exc!r}")
    return failed


def run_dataset2a(args) -> List[Tuple[str, str]]:
    subjects = [args.subject] if args.subject else [f"A0{i}" for i in range(1, 10)]
    present = [sub for sub in subjects if (args.data_dir_ds2 / f"{sub}T.gdf").exists()]
    if not present:
        raise FileNotFoundError(f"No Dataset 2a training GDF files found in {args.data_dir_ds2.resolve()}.")
    out_dir = args.output_dir / "BCICIV_2a"
    print(f"\n== Dataset 2a: {len(present)} subject(s) ==")
    _filter_response_once(FS_DS2, out_dir)
    failed = []
    for subject in present:
        try:
            plot_dataset2_subject(args.data_dir_ds2, out_dir, subject, args.channel, args.trial,
                                  args.pre_cue_sec, args.post_cue_sec)
        except Exception as exc:
            failed.append((subject, repr(exc)))
            print(f"  [FAILED] {subject}: {exc!r}")
    return failed


def run_dataset3(args) -> List[Tuple[str, str]]:
    pp = _import_preprocessor("preprocess_dataset3")
    session_glob = f"{args.session:02d}" if args.session else "0?"
    pattern = f"{args.subject or 'B0?'}{session_glob}[TE].gdf"
    files = sorted(args.data_dir_ds3.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Dataset 3 files ('{pattern}') in {args.data_dir_ds3.resolve()}.")
    out_dir = args.output_dir / "dataset_3"
    pre_dir = args.preprocessed_dir / "dataset_3"
    print(f"\n== Dataset 3: {len(files)} file(s) ==")
    _filter_response_once(FS_DS3, out_dir)
    failed = []
    for gdf in files:
        stem = gdf.stem
        try:
            session = pp.load_gdf_session(gdf)
        except pp.MissingLabelsError as exc:
            print(f"  [skip] {stem}: {exc}")
            continue
        except Exception as exc:
            failed.append((stem, repr(exc)))
            print(f"  [FAILED] {stem}: {exc!r}")
            continue
        try:
            plot_dataset3_inspection(session, stem, out_dir, args.channel, args.trial, args.pre_cue_sec, args.post_cue_sec)
            if args.no_epoch_plots:
                continue
            npz = pre_dir / f"{stem}_preprocessed.npz"
            if npz.exists():
                epoch_figures(3, npz, out_dir / stem, args, None if args.no_segmentation else session)
            else:
                print(f"  [skip] epoch figures for {stem}: {npz.name} not found (run preprocess_dataset3.py)")
        except Exception as exc:
            failed.append((stem, repr(exc)))
            print(f"  [FAILED] {stem}: {exc!r}")
            traceback.print_exc(limit=1)
    return failed


def run_dataset4(args) -> List[Tuple[str, str]]:
    pp = _import_preprocessor("preprocess_dataset4")
    files = sorted(args.data_dir_ds4.glob(f"{args.subject or 'S??'}T.mat"))
    if not files:
        raise FileNotFoundError(f"No Dataset 4 training files in {args.data_dir_ds4.resolve()}.")
    out_dir = args.output_dir / "dataset_4"
    pre_dir = args.preprocessed_dir / "dataset_4"
    print(f"\n== Dataset 4: {len(files)} subject(s) ==")
    _filter_response_once(FS_DS4, out_dir)
    failed = []
    for mat_path in files:
        sid = mat_path.stem[:-1]
        try:
            run = pp.load_mat_session(mat_path)[0]
            npz = pre_dir / f"{sid}_preprocessed.npz"
            ch_names = ([str(c) for c in np.load(npz)["channels"]] if npz.exists()
                        else [f"Ch{i:02d}" for i in range(1, run["X"].shape[0] + 1)])
            fallback = args.channels[0] if args.channels else DS4_DEFAULT_CHANNEL
            plot_dataset4_inspection(run, sid, ch_names, out_dir, args.channel, args.trial,
                                     args.pre_cue_sec, args.post_cue_sec, fallback)
            if args.no_epoch_plots:
                continue
            if npz.exists():
                epoch_figures(4, npz, out_dir / sid, args, None if args.no_segmentation else run)
            else:
                print(f"  [skip] epoch figures for {sid}: {npz.name} not found (run preprocess_dataset4.py)")
        except Exception as exc:
            failed.append((sid, repr(exc)))
            print(f"  [FAILED] {sid}: {exc!r}")
            traceback.print_exc(limit=1)
    return failed


RUNNERS = {"1": run_dataset1, "2a": run_dataset2a, "3": run_dataset3, "4": run_dataset4}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raw-vs-filtered inspection plots (Datasets 1, 2a, 3, 4) plus epoch-level figures "
                    "(Datasets 3, 4)."
    )
    parser.add_argument("--dataset", choices=["1", "2a", "3", "4", "all"], default="all",
                        help="Dataset to plot: 1, 2a, 3, 4 or all (default: all).")
    parser.add_argument("--subject", default=None,
                        help="One subject only: 1a (ds1), A01 (ds2a), B01 (ds3), S01 (ds4). Default: every subject.")
    parser.add_argument("--session", type=int, default=None,
                        help="Dataset 3 only: restrict to one session (3 = B0103T..B0903T, the paper's).")
    parser.add_argument("--data-dir-ds1", type=Path, default=DEFAULT_DATA_DIR_DS1)
    parser.add_argument("--data-dir-ds2", type=Path, default=DEFAULT_DATA_DIR_DS2)
    parser.add_argument("--data-dir-ds3", type=Path, default=DEFAULT_DATA_DIR_DS3)
    parser.add_argument("--data-dir-ds4", type=Path, default=DEFAULT_DATA_DIR_DS4)
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR,
                        help="Base folder holding dataset_3/ and dataset_4/ .npz files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR,
                        help="Base directory for plots (dataset-specific folders are created).")
    parser.add_argument("--channel", default="C3",
                        help="Electrode for the inspection plots (default: C3; dataset 4 has no channel "
                             "names, so it falls back to --channels[0] or index 5).")
    parser.add_argument("--trial", type=int, default=0, help="Zero-based cue index for inspection plots (default: 0).")
    parser.add_argument("--pre-cue-sec", type=float, default=1.0)
    parser.add_argument("--post-cue-sec", type=float, default=4.0)
    parser.add_argument("--set", choices=["train", "test"], default="train",
                        help="Datasets 3/4: which preprocessed epochs the epoch figures use.")
    parser.add_argument("--example-idx", type=int, default=0,
                        help="Datasets 3/4: index of the example trial of each class in the single-trial figure.")
    parser.add_argument("--channels", type=int, nargs="*", default=None,
                        help="Datasets 3/4: 0-based channel indices for the epoch figures "
                             "(default: all 3 for dataset 3; 5 7 9 for dataset 4).")
    parser.add_argument("--no-segmentation", action="store_true",
                        help="Datasets 3/4: skip the segmentation overview (faster).")
    parser.add_argument("--no-epoch-plots", action="store_true",
                        help="Datasets 3/4: only the raw-vs-filtered inspection plot.")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    selected = list(RUNNERS) if args.dataset == "all" else [args.dataset]
    all_failed = []
    for ds in selected:
        try:
            all_failed += [(f"ds{ds}:{name}", err) for name, err in RUNNERS[ds](args)]
        except FileNotFoundError as exc:
            if args.dataset != "all":
                raise
            print(f"\n[skip] dataset {ds}: {exc}")

    print(f"\nAll requested plots saved under: {args.output_dir.resolve()}")
    for name, err in all_failed:
        print(f"Failed: {name}: {err}")


if __name__ == "__main__":
    main()