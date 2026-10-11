"""
Visualization module for inspection of raw vs. preprocessed EEG signals in Dataset 3.
Plots:
1. Butterworth filter frequency response at fs=250 Hz.
2. Time-domain raw vs. filtered trial comparison for C3, Cz, and C4 channels.
3. Power Spectral Density (PSD) before and after 8-30 Hz filtering.
4. Left Hand vs. Right Hand class comparisons.
"""

import os
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def plot_filter_frequency_response(
    lowcut: float = 8.0,
    highcut: float = 30.0,
    fs: float = 250.0,
    order: int = 6,
    save_path: Optional[str] = None
):
    """
    Plot magnitude and phase response of the 6th-order Butterworth bandpass filter at fs=250 Hz.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')

    w, h = signal.sosfreqz(sos, worN=2000, fs=fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-5)), 'b', lw=1.8)
    ax1.set_title(f'6th-Order Butterworth Bandpass Filter ({lowcut} - {highcut} Hz, fs={fs} Hz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Gain (dB)', fontsize=10)
    ax1.axvline(lowcut, color='r', linestyle='--', alpha=0.7, label=f'Low Cutoff ({lowcut} Hz)')
    ax1.axvline(highcut, color='g', linestyle='--', alpha=0.7, label=f'High Cutoff ({highcut} Hz)')
    ax1.axhline(-3, color='k', linestyle=':', alpha=0.6, label='-3 dB')
    ax1.set_ylim(-70, 5)
    ax1.set_xlim(0, 50)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')

    angles = np.unwrap(np.angle(h))
    ax2.plot(w, np.degrees(angles), 'g', lw=1.5)
    ax2.set_xlabel('Frequency (Hz)', fontsize=10)
    ax2.set_ylabel('Phase (degrees)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_dataset3_trial_comparison(
    raw_trial: np.ndarray,
    filt_trial: np.ndarray,
    time_in_sec: np.ndarray,
    channel_names: List[str] = ['C3', 'Cz', 'C4'],
    subject_id: str = 'B01',
    trial_idx: int = 0,
    class_label: str = 'Left Hand',
    save_path: Optional[str] = None
):
    """
    Plot 3-channel (C3, Cz, C4) raw vs filtered trial side-by-side in time domain.

    Parameters
    ----------
    raw_trial : np.ndarray of shape (3, n_samples)
    filt_trial : np.ndarray of shape (3, n_samples)
    time_in_sec : np.ndarray of shape (n_samples,)
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    for i, ch in enumerate(channel_names[:3]):
        ax = axes[i]
        ax.plot(time_in_sec, raw_trial[i] * 1e6, color='#1f77b4', lw=1.0, alpha=0.7, label='Raw (μV)')
        ax.plot(time_in_sec, filt_trial[i] * 1e6, color='#d62728', lw=1.4, label='Preprocessed 8-30 Hz (μV)')
        ax.set_ylabel(f'{ch} (μV)', fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.set_title(f'Subject {subject_id} - Trial #{trial_idx+1} ({class_label}) Single-Trial Preprocessing [0.5 - 2.5 s]', fontsize=11, fontweight='bold')
            ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time relative to motor imagery cue (seconds)', fontsize=10)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_dataset3_psd(
    raw_trials: np.ndarray,
    filt_trials: np.ndarray,
    fs: float = 250.0,
    channel_names: List[str] = ['C3', 'Cz', 'C4'],
    subject_id: str = 'B01',
    save_path: Optional[str] = None
):
    """
    Plot Welch Power Spectral Density (PSD) comparing raw vs 8-30 Hz filtered signals across C3, Cz, and C4.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    n_per_seg = min(250, raw_trials.shape[-1])
    for ch_idx, ch in enumerate(channel_names[:3]):
        ax = axes[ch_idx]
        freqs, psd_raw = signal.welch(raw_trials[:, ch_idx, :], fs=fs, nperseg=n_per_seg, axis=-1)
        _, psd_filt = signal.welch(filt_trials[:, ch_idx, :], fs=fs, nperseg=n_per_seg, axis=-1)

        mean_raw = np.mean(psd_raw, axis=0) * 1e12  # μV²/Hz
        mean_filt = np.mean(psd_filt, axis=0) * 1e12

        ax.semilogy(freqs, mean_raw, label='Raw EEG', color='#1f77b4', lw=1.6)
        ax.semilogy(freqs, mean_filt, label='Preprocessed (8-30 Hz)', color='#d62728', lw=1.8)
        ax.axvspan(8, 30, color='gray', alpha=0.15, label='Passband (8-30 Hz)')

        ax.set_title(f'Channel: {ch}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_xlim(0, 60)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        if ch_idx == 0:
            ax.set_ylabel('PSD (μV²/Hz)', fontsize=10)
            ax.legend(loc='lower left', fontsize=9)

    fig.suptitle(f'Subject {subject_id} - Power Spectral Density (PSD) Comparison (C3, Cz, C4)', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_dataset3_erd_ers(
    filt_trials: np.ndarray,
    labels: np.ndarray,
    time_in_sec: np.ndarray,
    channel_names: List[str] = ['C3', 'Cz', 'C4'],
    subject_id: str = 'B01',
    save_path: Optional[str] = None
):
    """
    Plot trial-averaged band power dynamics (Mu/Beta ERD/ERS) comparing Left Hand vs. Right Hand across C3, Cz, C4.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    left_mask = (labels == 1)
    right_mask = (labels == 2)

    for ch_idx, ch in enumerate(channel_names[:3]):
        ax = axes[ch_idx]
        # Instantaneous power: x(t)^2 smoothed with a moving average window
        win_size = 25  # 100 ms smoothing
        kernel = np.ones(win_size) / win_size

        power_left = filt_trials[left_mask, ch_idx, :] ** 2 * 1e12
        power_right = filt_trials[right_mask, ch_idx, :] ** 2 * 1e12

        mean_p_left = np.mean(power_left, axis=0)
        mean_p_right = np.mean(power_right, axis=0)

        smooth_left = np.convolve(mean_p_left, kernel, mode='same')
        smooth_right = np.convolve(mean_p_right, kernel, mode='same')

        ax.plot(time_in_sec, smooth_left, label='Left Hand MI', color='#1f77b4', lw=2.0)
        ax.plot(time_in_sec, smooth_right, label='Right Hand MI', color='#ff7f0e', lw=2.0)
        ax.set_title(f'Channel: {ch}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time relative to cue (s)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        if ch_idx == 0:
            ax.set_ylabel('Band Power (μV²)', fontsize=10)
            ax.legend(loc='upper right')

    fig.suptitle(f'Subject {subject_id} - Sensorimotor Rhythm Power (8-30 Hz) during Motor Imagery', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
