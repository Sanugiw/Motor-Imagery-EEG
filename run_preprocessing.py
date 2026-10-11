"""
Master Preprocessing & Segmentation Script for Dataset 3 (BCI Competition IV Dataset IIb).
Paper Reference: 'The CSP-Based New Features Plus Non-Convex Log Sparse Feature Selection
for Motor Imagery EEG Classification', Shaorong Zhang et al., Sensors 2020.

Dataset 3 Specifications from Paper:
- 9 Subjects: B01 to B09 (Session 03T: B0103T.gdf ... B0903T.gdf).
- 3 EEG Channels: C3, Cz, C4.
- Sampling Rate: 250 Hz.
- Tasks: Left Hand (Class 1) vs. Right Hand (Class 2).
- Trials: 160 trials per subject (80 Left, 80 Right).
  -> 80 trials for Training Set (40 Left, 40 Right)
  -> 80 trials for Test Set (40 Left, 40 Right)
- Segmentation Window: 0.5 s to 2.5 s relative to motor imagery cue onset (500 samples).
- Preprocessing: 6th-order Butterworth 8-30 Hz bandpass filter + 10-subband filter bank for CSP-FB.
"""

import os
import numpy as np

from src.dataset import load_bci_comp4_dataset_2b
from src.preprocessing import (
    preprocess_and_segment_pipeline,
    get_filter_bank_subbands
)
from src.visualize import (
    plot_filter_frequency_response,
    plot_dataset3_trial_comparison,
    plot_dataset3_psd,
    plot_dataset3_erd_ers
)


def run_dataset3_preprocessing():
    data_dir = os.path.join('data', 'dataset_3')
    output_dir = os.path.join('preprocessed_data', 'dataset_3')
    fig_dir = os.path.join('results', 'figures', 'dataset_3')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 80)
    print(" BCI COMPETITION IV DATASET IIb (DATASET 3) PREPROCESSING PIPELINE")
    print(" Paper: Zhang et al., Sensors 2020")
    print("=" * 80)

    # 1. Plot Filter Frequency Response once
    print("\n[Step 1] Plotting 6th-order Butterworth Filter Frequency Response (8-30 Hz at 250 Hz)...")
    plot_filter_frequency_response(
        lowcut=8.0,
        highcut=30.0,
        fs=250.0,
        order=6,
        save_path=os.path.join(fig_dir, 'butterworth_6th_order_250Hz_response.png')
    )

    subjects = [f'B0{i}' for i in range(1, 10)]
    subbands = get_filter_bank_subbands(start_freq=8.0, end_freq=30.0, bandwidth=4.0, step=2.0)
    print(f"Filter Bank Sub-bands (10 bands): {subbands}")

    all_stats = []

    for subj in subjects:
        gdf_fname = f"{subj}03T.gdf"
        gdf_path = os.path.join(data_dir, gdf_fname)

        if not os.path.exists(gdf_path):
            print(f"[Warning] File not found: {gdf_path}. Skipping.")
            continue

        print(f"\n{'-' * 80}")
        print(f" Processing Subject: {subj} ({gdf_fname})")
        print(f"{'-' * 80}")

        # 1. Load continuous data and events
        loaded = load_bci_comp4_dataset_2b(
            gdf_path,
            t_min=0.5,
            t_max=2.5,
            t_epoch_min=-1.0,
            t_epoch_max=4.0
        )

        fs = loaded['sample_rate']
        ch_names = loaded['channel_names']
        cue_indices = loaded['cue_sample_indices']
        labels_all = loaded['labels_all']
        raw_continuous = loaded['raw_continuous']

        print(f"  [Info] Sampling Rate: {fs} Hz | Channels: {ch_names}")
        print(f"  [Info] Total Cues: {len(cue_indices)} (Left: {np.sum(labels_all == 1)}, Right: {np.sum(labels_all == 2)})")

        # 2. Continuous Preprocessing & Single-Trial Segmentation: [0.5 s, 2.5 s]
        trials_raw, trials_filt, trials_fb, time_vec = preprocess_and_segment_pipeline(
            continuous_raw=raw_continuous,
            cue_indices=cue_indices,
            fs=fs,
            t_min=0.5,
            t_max=2.5,
            lowcut=8.0,
            highcut=30.0,
            filter_order=6
        )

        n_trials, n_channels, n_samples = trials_filt.shape
        print(f"  [Segmentation] Extracted {n_trials} trials of shape ({n_channels} channels, {n_samples} samples) [{time_vec[0]:.2f} - {time_vec[-1]:.2f} s]")

        # 3. Train/Test split: 80 train, 80 test (as specified in paper)
        n_train = 80
        X_train_raw = trials_raw[:n_train]
        X_train_filt = trials_filt[:n_train]
        X_train_fb = trials_fb[:, :n_train, :, :]
        y_train = labels_all[:n_train]

        X_test_raw = trials_raw[n_train:]
        X_test_filt = trials_filt[n_train:]
        X_test_fb = trials_fb[:, n_train:, :, :]
        y_test = labels_all[n_train:]

        print(f"  [Split] Train Set: {X_train_filt.shape} (Left: {np.sum(y_train == 1)}, Right: {np.sum(y_train == 2)})")
        print(f"  [Split] Test Set:  {X_test_filt.shape} (Left: {np.sum(y_test == 1)}, Right: {np.sum(y_test == 2)})")

        # 4. Compute power and filtering metrics
        raw_power_uV2 = np.var(trials_raw) * 1e12
        filt_power_uV2 = np.var(trials_filt) * 1e12
        inband_ratio = (filt_power_uV2 / raw_power_uV2) * 100.0

        all_stats.append({
            'subject': subj,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'raw_var': raw_power_uV2,
            'filt_var': filt_power_uV2,
            'inband_ratio': inband_ratio
        })

        print(f"  [Power] Raw Variance: {raw_power_uV2:.2f} uV^2 -> 8-30 Hz Filtered: {filt_power_uV2:.2f} uV^2 (In-band: {inband_ratio:.1f}%)")

        # 5. Save Preprocessed Data to .npz
        save_file = os.path.join(output_dir, f"{subj}_preprocessed.npz")
        np.savez_compressed(
            save_file,
            # Train Set
            X_train_raw=X_train_raw,
            X_train_filt=X_train_filt,
            X_train_fb=X_train_fb,
            y_train=y_train,
            # Test Set
            X_test_raw=X_test_raw,
            X_test_filt=X_test_filt,
            X_test_fb=X_test_fb,
            y_test=y_test,
            # Shared Metadata
            time=time_vec,
            channel_names=np.array(ch_names),
            class_names=np.array(['Left Hand', 'Right Hand']),
            subbands=np.array(subbands),
            sample_rate=fs,
            subject_id=subj
        )
        print(f"  [Saved] Saved to: {save_file} ({os.path.getsize(save_file) / (1024*1024):.2f} MB)")

        # 6. Generate Verification Plots
        # A) Single Trial Time-Domain Comparison
        plot_dataset3_trial_comparison(
            raw_trial=trials_raw[0],
            filt_trial=trials_filt[0],
            time_in_sec=time_vec,
            channel_names=ch_names,
            subject_id=subj,
            trial_idx=0,
            class_label='Left Hand' if labels_all[0] == 1 else 'Right Hand',
            save_path=os.path.join(fig_dir, f"{subj}_raw_vs_filtered_trial0.png")
        )

        # B) Power Spectral Density Comparison
        plot_dataset3_psd(
            raw_trials=trials_raw,
            filt_trials=trials_filt,
            fs=fs,
            channel_names=ch_names,
            subject_id=subj,
            save_path=os.path.join(fig_dir, f"{subj}_psd_comparison.png")
        )

        # C) ERD/ERS Sensorimotor Power Comparison
        plot_dataset3_erd_ers(
            filt_trials=trials_filt,
            labels=labels_all,
            time_in_sec=time_vec,
            channel_names=ch_names,
            subject_id=subj,
            save_path=os.path.join(fig_dir, f"{subj}_erd_ers_power.png")
        )

    # Summary table
    print("\n" + "=" * 80)
    print(" PREPROCESSING SUMMARY ACROSS ALL 9 SUBJECTS (DATASET 3)")
    print("=" * 80)
    print(f"{'Subject':<10} | {'Train Trials':<12} | {'Test Trials':<12} | {'Raw Var (uV^2)':<14} | {'Filt Var (uV^2)':<15} | {'In-Band %':<10}")
    print("-" * 80)
    for s in all_stats:
        print(f"{s['subject']:<10} | {s['n_train']:<12} | {s['n_test']:<12} | {s['raw_var']:<14.2f} | {s['filt_var']:<15.2f} | {s['inband_ratio']:<10.1f}%")
    print("=" * 80)


if __name__ == '__main__':
    run_dataset3_preprocessing()
