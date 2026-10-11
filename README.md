# Motor Imagery / Movement BCI Preprocessing Pipeline

Implementation of the segmentation and preprocessing methodology described in:
> **"The CSP-Based New Features Plus Non-Convex Log Sparse Feature Selection for Motor Imagery EEG Classification"**  
> Shaorong Zhang, Zhibin Zhu, Benxin Zhang, Bao Feng, Tianyou Yu, Zhi Li. *Sensors*, 2020, 20, 4749.

This first phase implements the dataset loading, single-trial segmentation, and preprocessing in Python for **BCI Competition IV Dataset 3** (`S1.mat` and `S2.mat`).

---

## 1. Dataset Overview (BCI Competition IV - Dataset 3)

The dataset contains directionally modulated activity recorded across 10 sensor channels located over motor areas at a sampling rate of **400 Hz**:
- **Subjects**: `S1` and `S2`.
- **4 Classes**: `1 - right`, `2 - forward`, `3 - left`, `4 - backward` (40 trials/class = 160 training trials per subject).
- **Test Data**: 74 trials for S1, 73 trials for S2.
- **Trial Recording**: 400 samples from $-0.4\text{ s}$ to $+0.59\text{ s}$ relative to movement onset ($t = 0\text{ s}$).

---

## 2. Methodology & Paper Mapping

### A. Trial Segmentation (Section 2.3 & 2.4)
Single trials are segmented relative to task onset ($t = 0\text{ s}$):
- **Full Trial Window**: $[-0.4\text{ s}, +0.59\text{ s}]$ (400 samples) capturing both motor preparation and execution.
- **Post-Onset Motor Task Window**: $[0.0\text{ s}, +0.59\text{ s}]$ (238 samples) focusing strictly on the movement phase, matching the paper's principle of single trial motor task extraction starting from $t = 0\text{ s}$.

### B. Preprocessing & Filtering (Section 2.3 & 2.4.2)
1. **Baseline Removal / Demeaning**: Removes DC bias per trial and channel.
2. **6th-Order Butterworth Bandpass Filter (8–30 Hz)**:
   - Target band: Mu (8–12 Hz) and Beta (18–26 Hz) sensorimotor rhythms.
   - Numerically stable zero-phase bidirectional filtering via Second-Order Sections (`sosfiltfilt`).
3. **Filter Bank Sub-Band Generation (for CSP-FB)**:
   - 10 sub-bands with 4 Hz bandwidth and 2 Hz overlap across 8–30 Hz:  
     $[8-12, 10-14, 12-16, 14-18, 16-20, 18-22, 20-24, 22-26, 24-28, 26-30]\text{ Hz}$.

---

## 3. Directory Layout

```text
Motor-Imagery-EEG/
├── data/
│   ├── S1.mat                        # Subject 1 dataset
│   ├── S2.mat                        # Subject 2 dataset
│   └── SensorPos.pdf                 # Sensor layout
├── preprocessed_data/
│   ├── S1_preprocessed.npz           # Preprocessed arrays for Subject 1
│   └── S2_preprocessed.npz           # Preprocessed arrays for Subject 2
├── results/figures/
│   ├── butterworth_6th_order_freq_response.png
│   ├── S1_class_averages.png
│   ├── S1_psd_comparison.png
│   ├── S1_raw_vs_filtered_trial0_ch0.png
│   ├── S2_class_averages.png
│   ├── S2_psd_comparison.png
│   └── S2_raw_vs_filtered_trial0_ch0.png
├── src/
│   ├── __init__.py
│   ├── dataset.py                    # Data loader and binary pair extractor
│   ├── segmentation.py               # Flexible time-window segmentation
│   ├── preprocessing.py              # Butterworth filtering & filter banks
│   └── visualize.py                  # Verification and analysis plots
├── run_preprocessing.py              # Master script to execute full pipeline
└── README.md
```

---

## 4. Usage

To run the pipeline from the workspace root:

```bash
python run_preprocessing.py
```

### Loading Preprocessed Data in Python:

```python
import numpy as np

data = np.load('preprocessed_data/S1_preprocessed.npz')

# Access segmented and preprocessed arrays:
X_train = data['X_train_full_filt']   # Shape: (160 trials, 10 channels, 400 samples)
y_train = data['y_train']             # Shape: (160,) labels: 1, 2, 3, 4
X_test  = data['X_test_full_filt']    # Shape: (74 trials, 10 channels, 400 samples)
time    = data['time_full']           # Shape: (400,) [-0.4s to 0.59s]

# Or post-onset window (t >= 0):
X_train_post = data['X_train_post_filt'] # Shape: (160, 10, 238)
```
