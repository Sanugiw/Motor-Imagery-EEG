# Motor Imagery EEG Paper Implementation

## Goal

The paper studies methods for classifying motor imagery from EEG. It aims to improve CSP-based feature extraction and select informative features efficiently, while maintaining high classification accuracy.

## Proposed approach

The paper introduces three CSP-based feature extraction methods:

1. **CSP-Wavelet:** apply a discrete wavelet transform to CSP-filtered signals and calculate statistics from the wavelet coefficients.
2. **CSP-WPD:** apply wavelet packet decomposition to CSP-filtered signals and calculate statistics from the resulting coefficients.
3. **CSP-FB:** apply a filter bank to CSP-filtered signals and extract log-variance features from the sub-bands.

The extracted features are selected using a sparse optimization method with non-convex log regularization (LOG). Ensemble learning then performs a second stage of feature selection and classification.
=======
| Method | Outline |
| ------ | ------- |
| **CSP** | Apply CSP to band-pass filtered EEG and classify the log-variance features with FLDA. |
| **Wavelet-CSP** | Decompose each EEG channel using db4 DWT, reconstruct motor-imagery sub-bands as channels, then extract CSP features. |
| **WPD-CSP** | Use db4 wavelet packet decomposition to reconstruct motor-imagery sub-bands as channels, then extract CSP features. |
| **SFBCSP** | Extract CSP features from overlapping frequency sub-bands and select features with LASSO. |
| **SBLFB** | Extract CSP features from overlapping frequency sub-bands and select features with sparse Bayesian learning. |
| **CSP-Wavelet+LOG** | Extract wavelet statistics from CSP-filtered signals, select features with non-convex log regularization (LOG), then use ensemble FLDA. |
| **CSP-WPD+LOG** | Extract wavelet-packet statistics from CSP-filtered signals, select features with LOG, then use ensemble FLDA. |
| **CSP-FB+LOG** | Extract filter-bank features from CSP-filtered signals, select features with LOG, then use ensemble FLDA. |
| **CSP-Wavelet+LASSO** | Same CSP-Wavelet feature pipeline, using LASSO for the first feature-selection stage before ensemble FLDA. |
| **CSP-WPD+LASSO** | Same CSP-WPD feature pipeline, using LASSO for the first feature-selection stage before ensemble FLDA. |
| **CSP-FB+LASSO** | Same CSP-FB feature pipeline, using LASSO for the first feature-selection stage before ensemble FLDA. |

All methods are run through the unified `src/results_generation/generate_results.py` script. It reports per-subject classification accuracy and dataset-level mean and standard deviation. Use `--method` to select a single method or leave it as `all` to run every method.

## Implementation details

### Source folders

- `src/preprocess_dataset1.py` through `src/preprocess_dataset4.py`: the four dataset preprocessing scripts.
- `src/features/`: CSP, wavelet, wavelet-packet, and filter-bank feature pipelines.
- `src/feature_selection/`: LASSO, LOG, and sparse Bayesian selectors.
- `src/classifiers/`: FLDA ensemble implementation.
- `src/data/`: loading utilities for preprocessed trials.
- `src/results_generation/`: the unified result generator and shared pipeline helper.
- `src/visualize.py` and `src/visualize_eeg.py`: EEG inspection and plotting.

### Preprocessing

Run the matching preprocessing script before generating classifier results:

```powershell
python src/preprocess_dataset1.py
python src/preprocess_dataset2.py
python src/preprocess_dataset3.py
python src/preprocess_dataset4.py
```

The scripts read data from `Data/dataset_1` through `Data/dataset_4` and save compressed trial arrays under `preprocessed_data/dataset_1` through `preprocessed_data/dataset_4`. Each saved file contains train/test epochs, labels, sampling rate, and channel metadata. The common filter is a sixth-order 8–30 Hz Butterworth band-pass; epochs are cut from 0.5 to 2.5 seconds, except Dataset 1, which uses 0.5 to 3.5 seconds.

| Dataset | Data used by preprocessing |
| ------- | -------------------------- |
| `Data/dataset_1` | BCI Competition IV Dataset I calibration recordings; 100 train and 100 test trials per subject. |
| `Data/dataset_2` | BCI Competition IV Dataset IIa, 22 EEG channels; full four-class files and six two-class pairs per subject. Test labels are read from `true_labels` when available. |
| `Data/dataset_3` | BCI Competition IV Dataset IIb; the third training session has 160 trials per subject, split into 80 train and 80 test trials. |
| `Data/dataset_4` | Steyrl dataset; recordings are resampled from 512 Hz to 256 Hz when necessary, with 100 train and 60 test trials per subject. |

### Classifier pipeline

- CSP learns three spatial-filter pairs from the training data and uses normalized log-variance features with FLDA.
- Wavelet-CSP and WPD-CSP reconstruct motor-imagery sub-band signals with db4 DWT or WPD before CSP. The decomposition level is 3 for Dataset 1 and 4 for the other datasets.
- SFBCSP and SBLFB calculate CSP features on 17 overlapping 4 Hz bands spanning 4–40 Hz. They select features using LASSO and sparse Bayesian learning, respectively.
- CSP-Wavelet, CSP-WPD, and CSP-FB extract wavelet, packet, or filter-bank features from CSP-projected signals. Their LOG and LASSO variants apply feature selection followed by an FLDA threshold ensemble.
- The LOG and LASSO selectors use stratified 10-fold cross-validation to choose a regularization parameter. The FLDA ensemble evaluates coefficient thresholds from 0.0 to 0.8 in increments of 0.1.

### Generating results

Run all methods on all four datasets:

```powershell
python src/results_generation/generate_results.py
```

This writes each method's subject-level accuracy and mean/std files under `results/<method>/dataset_<dataset>/`, plus `results/all_methods_summary.csv`. The method-specific folders are ignored by Git; the combined summary stays trackable. Run all methods on one dataset with `python src/results_generation/generate_results.py --dataset 2a`. Run only CSP on all datasets with `python src/results_generation/generate_results.py --method CSP`:

```powershell
python src/results_generation/generate_results.py --method "CSP-Wavelet+LOG" --dataset 2a
```

For each selected method, the generator writes per-subject accuracies to `results/<method>/dataset_<dataset>/accuracies.csv` and mean/std summaries to `summary.csv`. Dataset 2a produces one row per subject and binary class pair. A short run can be limited with `--subject-limit N`.

Generate EEG inspection plots with `python src/visualize.py --dataset all`.

## Results currently in the repository

Existing CSP results cover all four datasets. Accuracy is shown as mean ± standard deviation in percent across subjects (Dataset 2a counts each subject-pair result as one row).

| Dataset | Result rows | CSP accuracy (%) |
| ------- | ----------: | ---------------: |
| 1 | 7 | 84.00 ± 11.69 |
| 2a | 54 | 81.58 ± 13.19 |
| 2b | 9 | 79.31 ± 16.01 |
| 4 | 14 | 51.43 ± 4.83 |

An earlier combined-runner snapshot for Dataset 1 is also stored at `results/paper_reproduction/dataset_1/paper_table.csv`. The LASSO variants do not have saved results yet; generate them with `--method "CSP-Wavelet+LASSO"` and the corresponding method name. These numbers are project outputs, not values copied from the paper.

**Evaluation caveat:** the current LOG/LASSO ensemble score is the best accuracy among the threshold models evaluated on the test labels. This uses the test set to choose the threshold and can make those ensemble results optimistic. Use a validation split to select the threshold before treating them as unbiased test accuracy.
## Evaluation

The proposed methods are evaluated on four public motor imagery EEG datasets. The experiments compare classification accuracy and feature extraction time against other methods. The paper reports CSP-FB + LOG as a strong option when considering both accuracy and computation time.

## Datasets

| Dataset                        | Description in the paper                                                                                                                                                                                                     |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BCI Competition IV Dataset I   | 59 channels at 100 Hz; 7 subjects imagine two of three movements (left hand, right hand, or right foot). The paper uses two runs of 100 trials each, with one run for training and the other for testing.                    |
| BCI Competition IV Dataset IIa | 22 channels at 250 Hz; 9 subjects imagine left hand, right hand, foot, or tongue movement. The two sessions are used for training and testing, and the four classes form six binary task pairs (54 subject/task sets total). |
| BCI Competition IV Dataset IIb | 3 channels at 250 Hz; 9 subjects perform left-hand or right-hand motor imagery. The paper uses the third training session, with 160 trials split evenly between training and testing.                                        |
| David Steyrl dataset (2016)    | 15 channels recorded at 512 Hz and downsampled to 256 Hz; 14 subjects imagine right-hand or foot movement. Runs 1–5 provide 100 training trials, and runs 6–8 provide 60 test trials per subject.                            |

## Reference

Zhang, S., Zhu, Z., Zhang, B., Feng, B., Yu, T., & Li, Z. (2020). _The CSP-Based New Features Plus Non-Convex Log Sparse Feature Selection for Motor Imagery EEG Classification_. Sensors, 20(17), 4749. [https://doi.org/10.3390/s20174749](https://doi.org/10.3390/s20174749)
