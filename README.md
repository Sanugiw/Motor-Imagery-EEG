# Motor Imagery EEG Paper Implementation

This repository reimplements and reproduces the methods of:

> Zhang, S., Zhu, Z., Zhang, B., Feng, B., Yu, T., & Li, Z. (2020). _The CSP-Based New Features Plus Non-Convex Log Sparse Feature Selection for Motor Imagery EEG Classification_. Sensors, 20(17), 4749. [https://doi.org/10.3390/s20174749](https://doi.org/10.3390/s20174749)

The paper is the original research; this codebase is an independent implementation in Python (NumPy, SciPy, scikit-learn, PyWavelets, MNE). All numbers in this README are outputs of this code, not values copied from the paper, unless marked "paper".

## What the paper proposes

Three feature extractors that apply CSP **first** and then analyse the 6 CSP-filtered signals:

1. **CSP-Wavelet:** db4 discrete wavelet transform; energy and standard deviation of the coefficients in the motor-imagery sub-bands.
2. **CSP-WPD:** the same statistics from db4 wavelet packet decomposition.
3. **CSP-FB:** a bank of 10 band-pass filters (8–30 Hz, 4 Hz wide, 2 Hz step); log-variance per band.

Features are selected in two stages:

- **Primary selection:** sparse regression with a non-convex log penalty (**LOG**), or **LASSO** for comparison.
- **Secondary selection:** a threshold ensemble. One FLDA model is trained for each weight threshold |w| > 0, 0.1, …, 0.8, and the best of the nine is reported.

## Methods implemented

| Method | Outline |
| ------ | ------- |
| **CSP** | CSP on 8–30 Hz EEG, normalised log-variance features, FLDA. |
| **Wavelet-CSP** | db4 DWT of each EEG channel, motor-imagery sub-bands rebuilt as channels, then CSP + FLDA. |
| **WPD-CSP** | The same with db4 wavelet packets. |
| **SFBCSP** | 17 overlapping 4 Hz bands (4–40 Hz) on unfiltered EEG, CSP per band, LASSO selection, FLDA. |
| **SBLFB** | The same features, sparse Bayesian learning selection, FLDA. |
| **CSP-Wavelet / CSP-WPD / CSP-FB + LOG** | Proposed features, LOG selection, threshold ensemble of FLDA models. |
| **CSP-Wavelet / CSP-WPD / CSP-FB + LASSO** | Proposed features, LASSO selection, threshold ensemble of FLDA models. |

## Repository layout

```text
Data/dataset_1 … dataset_4     raw recordings (see Datasets)
preprocessed_data/dataset_N    filtered, epoched trials (.npz)
src/preprocessing/             one preprocessing script per dataset
src/data/data_loader.py        loads preprocessed trials for every dataset
src/features/                  CSP, CSP-Wavelet, CSP-WPD, CSP-FB and baseline feature extractors
src/feature_selection/         LASSO, LOG, SBL, and F-score / GA / BPSO (Table 8)
src/classifiers/               threshold ensemble (FLDA, or SVM / BLDA for Table 9), BLDA
src/results_generation/        result generator, summary table + chart, Tables 6–9
src/visualize_eeg.py           inspection, segmentation and trial-timing plots
results/                       per-method accuracies, summaries, ablation tables
plots/                         EEG plots and the summary chart
```

## Pipeline

```text
raw files ─► preprocess_datasetN.py ─► preprocessed_data/*.npz
                                          │  data_loader.iter_dataset_trials()
                                          ▼
              result_runner.evaluate(method, trials)
              ├─ features/           CSP, CSP-Wavelet, CSP-WPD, CSP-FB, baselines
              ├─ feature_selection/  LASSO, LOG, SBL
              └─ classifiers/        threshold ensemble + FLDA
                                          ▼
              results/<method>/dataset_N/accuracies.csv
              ─► summary_table.py    summary table + Fig. 7-style chart
              ─► ablation_tables.py  Tables 6–9
```

### Preprocessing (paper Sections 2.1, 2.3)

Every script applies a sixth-order Butterworth 8–30 Hz band-pass with `sosfiltfilt` (zero phase) to the continuous recording and cuts epochs relative to the **cue**.

| Dataset | Recording | Train / test | Window after cue |
| ------- | --------- | ------------ | ---------------- |
| 1 · BCI IV-1 | 100 Hz, 59 channels | Calibration file: first 100 / last 100 trials | 0.5–3.5 s |
| 2 · BCI IV-2a | 250 Hz, 22 channels | Session T / session E; 6 binary class pairs per subject | 0.5–2.5 s |
| 3 · BCI IV-2b | 250 Hz, 3 channels | Session 03T only; first 80 / last 80 trials | 0.5–2.5 s |
| 4 · Steyrl 2016 | 512 → 256 Hz, 15 channels | Runs 1–5 (100) / runs 6–8 (60) | 0.5–2.5 s |

Dataset 4's markers are trial starts, and the cue follows 3 s later (`CUE_OFFSET` in `preprocess_dataset4.py`). Each `.npz` stores the filtered epochs (`X_train`, `X_test`), the unfiltered epochs of the same windows (`X_train_raw`, `X_test_raw`, used by SFBCSP and SBLFB), labels, sampling rate and channel names.

```powershell
python src/preprocessing/preprocess_dataset1.py
python src/preprocessing/preprocess_dataset2.py
python src/preprocessing/preprocess_dataset3.py --sessions 3
python src/preprocessing/preprocess_dataset4.py
```

### Features (Section 2.4)

- **CSP** (`csp_features.py`, Eqs. 1–5): trace-normalised class covariances; the generalised eigenproblem is solved as `eigh(C1, C1 + C2)`, which gives the same filters as C₂⁻¹C₁; m = 3 filter pairs.
- **Sub-band choice for CSP-Wavelet and CSP-WPD** (`mostly_inside()` in `csp_wavelet.py`): a sub-band is kept when more than half of it lies inside 8–30 Hz. This reproduces the sub-bands marked in the paper's Figures 3 and 4:

  | | 100 Hz (3 levels) | 250 / 256 Hz (4 levels) |
  | --- | --- | --- |
  | DWT | D3, D2 (6.25–25 Hz) | D4, D3 (7.8–31.25 Hz) |
  | WPD | S(3,1)–S(3,4) (6.25–31.25 Hz) | S(4,1)–S(4,3) (7.8–31.25 Hz) |

- **CSP-FB** (`csp_fb.py`): 10 bands, 6 CSP signals, 60 log-variance features.

### Feature selection and classification (Sections 2.5, 2.6)

- **LASSO** (`lasso_selection.py`): scikit-learn `Lasso` with `alpha = λ / n`, so the objective equals the paper's ½‖y − Xw‖² + λ‖w‖₁.
- **LOG** (`log_selection.py`): iterative log-thresholding (Eq. 17), a = 0.001.
- **λ for both:** 10-fold cross-validation over the paper's grid 2⁻⁵, 2⁻⁴·⁸, …, 2⁵.
- **Threshold ensemble** (`ensemble_flda.py`): one model per threshold 0, 0.1, …, 0.8; the reported score is the best test accuracy of the nine, as in the paper. The classifier can be swapped (`make_model`) for the Table 9 comparison.

## Running

```powershell
# accuracy per subject for every method and dataset
python src/results_generation/generate_results.py
python src/results_generation/generate_results.py --method "CSP-FB+LOG" --dataset dataset_4
python src/results_generation/generate_results.py --subject-limit 2      # quick trial run

# summary table (mean ± std) and Fig. 7-style bar chart
python src/results_generation/summary_table.py

# paper Tables 6–9 (about 1 hour on 7 cores; cached and resumable)
python src/results_generation/ablation_tables.py
python src/results_generation/ablation_tables.py --tables-only           # rebuild from cache

# EEG plots: raw vs filtered, segmentation, trial timing
python src/visualize_eeg.py --dataset all
```

`generate_results.py` runs the datasets listed in `DATASET_KEYS` in `result_runner.py`. It writes `results/<method>/dataset_N/accuracies.csv` and `summary.csv`, and updates only the rows it ran in `results/all_methods_summary.csv`.

| Output | Contents |
| ------ | -------- |
| `results/summary_table.md`, `.csv` | Mean ± std accuracy per method and dataset, plus "All data" |
| `plots/summary_accuracy.png` | Grouped bar chart in the style of the paper's Figure 7 |
| `results/ablation/ablation_tables.html`, `.tex`, `.md`, `table6…9.csv` | Paper Tables 6–9 in the paper's layout (the `.tex` needs `booktabs` and `multirow`) |
| `plots/dataset_N/trial_timing_*.png` | One trial per dataset with trial start, cue and epoch window marked |

"All data" pools all 84 units (7 + 54 + 9 + 14), as the paper does, and std is the population std; both reproduce the paper's own numbers.

## Results

Mean ± std classification accuracy (%), from `results/summary_table.md`:

| Method | Dataset 1 | Dataset 2 | Dataset 3 | Dataset 4 | All data |
| --- | --- | --- | --- | --- | --- |
| CSP | 84.00 ± 11.69 | 81.58 ± 13.19 | 79.31 ± 16.01 | 77.14 ± 13.38 | 80.80 ± 13.57 |
| Wavelet-CSP | 80.57 ± 14.78 | 81.47 ± 13.67 | 77.22 ± 16.20 | 77.14 ± 12.65 | 80.22 ± 14.02 |
| WPD-CSP | 83.00 ± 10.34 | 81.25 ± 13.33 | 76.53 ± 17.24 | 78.57 ± 12.71 | 80.44 ± 13.60 |
| SFBCSP | 77.00 ± 16.56 | 78.34 ± 13.39 | 69.31 ± 26.98 | 65.71 ± 12.77 | 75.16 ± 16.39 |
| SBLFB | 73.43 ± 11.93 | 76.48 ± 13.44 | 72.22 ± 15.07 | 65.12 ± 12.64 | 73.88 ± 14.01 |
| CSP-Wavelet+LOG | 86.43 ± 10.43 | 82.86 ± 12.33 | 77.22 ± 15.75 | 80.83 ± 12.83 | 82.21 ± 12.87 |
| CSP-WPD+LOG | 84.71 ± 11.95 | 82.93 ± 12.67 | 77.22 ± 14.78 | 78.45 ± 14.55 | 81.72 ± 13.40 |
| CSP-FB+LOG | 85.86 ± 12.76 | 83.24 ± 12.94 | 79.03 ± 15.17 | 82.02 ± 11.04 | 82.81 ± 12.99 |

Compared with the paper (All data: CSP 79.58, CSP-Wavelet+LOG 82.25, CSP-WPD+LOG 82.38, CSP-FB+LOG 82.3), the proposed methods reproduce closely and keep their advantage over CSP. SFBCSP and SBLFB stay below the paper on Datasets 3 and 4. In Tables 6–9 the paper's orderings hold: the secondary selection adds 2–4 points, LOG ≥ LASSO > F-score > GA/BPSO, and FLDA, SVM and BLDA are within 0.5 points of each other.

### Evaluation caveats

- **Test-set selection.** The threshold ensemble reports the best of nine models on the test set, as the paper does. This favours the ensemble methods over the single-model baselines. Without it (Table 7, "without secondary selection"), CSP-FB+LOG scores 78.64 against CSP's 80.80.
- **"All data" weighting.** Dataset 2 contributes 54 of the 84 units, but these come from only 9 subjects, each counted 6 times. Report per-dataset results alongside it, and use a paired test across subjects when claiming one method beats another.

## Implementation assumptions

The paper leaves the following details open. They are listed so the results can be interpreted and reproduced.

### Most likely to affect results

1. **Feature standardisation before LASSO/LOG.** Features are standardised (`StandardScaler`) before selection. The secondary thresholds 0.1–0.8 act on the resulting weights, so scaling changes which features pass them.
2. **GA and BPSO settings** (Table 8). 10 agents, 100 iterations, KNN (k = 5) on a 20% hold-out, fitness 0.99·error + 0.01·(selected ÷ total); GA: roulette selection, one-point crossover 0.8, mutation 0.01, elitist replacement; BPSO: c1 = c2 = 2, inertia 0.9 → 0.4, velocity limit 6, sigmoid transfer. These follow the paper's reference [67] from memory and have not been checked against it; they are the likeliest cause of GA/BPSO scoring 1–3 points below the paper.
3. **SBL pruning cutoff.** SBLFB drops features whose estimated precision is ≥ 10⁵, chosen from the observed gap between kept (≈10²) and pruned (≈10⁷·⁵) features. With the earlier cutoff of 10⁸, nothing was pruned and Dataset 1 accuracy was 52.9% instead of 73.4%.
4. **λ selection criterion.** Cross-validation uses the mean squared error of the regression on each validation fold (stratified, shuffled, seed 42). Classification accuracy would be the other natural choice and can pick a different λ.

### Other choices

| Area | Choice |
| ---- | ------ |
| Dataset 4 cue | Trial marker + 3 s, from the dataset's paradigm description; confirmed by accuracy rising from chance (51%) to 77% |
| Dataset 4 downsampling | Zero-phase FIR anti-aliasing decimation, 512 → 256 Hz |
| Dataset 2 evaluation cues | Event 783, falling back to trial start + 2 s |
| Dataset 3 artifact trials | Kept and flagged, not removed |
| Sub-band rule | More than half of the band inside 8–30 Hz (reproduces Figs. 3–4) |
| Wavelet edges | PyWavelets `mode="symmetric"` |
| Wavelet-CSP / WPD-CSP | Each kept sub-band becomes its own channel before CSP |
| SFBCSP / SBLFB | Filter order 6, 3 CSP pairs per band, one FLDA without the threshold ensemble |
| LASSO solver | scikit-learn coordinate descent instead of the SLEP toolbox |
| LOG solver | Starts from w = 0; at most 5,000 iterations; tolerance 1e-7 |
| "Without secondary selection" | FLDA on all features with \|w\| > 0 (the ensemble's threshold-0 model) |
| F-score | Fisher score on standardised features; subset size from 1 to all, chosen by 10-fold FLDA cross-validation |
| BLDA | scikit-learn `BayesianRidge` on ±1 labels, default priors, classified by sign |
| SVM | scikit-learn `SVC(kernel="linear", C=1)`, LIBSVM's default C |
| Seeds | 42 for cross-validation folds and the GA/BPSO searches |

## Differences from the paper still open

- **SFBCSP / SBLFB filtering.** The 17-band filters run on 2 s epochs that are already cut, so filter start-up effects corrupt part of each epoch. An experiment with 2 s of padding on both sides raised SFBCSP overall from 75.2% to 77.0% (Dataset 3 +6.8, Dataset 4 +3.8; Wilcoxon p = 0.055) and SBLFB from 73.9% to 74.9% (p = 0.11), and changed CSP-FB by less than 0.5 points. It is not adopted.
- **Reconstructed methods.** SBL, GA, BPSO and BLDA are written from the cited references, not the authors' code.

## Datasets

| Dataset | Description in the paper |
| ------- | ------------------------ |
| BCI Competition IV Dataset I | 59 channels at 100 Hz; 7 subjects imagine two of left hand, right hand or right foot. Two runs of 100 trials: one for training, one for testing. |
| BCI Competition IV Dataset IIa | 22 channels at 250 Hz; 9 subjects imagine left hand, right hand, foot or tongue. Session T trains, session E tests; the four classes form six binary pairs (54 subject-pair sets). |
| BCI Competition IV Dataset IIb | 3 channels at 250 Hz; 9 subjects imagine left or right hand. The third training session's 160 trials are split evenly into training and test. |
| Steyrl dataset (2016) | 15 channels at 512 Hz, downsampled to 256 Hz; 14 subjects imagine right hand or feet. Runs 1–5 give 100 training trials, runs 6–8 give 60 test trials. |

Sources: [BCI Competition IV](http://www.bbci.de/competition/IV/), [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets).
