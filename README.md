# Paper Outline

## Goal

The paper studies methods for classifying motor imagery from EEG. It aims to improve CSP-based feature extraction and select informative features efficiently, while maintaining high classification accuracy.

## Proposed approach

The paper introduces three CSP-based feature extraction methods:

1. **CSP-Wavelet:** apply a discrete wavelet transform to CSP-filtered signals and calculate statistics from the wavelet coefficients.
2. **CSP-WPD:** apply wavelet packet decomposition to CSP-filtered signals and calculate statistics from the resulting coefficients.
3. **CSP-FB:** apply a filter bank to CSP-filtered signals and extract log-variance features from the sub-bands.

The extracted features are selected using a sparse optimization method with non-convex log regularization (LOG). Ensemble learning then performs a second stage of feature selection and classification.

## Evaluation

The proposed methods are evaluated on four public motor imagery EEG datasets. The experiments compare classification accuracy and feature extraction time against other methods. The paper reports CSP-FB + LOG as a strong option when considering both accuracy and computation time.

## Datasets

| Dataset | Description in the paper |
| --- | --- |
| BCI Competition IV Dataset I | 59 channels at 100 Hz; 7 subjects imagine two of three movements (left hand, right hand, or right foot). The paper uses two runs of 100 trials each, with one run for training and the other for testing. |
| BCI Competition IV Dataset IIa | 22 channels at 250 Hz; 9 subjects imagine left hand, right hand, foot, or tongue movement. The two sessions are used for training and testing, and the four classes form six binary task pairs (54 subject/task sets total). |
| BCI Competition IV Dataset IIb | 3 channels at 250 Hz; 9 subjects perform left-hand or right-hand motor imagery. The paper uses the third training session, with 160 trials split evenly between training and testing. |
| David Steyrl dataset (2016) | 15 channels recorded at 512 Hz and downsampled to 256 Hz; 14 subjects imagine right-hand or foot movement. Runs 1–5 provide 100 training trials, and runs 6–8 provide 60 test trials per subject. |

## Reference

Zhang, S., Zhu, Z., Zhang, B., Feng, B., Yu, T., & Li, Z. (2020). *The CSP-Based New Features Plus Non-Convex Log Sparse Feature Selection for Motor Imagery EEG Classification*. Sensors, 20(17), 4749. [https://doi.org/10.3390/s20174749](https://doi.org/10.3390/s20174749)
