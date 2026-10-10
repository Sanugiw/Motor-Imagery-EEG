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

## Reference

Zhang, S., Zhu, Z., Zhang, B., Feng, B., Yu, T., & Li, Z. (2020). *The CSP-Based New Features Plus Non-Convex Log Sparse Feature Selection for Motor Imagery EEG Classification*. Sensors, 20(17), 4749. [https://doi.org/10.3390/s20174749](https://doi.org/10.3390/s20174749)
