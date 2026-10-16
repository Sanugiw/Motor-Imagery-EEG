Average classification accuracy (%). "All data" pools every subject and Dataset 2 class pair, as in the paper's Tables 6-9.

**Table 6.** The effect of secondary feature selection on average classification accuracy of various datasets (the LASSO method).

| Datasets | Without Secondary Feature Selection: CSP-Wavelet+LASSO | Without Secondary Feature Selection: CSP-WPD+LASSO | Without Secondary Feature Selection: CSP-FB+LASSO | With Secondary Feature Selection: CSP-Wavelet+LASSO | With Secondary Feature Selection: CSP-WPD+LASSO | With Secondary Feature Selection: CSP-FB+LASSO |
|---|---|---|---|---|---|---|
| dataset 1 | 81.14 | 81.43 | 81.29 | 84.14 | 83.14 | 86.14 |
| dataset 2 | 80.12 | 80.65 | 81.04 | 82.48 | 83.04 | 83.73 |
| dataset 3 | 76.67 | 75.14 | 72.22 | 77.22 | 76.53 | 74.86 |
| dataset 4 | 76.55 | 74.05 | 76.90 | 79.88 | 78.45 | 80.71 |
| All data | 79.24 | 79.02 | 79.43 | 81.62 | 81.58 | 82.48 |

**Table 7.** The effect of secondary feature selection on average classification accuracy of various datasets (the LOG method).

| Datasets | Without Secondary Feature Selection: CSP-Wavelet+LOG | Without Secondary Feature Selection: CSP-WPD+LOG | Without Secondary Feature Selection: CSP-FB+LOG | With Secondary Feature Selection: CSP-Wavelet+LOG | With Secondary Feature Selection: CSP-WPD+LOG | With Secondary Feature Selection: CSP-FB+LOG |
|---|---|---|---|---|---|---|
| dataset 1 | 84.00 | 83.00 | 81.00 | 86.43 | 84.71 | 85.86 |
| dataset 2 | 80.45 | 80.65 | 79.41 | 82.86 | 82.93 | 83.24 |
| dataset 3 | 76.11 | 76.94 | 75.97 | 77.22 | 77.22 | 79.03 |
| dataset 4 | 77.86 | 75.60 | 76.19 | 80.83 | 78.45 | 82.02 |
| All data | 79.85 | 79.60 | 78.64 | 82.21 | 81.72 | 82.81 |

**Table 8.** The effect of different feature selection methods on average classification accuracy of various datasets.

| Feature Extraction | Feature Selection | Dataset 1 | Dataset 2 | Dataset 3 | Dataset 4 | All Data |
|---|---|---|---|---|---|---|
| CSP-Wavelet | F-score | 81.43 | 80.09 | 73.61 | 77.50 | 79.08 |
|  | GA | 72.43 | 78.31 | 72.22 | 75.95 | 76.77 |
|  | BPSO | 77.57 | 77.53 | 74.44 | 76.07 | 76.96 |
|  | LASSO | 84.14 | 82.48 | 77.22 | 79.88 | 81.62 |
|  | LOG | 86.43 | 82.86 | 77.22 | 80.83 | 82.21 |
| CSP-WPD | F-score | 82.57 | 79.48 | 74.72 | 76.55 | 78.74 |
|  | GA | 73.71 | 78.15 | 75.28 | 74.29 | 76.83 |
|  | BPSO | 77.43 | 78.97 | 74.72 | 72.62 | 77.33 |
|  | LASSO | 83.14 | 83.04 | 76.53 | 78.45 | 81.58 |
|  | LOG | 84.71 | 82.93 | 77.22 | 78.45 | 81.72 |
| CSP-FB | F-score | 83.29 | 81.30 | 79.03 | 76.79 | 80.47 |
|  | GA | 76.43 | 78.97 | 74.58 | 77.26 | 78.01 |
|  | BPSO | 79.29 | 80.63 | 73.61 | 77.14 | 79.19 |
|  | LASSO | 86.14 | 83.73 | 74.86 | 80.71 | 82.48 |
|  | LOG | 85.86 | 83.24 | 79.03 | 82.02 | 82.81 |

**Table 9.** The effect of different classifiers on average classification accuracy of various datasets.

| Datasets | CSP-Wavelet+LOG: SVM | CSP-Wavelet+LOG: BLDA | CSP-Wavelet+LOG: FLDA | CSP-WPD+LOG: SVM | CSP-WPD+LOG: BLDA | CSP-WPD+LOG: FLDA | CSP-FB+LOG: SVM | CSP-FB+LOG: BLDA | CSP-FB+LOG: FLDA |
|---|---|---|---|---|---|---|---|---|---|
| dataset 1 | 85.43 | 85.86 | 86.43 | 83.86 | 83.86 | 84.71 | 87.14 | 86.29 | 85.86 |
| dataset 2 | 82.63 | 83.01 | 82.86 | 82.69 | 83.09 | 82.93 | 83.06 | 83.81 | 83.24 |
| dataset 3 | 78.19 | 78.47 | 77.22 | 78.19 | 77.36 | 77.22 | 78.75 | 79.86 | 79.03 |
| dataset 4 | 80.95 | 80.95 | 80.83 | 79.52 | 79.52 | 78.45 | 83.33 | 81.19 | 82.02 |
| All data | 82.11 | 82.42 | 82.21 | 81.78 | 81.95 | 81.72 | 82.99 | 83.16 | 82.81 |
