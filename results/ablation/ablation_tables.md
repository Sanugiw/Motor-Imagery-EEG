Average classification accuracy (%), as in the paper's Tables 6-9.

### Table 6. Effect of secondary feature selection (LASSO)

| Datasets | CSP-Wavelet+LASSO (without) | CSP-WPD+LASSO (without) | CSP-FB+LASSO (without) | CSP-Wavelet+LASSO (with) | CSP-WPD+LASSO (with) | CSP-FB+LASSO (with) |
|---|---|---|---|---|---|---|
| Dataset 1 | 81.14 | 81.43 | 81.29 | 84.14 | 83.14 | 86.14 |
| Dataset 2 | 80.12 | 80.65 | 81.04 | 82.48 | 83.04 | 83.73 |
| Dataset 3 | 76.67 | 75.14 | 72.22 | 77.22 | 76.53 | 74.86 |
| Dataset 4 | 76.55 | 74.05 | 76.90 | 79.88 | 78.45 | 80.71 |
| All data | 79.24 | 79.02 | 79.43 | 81.62 | 81.58 | 82.48 |

### Table 7. Effect of secondary feature selection (LOG)

| Datasets | CSP-Wavelet+LOG (without) | CSP-WPD+LOG (without) | CSP-FB+LOG (without) | CSP-Wavelet+LOG (with) | CSP-WPD+LOG (with) | CSP-FB+LOG (with) |
|---|---|---|---|---|---|---|
| Dataset 1 | 84.00 | 83.00 | 81.00 | 86.43 | 84.71 | 85.86 |
| Dataset 2 | 80.45 | 80.65 | 79.41 | 82.86 | 82.93 | 83.24 |
| Dataset 3 | 76.11 | 76.94 | 75.97 | 77.22 | 77.22 | 79.03 |
| Dataset 4 | 77.86 | 75.60 | 76.19 | 80.83 | 78.45 | 82.02 |
| All data | 79.85 | 79.60 | 78.64 | 82.21 | 81.72 | 82.81 |

### Table 8. Effect of feature selection method

| Feature extraction | Feature selection | Dataset 1 | Dataset 2 | Dataset 3 | Dataset 4 | All data |
|---|---|---|---|---|---|---|
| CSP-Wavelet | F-score | 81.43 | 80.09 | 73.61 | 77.50 | 79.08 |
| CSP-Wavelet | GA | 72.43 | 78.31 | 72.22 | 75.95 | 76.77 |
| CSP-Wavelet | BPSO | 77.57 | 77.53 | 74.44 | 76.07 | 76.96 |
| CSP-Wavelet | LASSO | 84.14 | 82.48 | 77.22 | 79.88 | 81.62 |
| CSP-Wavelet | LOG | 86.43 | 82.86 | 77.22 | 80.83 | 82.21 |
| CSP-WPD | F-score | 82.57 | 79.48 | 74.72 | 76.55 | 78.74 |
| CSP-WPD | GA | 73.71 | 78.15 | 75.28 | 74.29 | 76.83 |
| CSP-WPD | BPSO | 77.43 | 78.97 | 74.72 | 72.62 | 77.33 |
| CSP-WPD | LASSO | 83.14 | 83.04 | 76.53 | 78.45 | 81.58 |
| CSP-WPD | LOG | 84.71 | 82.93 | 77.22 | 78.45 | 81.72 |
| CSP-FB | F-score | 83.29 | 81.30 | 79.03 | 76.79 | 80.47 |
| CSP-FB | GA | 76.43 | 78.97 | 74.58 | 77.26 | 78.01 |
| CSP-FB | BPSO | 79.29 | 80.63 | 73.61 | 77.14 | 79.19 |
| CSP-FB | LASSO | 86.14 | 83.73 | 74.86 | 80.71 | 82.48 |
| CSP-FB | LOG | 85.86 | 83.24 | 79.03 | 82.02 | 82.81 |

### Table 9. Effect of classifier (LOG selection)

| Datasets | CSP-Wavelet+LOG SVM | CSP-Wavelet+LOG BLDA | CSP-Wavelet+LOG FLDA | CSP-WPD+LOG SVM | CSP-WPD+LOG BLDA | CSP-WPD+LOG FLDA | CSP-FB+LOG SVM | CSP-FB+LOG BLDA | CSP-FB+LOG FLDA |
|---|---|---|---|---|---|---|---|---|---|
| Dataset 1 | 85.43 | 85.86 | 86.43 | 83.86 | 83.86 | 84.71 | 87.14 | 86.29 | 85.86 |
| Dataset 2 | 82.63 | 83.01 | 82.86 | 82.69 | 83.09 | 82.93 | 83.06 | 83.81 | 83.24 |
| Dataset 3 | 78.19 | 78.47 | 77.22 | 78.19 | 77.36 | 77.22 | 78.75 | 79.86 | 79.03 |
| Dataset 4 | 80.95 | 80.95 | 80.83 | 79.52 | 79.52 | 78.45 | 83.33 | 81.19 | 82.02 |
| All data | 82.11 | 82.42 | 82.21 | 81.78 | 81.95 | 81.72 | 82.99 | 83.16 | 82.81 |
