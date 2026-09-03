# PDE-regularized-Dynamics-informed-Diffusion-with-Uncertainty-aware-Filtering-
pre-print: SIAM Journal on Mathematics of DataScience([arXiv:2604.09058](https://arxiv.org/abs/2604.09058))

# Requirement
Python: 3.8.20
CUDA : 12.4
GPU: RTX A6000

# Main result

### (a) SST

| Method | CRPS | MSE | SSR | Time [s] |
|---|---|---|---|---|
| Perturbation | 0.281 ± 0.004 | 0.180 ± 0.011 | 0.411 ± 0.046 | 0.4241 |
| Dropout | 0.266 ± 0.003 | <ins>0.164 ± 0.004</ins> | 0.406 ± 0.042 | 0.4242 |
| DDPM | 0.246 ± 0.005 | 0.176 ± 0.005 | 0.674 ± 0.011 | 0.3054 |
| MCVD | **0.216 ± 0.056** | **0.161 ± 0.049** | <ins>0.926 ± 0.085</ins> | 78.9950 |
| DYffusion | <ins>0.224 ± 0.002</ins> | 0.173 ± 0.001 | **1.033 ± 0.005** | 4.6722 |
| PDYffusion | 0.226 ± 0.001 | 0.166 ± 0.003 | 0.683 ± 0.006 | 13.0880 |

### (b) Navier–Stokes

| Method | CRPS | MSE | SSR | Time [s] |
|---|---|---|---|---|
| Perturbation | 0.090 ± 0.001 | 0.028 ± 0.000 | 0.448 ± 0.002 | 0.0895 |
| Dropout | 0.078 ± 0.001 | 0.027 ± 0.001 | 0.715 ± 0.005 | 0.0886 |
| DDPM | 0.180 ± 0.004 | 0.105 ± 0.010 | 0.573 ± 0.001 | 0.0692 |
| MCVD | 0.152 ± 0.044 | 0.070 ± 0.033 | 0.524 ± 0.064 | 58.4255 |
| DYffusion | <ins>0.067 ± 0.003</ins> | <ins>0.022 ± 0.002</ins> | **0.877 ± 0.006** | 2.9715 |
| PDYffusion | **0.059 ± 0.005** | **0.017 ± 0.002** | <ins>0.731 ± 0.008</ins> | 10.3917 |

### (c) Spring-mesh

| Method | CRPS | MSE | SSR | Time [s] |
|---|---|---|---|---|
| Perturbation | 0.0151 ± 0.0004 | 9.05e−4 ± 6.55e−5 | 1.361 ± 0.038 | 0.0152 |
| Dropout | 0.0138 ± 0.0006 | 7.27e−4 ± 6.80e−5 | **1.017 ± 0.029** | 0.0161 |
| DDPM | 0.0165 ± 0.0010 | 14.13e−4 ± 1.05e−4 | 0.763 ± 0.009 | 0.0098 |
| MCVD | 0.0146 ± 0.0094 | 6.77e−4 ± 1.42e−4 | 0.782 ± 0.055 | 33.4163 |
| DYffusion | <ins>0.0103 ± 0.0022</ins> | <ins>4.20e−4 ± 2.10e−4</ins> | <ins>1.133 ± 0.083</ins> | 1.1005 |
| PDYffusion | **0.0092 ± 0.0019** | **4.01e−4 ± 2.21e−4** | 1.204 ± 0.095 | 6.2841 |

### (d) Wave

| Method | CRPS | MSE | SSR | Time [s] |
|---|---|---|---|---|
| Perturbation | 4.88e−3 ± 2.30e−4 | 3.30e−5 ± 4.04e−6 | <ins>1.164 ± 0.022</ins> | 0.0008 |
| Dropout | 6.51e−3 ± 3.18e−4 | 3.26e−5 ± 4.86e−6 | 1.214 ± 0.065 | 0.0008 |
| DDPM | 9.85e−3 ± 2.30e−4 | 14.31e−5 ± 5.27e−6 | 1.901 ± 0.079 | 0.0001 |
| MCVD | 9.31e−3 ± 4.86e−4 | 26.34e−5 ± 6.11e−6 | 1.400 ± 0.091 | 7.2594 |
| DYffusion | <ins>2.02e−3 ± 3.46e−4</ins> | <ins>1.94e−5 ± 3.43e−6</ins> | 0.836 ± 0.043 | 0.5258 |
| PDYffusion | **1.82e−3 ± 3.01e−4** | **1.57e−5 ± 4.50e−6** | **0.913 ± 0.056** | 3.7763 |

# Acknowledgement
This repository was developed with support from the 서울시립대학교 데이터 사이언스 플러스 차세대 융합인재 양성사업단 - http://dsplus.uos.ac.kr/
