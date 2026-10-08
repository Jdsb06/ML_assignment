# Machine Learning Assignment 1: Polynomial Regression

**Author:** Jashandeep Singh Bedi (Roll Number: `IMT2024022`)  
**Institution:** International Institute of Information Technology, Bangalore (IIIT-B)  
**Course:** Machine Learning (SEM 5)  
**Repository:** [https://github.com/Jdsb06/ML_assignment](https://github.com/Jdsb06/ML_assignment)

---

## 1. Overview

This project implements regularized polynomial regression architectures to solve two distinct real-world geothermal energy engineering problems using personalized datasets for roll number **`IMT2024022`**:

1. **Phase 1: Power Plant Steam Turbine Optimization (`var1`)**
   - **Goal:** Predict the **Net Power Score ($y$)** of a multi-stage steam turbine from 6 operational percentage deviation parameters ($x_1, \dots, x_6$).
   - **Optimal Architecture:** Degree 5 Polynomial Regression with Feature Standardization and **$L_1$ Lasso Regularization** ($\alpha = 0.015$).
   - **Performance (10-Fold CV):** **$\text{MSE} = 0.33139 \pm 0.04240$**, **$R^2 = 0.96681 \pm 0.00550$**.
   - **Boundary Clamping & Covariate Shift:** Auditing coordinate boundaries revealed that 51.25% of test coordinates are clamped at $\pm 1.0$ versus 31.17% in train. An extreme-boundary holdout evaluation confirmed that $\alpha = 0.015$ reduces boundary holdout MSE to $0.4750$ (down from $0.5309$ at $\alpha=0.007$).
   - **Sparsity:** Retains 97 active non-zero monomials (full fit) out of 461 (21.0% active), pruning 79.0% of uninformative interactions.

2. **Phase 2: Subterranean Thermal Reservoir Mapping (`var2`)**
   - **Goal:** Predict the **Thermal Anomaly Score ($y$)** across a 3D geological survey block from spatial offset coordinates ($x_1, x_2, x_3$).
   - **Optimal Architecture:** Degree 10 Polynomial Regression with Feature Standardization and **$L_2$ Ridge Regularization** ($\alpha = 0.7$).
   - **Performance (10-Fold CV):** **$\text{MSE} = 0.26516 \pm 0.07293$**, **$R^2 = 0.99510 \pm 0.00141$**.
   - **Parsimony vs. Nominal Minimum:** Across multiple random CV seeds, Degree 8 Ridge ($\alpha = 0.05$, $\text{MSE} = 0.26907 \pm 0.06560$, $R^2 = 0.99503$) and Degree 10 Ridge differ by merely $\approx 0.0039$ MSE, well within fold standard error ($\approx 0.07$). Degree 8 serves as an equally competitive, parsimonious model with 42% fewer terms (164 vs 285).

---

## 2. Repository Layout

The repository is modularly structured into dedicated directories for data, core code, exploratory experiment scripts, predictions, reports, and visualization figures:

```text
ML_assignment/
├── data/                               # Canonical dataset files
│   ├── IMT2024022_train_var1.csv       # Training dataset for Phase 1 (N=1000, 6 features)
│   ├── IMT2024022_test_var1.csv        # Testing dataset for Phase 1 (N=1000)
│   ├── IMT2024022_train_var2.csv       # Training dataset for Phase 2 (N=1000, 3 features)
│   ├── IMT2024022_test_var2.csv        # Testing dataset for Phase 2 (N=1000)
│   └── sample_submission.csv           # Benchmark submission format sample
│
├── code/                               # Production training and inference code
│   ├── train_var1.py                   # Self-contained training & inference script for Phase 1
│   ├── train_var2.py                   # Self-contained training & inference script for Phase 2
│   ├── generate_predictions.py         # End-to-end execution & validation script
│   ├── generate_final_models.py        # Generates figures & verifies predictions from benchmark CSVs
│   ├── eda.py                          # Exploratory data analysis script
│   └── experiments/                    # Research and exploratory tuning scripts
│       ├── model_search.py             # Initial grid search across degrees
│       ├── advanced_analysis.py        # Monomial interaction audits and penalty sweeps
│       ├── explore_var2.py             # Spatial coordinate analysis for Phase 2
│       ├── deep_dive.py                # CV stability & boundary clamping analysis for Phase 1
│       ├── deep_dive_var2.py           # Multi-seed evaluation of Degree 8 vs 10 for Phase 2
│       ├── final_tuning.py             # Fine-grained alpha hyperparameter tuning
│       ├── clamping_stratified_analysis.py # Stratified clamping error & importance weighting analysis
│       ├── run_systematic_benchmark.py # Runs reproducible 10-fold CV benchmark for all models
│       ├── var1_benchmark.csv          # Canonical benchmark CSV for Phase 1
│       └── var2_benchmark.csv          # Canonical benchmark CSV for Phase 2
│
├── predictions/                        # Canonical prediction outputs
│   ├── IMT2024022_pred_var1.csv        # Final test predictions for Phase 1 (1000 rows, header 'y')
│   └── IMT2024022_pred_var2.csv        # Final test predictions for Phase 2 (1000 rows, header 'y')
│
├── figures/                            # High-resolution evaluation figures (300 DPI)
│   ├── actual_vs_predicted.png         # Out-of-fold actual vs predicted scatter plots
│   ├── model_comparison_degrees.png    # Validation error vs degree curves (OLS vs Ridge vs Lasso)
│   ├── residual_analysis.png           # Residual distributions
│   └── train_test_distribution.png     # Train vs test prediction distribution alignment
│
├── report/                             # Technical report documentation
│   ├── report.tex                      # LaTeX source code (strict 5-page layout)
│   ├── report.pdf                      # Compiled technical report PDF
│   └── figures/                        # Figures embedded in the report
│
├── report.pdf                          # Root copy of report PDF for easy grading access
├── IMT2024022_pred_var1.csv            # Root mirror of Phase 1 predictions for grader convenience
├── IMT2024022_pred_var2.csv            # Root mirror of Phase 2 predictions for grader convenience
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # Documentation
```

---

## 3. Results Summary (10-Fold Cross-Validation)

All models are evaluated on the exact same 10-fold cross-validation splits (`random_state=42`, shuffle enabled). Benchmark numbers are loaded directly from `code/experiments/var1_benchmark.csv` and `code/experiments/var2_benchmark.csv`.

### Phase 1: Steam Turbine Optimization (`var1`)
| Model | Degree | Parameters ($p$) | 10-Fold CV MSE | 10-Fold CV $R^2$ | Active Terms | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| OLS | 1 | 6 | 9.3333 | 0.0781 | 6 | Severe underfitting |
| OLS | 2 | 27 | 2.9318 | 0.7063 | 27 | Captures quadratic curvature |
| OLS | 3 | 83 | 1.0140 | 0.8986 | 83 | Captures cubic interactions |
| OLS | 4 | 209 | 0.7723 | 0.9229 | 209 | Multicollinearity emerges |
| Lasso ($\alpha=0.007$) | 4 | 209 | 0.5877 | 0.9410 | 112 | Prunes quartic interactions |
| OLS | 5 | 461 | 1.3374 | 0.8691 | 461 | High-dimensional overfitting ($p \approx N/2$) |
| Ridge ($\alpha=20$) | 5 | 461 | 0.5063 | 0.9491 | 461 | Shrinkage without sparsity |
| **Lasso ($\alpha=0.015$)** | **5** | **461** | **0.3314** | **0.9668** | **97** | **Optimal Model (Min Boundary Holdout MSE = 0.4750)** |
| OLS | 6 | 923 | 2342.97 | -220.61 | 923 | Singular Gram matrix ($p=923 > N_{\text{fold}}=900$) |
| Ridge ($\alpha=30$) | 6 | 923 | 0.6437 | 0.9348 | 923 | Ridge regularizes ill-conditioned Gram matrix |
| Lasso ($\alpha=0.02$) | 6 | 923 | 0.3611 | 0.9639 | 104 | Higher variance than Degree 5 |

### Phase 1: Stratified Error by Clamped Coordinates & Importance Weighting

An audit of coordinate clamping at domain boundaries ($\pm 1.0$) reveals an acute covariate shift between training (31.17% clamped coordinates) and test (51.25% clamped coordinates). Below is the 10-fold cross-validation performance of Degree 5 Lasso ($\alpha = 0.015$) vs. unregularized OLS stratified by the number of clamped coordinates ($k \in \{0, \dots, 6\}$):

| Clamping Stratum ($k$) | Train Dist. ($N$) | Test Dist. ($N$) | Degree 5 Lasso ($\alpha=0.015$) MSE | Degree 5 OLS MSE |
|---|:---:|:---:|:---:|:---:|
| $k=0$ (Interior) | 11.2% (112) | 1.1% (11) | 0.2312 | 0.5155 |
| $k=1$ | 30.6% (306) | 8.6% (86) | 0.3025 | 0.7617 |
| $k=2$ | 29.5% (295) | 23.8% (238) | 0.3078 | 1.2026 |
| $k=3$ | 19.2% (192) | 29.3% (293) | 0.3758 | 2.8806 |
| $k=4$ | 7.9% (79) | 24.6% (246) | 0.5506 | 1.4410 |
| $k=5$ | 1.4% (14) | 10.6% (106) | 0.4250 | 1.7151 |
| $k=6$ (Boundary) | 0.2% (2) | 2.0% (20) | 0.2737 | 0.4236 |
| **Overall CV MSE** | **100% (1000)** | — | **0.3314** | **1.3374** |
| **Importance-Weighted MSE** | — | **100% (1000)** | **0.3979** (+20.1% vs CV) | **1.7462** (+30.6% vs CV) |

**Key Takeaways:**
1. **CV MSE is optimistic:** Because the training set overrepresents interior points ($71.3\%$ with $k \le 2$) where prediction error is lower, standard 10-fold CV MSE ($0.3314$) is optimistic for the true test distribution.
2. **Importance-Weighted MSE:** Re-weighting the stratum errors by the test distribution $P_{\text{test}}(k)$ yields an expected test error of **$\text{MSE}_{\text{IW}} = 0.3979$** ($\approx 20.1\%$ higher).
3. **Phase 2 Clamping:** In Phase 2, boundary clamping also occurs at $\pm 1.0$ (26.17% of train coordinates vs. 24.83% of test coordinates), but the distribution across strata $k \in \{0, 1, 2, 3\}$ is balanced between train and test ($40.5\%$ vs $42.9\%$ at $k=0$, $42.1\%$ vs $41.5\%$ at $k=1$, $15.8\%$ vs $13.8\%$ at $k=2$, $1.6\%$ vs $1.8\%$ at $k=3$), indicating absence of significant covariate shift.

### Phase 2: Thermal Reservoir Mapping (`var2`)
| Model | Degree | Parameters ($p$) | 10-Fold CV MSE | 10-Fold CV $R^2$ | Active Terms | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| OLS | 2 | 9 | 23.9292 | 0.5630 | 9 | Severe spatial underfitting |
| OLS | 4 | 34 | 4.0213 | 0.9258 | 34 | Rapid error reduction |
| OLS | 6 | 83 | 0.5587 | 0.9898 | 83 | Sub-unit MSE reached |
| Ridge ($\alpha=0.1$) | 7 | 119 | 0.3519 | 0.9935 | 119 | Smooth spatial gradient |
| OLS | 8 | 164 | 0.2709 | 0.9950 | 164 | Unregularized peak |
| Ridge ($\alpha=0.05$) | 8 | 164 | 0.2691 | 0.9950 | 164 | **Parsimonious Model** (within fold noise of d10) |
| OLS | 10 | 285 | 0.3710 | 0.9930 | 285 | Boundary oscillation flaring |
| **Ridge ($\alpha=0.7$)** | **10** | **285** | **0.2652** | **0.9951** | **285** | **Primary Submission Model** (Nominal minimum CV MSE) |
| Lasso ($\alpha=3\times 10^{-4}$) | 10 | 285 | 0.2628 | 0.9951 | 175 | High performance sparse alternative |
| OLS | 11 | 363 | 2.2323 | 0.9593 | 363 | Onset of severe Runge instability |
| Ridge ($\alpha=1.2$) | 11 | 363 | 0.2841 | 0.9947 | 363 | Controlled boundary shrinkage |
| OLS | 12 | 454 | 7.7129 | 0.8603 | 454 | Ill-conditioned spatial polynomials |
| Ridge ($\alpha=1.8$) | 12 | 454 | 0.2706 | 0.9950 | 454 | Robust regularized asymptote |

---

## 4. Setup and Reproduction

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- `pip` package manager

### Environment Setup
```bash
# Clone the repository
git clone git@github.com:Jdsb06/ML_assignment.git
cd ML_assignment

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Reproducing Predictions and Benchmarks
You can run the scripts either from the repository root or from inside the `code/` directory:

```bash
# 1. End-to-end prediction generation & verification
python3 code/generate_predictions.py

# 2. Or run individual models:
python3 code/train_var1.py --degree 5 --alpha 0.015
python3 code/train_var2.py --degree 10 --alpha 0.7

# 3. Run Exploratory Data Analysis:
python3 code/eda.py

# 4. Regenerate evaluation figures from benchmark CSVs:
python3 code/generate_final_models.py

# 5. Run stratified clamping & importance weighting audit:
python3 code/experiments/clamping_stratified_analysis.py

# 6. Re-run complete systematic benchmark sweep:
python3 code/experiments/run_systematic_benchmark.py
```
