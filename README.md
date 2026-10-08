# Machine Learning Assignment 1: Polynomial Regression

**Author:** Jashandeep Singh Bedi (Roll Number: `IMT2024022`)  
**Institution:** International Institute of Information Technology, Bangalore (IIIT-B)  
**Course:** Machine Learning (SEM 5)  
**Repository:** [https://github.com/Jdsb06/ML_assignment](https://github.com/Jdsb06/ML_assignment)

---

## 1. Overview

This project implements optimal polynomial regression models to solve two distinct real-world geothermal energy engineering problems using personalized datasets for roll number **`IMT2024022`**:

1. **Phase 1: Power Plant Steam Turbine Optimization (`var1`)**
   - **Goal:** Predict the **Net Power Score ($y$)** of a multi-stage steam turbine from 6 operational valve and turbine percentage deviation parameters ($x_1, \dots, x_6$).
   - **Optimal Architecture:** Degree 5 Polynomial Regression with Feature Standardization and **$L_1$ Lasso Regularization** ($\alpha = 0.007$).
   - **Performance (10-Fold CV):** **$\text{MSE} = 0.32046 \pm 0.04231$**, **$R^2 = 0.96781 \pm 0.00590$**.
   - **Sparsity:** Prunes 461 interaction terms down to 142 active non-zero monomials (~30.8%), mitigating the curse of dimensionality.

2. **Phase 2: Subterranean Thermal Reservoir Mapping (`var2`)**
   - **Goal:** Predict the **Thermal Anomaly Score ($y$)** across a 3D geological survey block from spatial offset coordinates ($x_1, x_2, x_3$).
   - **Optimal Architecture:** Degree 10 Polynomial Regression with Feature Standardization and **$L_2$ Ridge Regularization** ($\alpha = 0.7$).
   - **Performance (10-Fold CV):** **$\text{MSE} = 0.26516 \pm 0.07293$**, **$R^2 = 0.99510 \pm 0.00141$**.
   - **Stability:** Regularizes multicollinear high-degree spatial monomials, dampening Runge oscillations at boundary coordinates.

---

## 2. Structured Directory Layout

The repository is modularly structured into dedicated directories for data, code, predictions, reports, and visualization figures:

```text
ML_assignment/
├── data/                               # Dataset files
│   ├── IMT2024022_train_var1.csv       # Training dataset for Phase 1 (N=1000, 6 features)
│   ├── IMT2024022_test_var1.csv        # Testing dataset for Phase 1 (N=1000)
│   ├── IMT2024022_train_var2.csv       # Training dataset for Phase 2 (N=1000, 3 features)
│   ├── IMT2024022_test_var2.csv        # Testing dataset for Phase 2 (N=1000)
│   └── sample_submission.csv           # Benchmark submission format sample
│
├── code/                               # Core Python implementation scripts
│   ├── train_var1.py                   # Self-contained training & inference script for Phase 1
│   ├── train_var2.py                   # Self-contained training & inference script for Phase 2
│   ├── generate_predictions.py         # End-to-end execution & validation script
│   ├── generate_final_models.py        # Generates figures, predictions, and validation logs
│   └── eda.py                          # Exploratory data analysis script
│
├── predictions/                        # Model prediction outputs
│   ├── IMT2024022_pred_var1.csv        # Final test predictions for Phase 1 (1000 rows, header 'y')
│   └── IMT2024022_pred_var2.csv        # Final test predictions for Phase 2 (1000 rows, header 'y')
│
├── figures/                            # High-resolution evaluation figures (300 DPI)
│   ├── actual_vs_predicted.png         # Out-of-fold actual vs predicted scatter plots
│   ├── model_comparison_degrees.png    # Validation error vs degree curves (OLS vs Ridge vs Lasso)
│   ├── residual_analysis.png           # Gaussian residual distributions
│   └── train_test_distribution.png     # Train vs test prediction distribution alignment
│
├── report/                             # Project report documentation
│   ├── report.tex                      # LaTeX source code for the 5-page report
│   ├── report.pdf                      # Compiled PDF report
│   └── figures/                        # Linked report figures
│
├── IMT2024022/                         # Original roll-number folder (retained for backward compatibility)
│   ├── IMT2024022_train_var1.csv
│   ├── IMT2024022_test_var1.csv
│   ├── IMT2024022_train_var2.csv
│   ├── IMT2024022_test_var2.csv
│   ├── IMT2024022_pred_var1.csv
│   └── IMT2024022_pred_var2.csv
│
├── report.pdf                          # Root copy of the report PDF for easy grading access
├── IMT2024022_pred_var1.csv            # Root copy of Phase 1 predictions for grader convenience
├── IMT2024022_pred_var2.csv            # Root copy of Phase 2 predictions for grader convenience
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # Documentation
```

---

## 3. Results Summary

### Phase 1: Steam Turbine Optimization (`var1`)
| Model / Configuration | Polynomial Degree | Parameters ($p$) | 10-Fold CV MSE | 10-Fold CV $R^2$ | Notes |
|---|:---:|:---:|:---:|:---:|---|
| OLS (Linear) | 1 | 6 | 9.3457 | 0.0760 | Severe underfitting |
| OLS (Quadratic) | 2 | 27 | 2.9550 | 0.7063 | Captures 2nd-order curvature |
| OLS (Cubic) | 3 | 83 | 1.0471 | 0.8958 | Good fit, high bias |
| OLS (Quartic) | 4 | 209 | 0.8437 | 0.9162 | Mild multicollinearity |
| OLS (Quintic) | 5 | 461 | 1.6506 | 0.8414 | Overfitting ($p \approx N/2$) |
| OLS (Sextic) | 6 | 923 | 140.82 | -13.13 | Catastrophic blow-up ($p > N_{\text{tr}}$) |
| Ridge ($\alpha=20.0$) | 5 | 461 | 0.5063 | 0.9491 | Suppresses coefficient variance |
| **Lasso ($\alpha=0.007$)** | **5** | **142 active** | **0.3205** | **0.9678** | **Optimal Model (Sparsity + Low Bias)** |

### Phase 2: Thermal Reservoir Mapping (`var2`)
| Model / Configuration | Polynomial Degree | Parameters ($p$) | 10-Fold CV MSE | 10-Fold CV $R^2$ | Notes |
|---|:---:|:---:|:---:|:---:|---|
| OLS | 2 | 9 | 23.810 | 0.5703 | Underfitting |
| OLS | 4 | 34 | 3.958 | 0.9287 | Rapid error reduction |
| OLS | 6 | 83 | 0.558 | 0.9900 | High fidelity |
| OLS | 8 | 164 | 0.2709 | 0.9950 | Parsimonious baseline |
| OLS | 8 | 164 | 0.2691 | 0.9950 | Mild shrinkage |
| OLS | 10 | 285 | 0.4281 | 0.9923 | Onset of overfitting |
| OLS | 11 | 363 | 2.4747 | 0.9562 | High variance instability |
| OLS | 12 | 454 | 10.412 | 0.8168 | Severe ill-conditioning |
| **Ridge ($\alpha=0.7$)** | **10** | **285** | **0.2652** | **0.9951** | **Optimal Model (Global Minimum MSE)** |
| Lasso ($\alpha=0.0003$) | 10 | 190 active | 0.2626 | 0.9951 | High performance sparse alternative |

---

## 4. Setup and Reproduction

### Prerequisites
- Python 3.10+ (or Python 3.14)
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

### Reproducing Predictions
You can run the scripts either from the repository root or from inside the `code/` directory:

```bash
# End-to-end prediction generation & verification
python3 code/generate_predictions.py

# Or run individual models:
python3 code/train_var1.py --degree 5 --alpha 0.007
python3 code/train_var2.py --degree 10 --alpha 0.7

# Run Exploratory Data Analysis
python3 code/eda.py

# Regenerate evaluation plots and model checkpoints
python3 code/generate_final_models.py
```

---


