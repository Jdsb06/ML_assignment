# Machine Learning Assignment 1: Polynomial Regression

**Author:** Jatin Sharma (Roll Number: `IMT2024022`)  
**Institution:** International Institute of Information Technology, Bangalore (IIIT-B)  
**Course:** Machine Learning (SEM 5)  
**Repository:** [https://github.com/Jdsb06/ML_assignment](https://github.com/Jdsb06/ML_assignment)

---

## 1. Overview

This project implements optimal polynomial regression models to solve two distinct real-world geothermal engineering challenges using personalized datasets for roll number **`IMT2024022`**:

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

## 2. Repository Structure

```text
ML_assignment/
├── figures/                            # High-resolution evaluation plots
│   ├── actual_vs_predicted.png         # Out-of-fold actual vs predicted scatter plots
│   ├── model_comparison_degrees.png    # Validation error vs degree curves (OLS vs Ridge vs Lasso)
│   ├── residual_analysis.png           # Gaussian residual distributions
│   └── train_test_distribution.png     # Train vs test prediction distribution alignment
├── IMT2024022/                         # Student dataset folder
│   ├── IMT2024022_train_var1.csv       # Training dataset for Phase 1 (N=1000, 6 features)
│   ├── IMT2024022_test_var1.csv        # Testing dataset for Phase 1 (N=1000)
│   ├── IMT2024022_train_var2.csv       # Training dataset for Phase 2 (N=1000, 3 features)
│   ├── IMT2024022_test_var2.csv        # Testing dataset for Phase 2 (N=1000)
│   ├── IMT2024022_pred_var1.csv        # Final test predictions for Phase 1
│   └── IMT2024022_pred_var2.csv        # Final test predictions for Phase 2
├── IMT2024022_pred_var1.csv            # Root copy of test predictions for Phase 1
├── IMT2024022_pred_var2.csv            # Root copy of test predictions for Phase 2
├── report.pdf                          # Comprehensive 4-page technical report (PDF)
├── report.tex                          # LaTeX source code for the report
├── train_var1.py                       # Self-contained training & inference script for Phase 1
├── train_var2.py                       # Self-contained training & inference script for Phase 2
├── generate_predictions.py             # End-to-end execution & validation script
├── generate_final_models.py            # Generates figures, predictions, and validation logs
├── eda.py                              # Exploratory data analysis script
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Git ignore configuration
└── README.md                           # Project documentation
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
| OLS | 10 | 285 | 0.4281 | 0.9923 | Onset of overfitting |
| OLS | 11 | 363 | 2.4747 | 0.9562 | High variance instability |
| OLS | 12 | 454 | 10.412 | 0.8168 | Severe ill-conditioning |
| **Ridge ($\alpha=0.7$)** | **10** | **285** | **0.2652** | **0.9951** | **Optimal Model (Global Minimum MSE)** |
| Lasso ($\alpha=0.0003$) | 10 | 190 active | 0.2626 | 0.9951 | High performance sparse alternative |

---

## 4. Setup and Reproduction

### Prerequisites
- Python 3.10+ (or Python 3.14 venv)
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
To generate and verify predictions for both Phase 1 and Phase 2:
```bash
python3 generate_predictions.py
```

To run individual phases:
```bash
# Run Phase 1
python3 train_var1.py --degree 5 --alpha 0.007

# Run Phase 2
python3 train_var2.py --degree 10 --alpha 0.7
```

To regenerate all evaluation figures:
```bash
python3 generate_final_models.py
```

---

## 5. Deliverables Verification
- Prediction files adhere strictly to `sample_submission.csv` format (header `y`, 1000 float rows, zero missing values).
- PDF report compiled via LaTeX (`report.pdf`) provides rigorous theoretical justification and empirical analysis.
