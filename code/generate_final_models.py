#!/usr/bin/env python3
"""
Comprehensive Model Generation, Figures, and Verification Pipeline
Reads benchmark results directly from code/experiments/*.csv to ensure 100% data consistency.
Author: Jashandeep Singh Bedi (IMT2024022)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

ROLL_NO = "IMT2024022"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'code' else SCRIPT_DIR
FIG_DIR = REPO_ROOT / "figures"
EXP_DIR = SCRIPT_DIR / "experiments"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "report" / "figures").mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})


def resolve_path(filename):
    candidates = [
        REPO_ROOT / "data" / filename,
        REPO_ROOT / ROLL_NO / filename,
        REPO_ROOT / filename,
        Path(filename)
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def main():
    train1 = pd.read_csv(resolve_path(f"{ROLL_NO}_train_var1.csv"))
    test1 = pd.read_csv(resolve_path(f"{ROLL_NO}_test_var1.csv"))
    train2 = pd.read_csv(resolve_path(f"{ROLL_NO}_train_var2.csv"))
    test2 = pd.read_csv(resolve_path(f"{ROLL_NO}_test_var2.csv"))

    X1_train = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
    y1_train = train1['y'].values
    X1_test = test1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values

    X2_train = train2[['x1', 'x2', 'x3']].values
    y2_train = train2['y'].values
    X2_test = test2[['x1', 'x2', 'x3']].values

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # -------------------------------------------------------------------------
    # 1. PHASE 1: VAR 1 MODEL (Degree 5 Lasso, alpha=0.015)
    # -------------------------------------------------------------------------
    # alpha=0.015 selected based on extreme-boundary holdout cross-validation
    print("=" * 65)
    print("PHASE 1: STEAM TURBINE OPTIMIZATION (var1, Degree 5 Lasso alpha=0.015)")
    print("=" * 65)
    oof_pred1 = np.zeros(len(y1_train))
    cv_mse1, cv_r21 = [], []
    for tr, val in kf.split(X1_train):
        m = Pipeline([
            ('poly', PolynomialFeatures(degree=5, include_bias=False)),
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=0.015, max_iter=10000, random_state=42))
        ])
        m.fit(X1_train[tr], y1_train[tr])
        p = m.predict(X1_train[val])
        oof_pred1[val] = p
        cv_mse1.append(mean_squared_error(y1_train[val], p))
        cv_r21.append(r2_score(y1_train[val], p))

    print(f"Var1 10-Fold CV MSE: {np.mean(cv_mse1):.5f} ± {np.std(cv_mse1):.5f}")
    print(f"Var1 10-Fold CV R2:  {np.mean(cv_r21):.5f} ± {np.std(cv_r21):.5f}")

    model1 = Pipeline([
        ('poly', PolynomialFeatures(degree=5, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lasso', Lasso(alpha=0.015, max_iter=10000, random_state=42))
    ])
    model1.fit(X1_train, y1_train)
    test_pred1 = model1.predict(X1_test)
    non_zeros_v1 = np.sum(model1.named_steps['lasso'].coef_ != 0)
    print(f"Var1 Full Train Sparsity: {non_zeros_v1}/461 active terms")

    # -------------------------------------------------------------------------
    # 2. PHASE 2: VAR 2 MODEL (Degree 10 Ridge, alpha=0.7)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 2: SUBTERRANEAN THERMAL RESERVOIR (var2, Degree 10 Ridge alpha=0.7)")
    print("=" * 65)
    oof_pred2 = np.zeros(len(y2_train))
    cv_mse2, cv_r22 = [], []
    for tr, val in kf.split(X2_train):
        m = Pipeline([
            ('poly', PolynomialFeatures(degree=10, include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=0.7, random_state=42))
        ])
        m.fit(X2_train[tr], y2_train[tr])
        p = m.predict(X2_train[val])
        oof_pred2[val] = p
        cv_mse2.append(mean_squared_error(y2_train[val], p))
        cv_r22.append(r2_score(y2_train[val], p))

    print(f"Var2 10-Fold CV MSE: {np.mean(cv_mse2):.5f} ± {np.std(cv_mse2):.5f}")
    print(f"Var2 10-Fold CV R2:  {np.mean(cv_r22):.5f} ± {np.std(cv_r22):.5f}")

    model2 = Pipeline([
        ('poly', PolynomialFeatures(degree=10, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=0.7, random_state=42))
    ])
    model2.fit(X2_train, y2_train)
    test_pred2 = model2.predict(X2_test)

    # -------------------------------------------------------------------------
    # 3. SAVE PREDICTION FILES
    # -------------------------------------------------------------------------
    df1 = pd.DataFrame({'y': test_pred1})
    df2 = pd.DataFrame({'y': test_pred2})
    
    # Save canonical copies in predictions/ and root
    df1.to_csv(REPO_ROOT / "predictions" / f"{ROLL_NO}_pred_var1.csv", index=False)
    df2.to_csv(REPO_ROOT / "predictions" / f"{ROLL_NO}_pred_var2.csv", index=False)
    df1.to_csv(REPO_ROOT / f"{ROLL_NO}_pred_var1.csv", index=False)
    df2.to_csv(REPO_ROOT / f"{ROLL_NO}_pred_var2.csv", index=False)

    print("\n" + "=" * 65)
    print("GENERATING VISUALIZATION FIGURES FROM CODE/EXPERIMENT CSVs")
    print("=" * 65)

    def save_dual(fig, name):
        fig.savefig(FIG_DIR / name, dpi=300)
        fig.savefig(REPO_ROOT / "report" / "figures" / name, dpi=300)
        plt.close(fig)

    # Figure 1: Actual vs Out-of-fold predicted
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(y1_train, oof_pred1, alpha=0.5, color='#1f77b4', edgecolors='none', s=25)
    lims1 = [min(y1_train.min(), oof_pred1.min()), max(y1_train.max(), oof_pred1.max())]
    axes[0].plot(lims1, lims1, 'r--', lw=2, label='Ideal Fit')
    axes[0].set_title(f'Phase 1 (var1): Degree 5 Lasso ($\\alpha=0.015$)\n10-Fold CV MSE = {np.mean(cv_mse1):.4f}, $R^2$ = {np.mean(cv_r21):.4f}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Actual Net Power Score ($y$)', fontsize=11)
    axes[0].set_ylabel('Out-of-Fold Predicted ($y$)', fontsize=11)
    axes[0].legend(frameon=True)

    axes[1].scatter(y2_train, oof_pred2, alpha=0.5, color='#2ca02c', edgecolors='none', s=25)
    lims2 = [min(y2_train.min(), oof_pred2.min()), max(y2_train.max(), oof_pred2.max())]
    axes[1].plot(lims2, lims2, 'r--', lw=2, label='Ideal Fit')
    axes[1].set_title(f'Phase 2 (var2): Degree 10 Ridge ($\\alpha=0.7$)\n10-Fold CV MSE = {np.mean(cv_mse2):.4f}, $R^2$ = {np.mean(cv_r22):.4f}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Actual Thermal Anomaly Score ($y$)', fontsize=11)
    axes[1].set_ylabel('Out-of-Fold Predicted ($y$)', fontsize=11)
    axes[1].legend(frameon=True)
    plt.tight_layout()
    save_dual(fig, 'actual_vs_predicted.png')

    # Figure 2: Residual distributions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    res1 = y1_train - oof_pred1
    res2 = y2_train - oof_pred2
    sns.histplot(res1, kde=True, ax=axes[0], color='#1f77b4', bins=30)
    axes[0].set_title(fr'Phase 1 Residuals ($\mu$={np.mean(res1):.3f}, $\sigma$={np.std(res1):.3f})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(r'Residual ($y - \hat{y}$)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)

    sns.histplot(res2, kde=True, ax=axes[1], color='#2ca02c', bins=30)
    axes[1].set_title(fr'Phase 2 Residuals ($\mu$={np.mean(res2):.3f}, $\sigma$={np.std(res2):.3f})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(r'Residual ($y - \hat{y}$)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    plt.tight_layout()
    save_dual(fig, 'residual_analysis.png')

    # Figure 3: Degree Comparison Curves LOADED DIRECTLY FROM CSV
    var1_bench = pd.read_csv(EXP_DIR / "var1_benchmark.csv")
    var2_bench = pd.read_csv(EXP_DIR / "var2_benchmark.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(var1_bench['degree'], var1_bench['ols_cv_mse'], 'o-', color='#d62728', lw=2, label='OLS (Unregularized)')
    axes[0].plot(var1_bench['degree'], var1_bench['ridge_cv_mse'], 's-', color='#ff7f0e', lw=2, label='Ridge Regression')
    axes[0].plot(var1_bench['degree'], var1_bench['lasso_cv_mse'], '^-', color='#1f77b4', lw=2.5, label='Lasso Regression (Chosen)')
    axes[0].set_yscale('log')
    axes[0].set_title('Phase 1 (var1): 10-Fold CV MSE vs Degree', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Polynomial Degree', fontsize=11)
    axes[0].set_ylabel('10-Fold CV MSE (Log Scale)', fontsize=11)
    axes[0].set_xticks(var1_bench['degree'])
    axes[0].axvline(5, color='gray', linestyle=':', label='Chosen: Degree 5')
    axes[0].legend(frameon=True)

    axes[1].plot(var2_bench['degree'], var2_bench['ols_cv_mse'], 'o-', color='#d62728', lw=2, label='OLS (Unregularized)')
    axes[1].plot(var2_bench['degree'], var2_bench['ridge_cv_mse'], 's-', color='#2ca02c', lw=2.5, label='Ridge Regression (Chosen)')
    axes[1].plot(var2_bench['degree'], var2_bench['lasso_cv_mse'], '^-', color='#1f77b4', lw=2, label='Lasso Regression')
    axes[1].set_yscale('log')
    axes[1].set_title('Phase 2 (var2): 10-Fold CV MSE vs Degree', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Polynomial Degree', fontsize=11)
    axes[1].set_ylabel('10-Fold CV MSE (Log Scale)', fontsize=11)
    axes[1].set_xticks(var2_bench['degree'])
    axes[1].axvline(10, color='gray', linestyle=':', label='Candidate: Degree 10')
    axes[1].legend(frameon=True)
    plt.tight_layout()
    save_dual(fig, 'model_comparison_degrees.png')

    # Figure 4: Train vs Test Target Distributions (Reflecting Covariate Boundary Clamping)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    sns.histplot(y1_train, kde=True, ax=axes[0], color='#1f77b4', label='Train Ground Truth', stat='density', bins=30)
    sns.histplot(test_pred1, kde=True, ax=axes[0], color='#ff7f0e', label='Test Predictions (alpha=0.015)', stat='density', bins=30)
    axes[0].set_title('Phase 1: Target Density (51% Clamped Test vs 31% Train)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Net Power Score ($y$)', fontsize=11)
    axes[0].legend(frameon=True)

    sns.histplot(y2_train, kde=True, ax=axes[1], color='#2ca02c', label='Train Ground Truth', stat='density', bins=30)
    sns.histplot(test_pred2, kde=True, ax=axes[1], color='#9467bd', label='Test Predictions', stat='density', bins=30)
    axes[1].set_title('Phase 2: Target Density (Degree 10 Ridge)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Thermal Anomaly Score ($y$)', fontsize=11)
    axes[1].legend(frameon=True)
    plt.tight_layout()
    save_dual(fig, 'train_test_distribution.png')

    print("All figures successfully regenerated from code/experiments CSVs!")


if __name__ == '__main__':
    main()
