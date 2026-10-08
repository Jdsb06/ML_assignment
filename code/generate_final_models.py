#!/usr/bin/env python3
"""
Comprehensive Model Generation, Figures, and Verification Pipeline
Author: Jashandeep Singh Bedi (IMT2024022)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

ROLL_NO = "IMT2024022"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'code' else SCRIPT_DIR
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "report" / "figures").mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})


def resolve_path(relative_candidate):
    candidates = [
        REPO_ROOT / "data" / relative_candidate,
        REPO_ROOT / ROLL_NO / relative_candidate,
        REPO_ROOT / relative_candidate,
        Path(relative_candidate)
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

    # 1. PHASE 1 (var1): Degree 5 Lasso (alpha=0.007)
    print("=" * 65)
    print("PHASE 1: STEAM TURBINE OPTIMIZATION (var1)")
    print("=" * 65)
    oof_pred1 = np.zeros(len(y1_train))
    cv_mse1, cv_r21 = [], []
    for tr, val in kf.split(X1_train):
        m = Pipeline([
            ('poly', PolynomialFeatures(degree=5, include_bias=False)),
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=0.007, max_iter=10000, random_state=42))
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
        ('lasso', Lasso(alpha=0.007, max_iter=10000, random_state=42))
    ])
    model1.fit(X1_train, y1_train)
    test_pred1 = model1.predict(X1_test)

    # 2. PHASE 2 (var2): Degree 10 Ridge (alpha=0.7)
    print("\n" + "=" * 65)
    print("PHASE 2: SUBTERRANEAN THERMAL RESERVOIR (var2)")
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

    # 3. Save Predictions
    for p_df, filename in [(pd.DataFrame({'y': test_pred1}), f"{ROLL_NO}_pred_var1.csv"),
                           (pd.DataFrame({'y': test_pred2}), f"{ROLL_NO}_pred_var2.csv")]:
        for d in [REPO_ROOT, REPO_ROOT / "predictions", REPO_ROOT / "data", REPO_ROOT / ROLL_NO]:
            d.mkdir(parents=True, exist_ok=True)
            p_df.to_csv(d / filename, index=False)

    print("\n" + "=" * 65)
    print("GENERATING VISUALIZATION FIGURES")
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
    axes[0].set_title(f'Phase 1 (var1): Degree 5 Lasso\nCV MSE = {np.mean(cv_mse1):.4f}, $R^2$ = {np.mean(cv_r21):.4f}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Actual Net Power Score ($y$)', fontsize=11)
    axes[0].set_ylabel('Out-of-Fold Predicted ($y$)', fontsize=11)
    axes[0].legend(frameon=True)

    axes[1].scatter(y2_train, oof_pred2, alpha=0.5, color='#2ca02c', edgecolors='none', s=25)
    lims2 = [min(y2_train.min(), oof_pred2.min()), max(y2_train.max(), oof_pred2.max())]
    axes[1].plot(lims2, lims2, 'r--', lw=2, label='Ideal Fit')
    axes[1].set_title(f'Phase 2 (var2): Degree 10 Ridge\nCV MSE = {np.mean(cv_mse2):.4f}, $R^2$ = {np.mean(cv_r22):.4f}', fontsize=12, fontweight='bold')
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

    # Figure 3: Degree Comparison Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    deg1_list = [1, 2, 3, 4, 5, 6]
    ols1_mse = [9.3457, 2.9550, 1.0471, 0.8437, 1.6506, 140.82]
    ridge1_mse = [9.3446, 2.9542, 1.0352, 0.7154, 0.5433, 0.6551]
    lasso1_mse = [9.3491, 2.9586, 1.0042, 0.6066, 0.3205, 0.3549]

    axes[0].plot(deg1_list, ols1_mse, 'o-', color='#d62728', lw=2, label='OLS (Unregularized)')
    axes[0].plot(deg1_list, ridge1_mse, 's-', color='#ff7f0e', lw=2, label='Ridge Regression')
    axes[0].plot(deg1_list, lasso1_mse, '^-', color='#1f77b4', lw=2.5, label='Lasso Regression (Optimal)')
    axes[0].set_yscale('log')
    axes[0].set_title('Phase 1 (var1): Cross-Validation MSE vs Degree', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Polynomial Degree', fontsize=11)
    axes[0].set_ylabel('10-Fold CV MSE (Log Scale)', fontsize=11)
    axes[0].set_xticks(deg1_list)
    axes[0].axvline(5, color='gray', linestyle=':', label='Chosen: Degree 5')
    axes[0].legend(frameon=True)

    deg2_list = list(range(1, 13))
    ols2_mse = [41.20, 23.81, 12.80, 3.96, 1.56, 0.558, 0.372, 0.279, 0.297, 0.428, 2.47, 10.41]
    ridge2_mse = [41.20, 23.81, 12.80, 3.96, 1.56, 0.557, 0.360, 0.269, 0.289, 0.265, 0.291, 0.278]
    lasso2_mse = [41.20, 23.81, 12.80, 3.96, 1.56, 0.558, 0.363, 0.282, 0.286, 0.263, 0.282, 0.287]

    axes[1].plot(deg2_list, ols2_mse, 'o-', color='#d62728', lw=2, label='OLS (Unregularized)')
    axes[1].plot(deg2_list, ridge2_mse, 's-', color='#2ca02c', lw=2.5, label='Ridge Regression (Optimal)')
    axes[1].plot(deg2_list, lasso2_mse, '^-', color='#1f77b4', lw=2, label='Lasso Regression')
    axes[1].set_yscale('log')
    axes[1].set_title('Phase 2 (var2): Cross-Validation MSE vs Degree', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Polynomial Degree', fontsize=11)
    axes[1].set_ylabel('10-Fold CV MSE (Log Scale)', fontsize=11)
    axes[1].set_xticks(deg2_list)
    axes[1].axvline(10, color='gray', linestyle=':', label='Chosen: Degree 10')
    axes[1].legend(frameon=True)
    plt.tight_layout()
    save_dual(fig, 'model_comparison_degrees.png')

    # Figure 4: Train vs Test Target Distributions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    sns.histplot(y1_train, kde=True, ax=axes[0], color='#1f77b4', label='Train Ground Truth', stat='density', bins=30)
    sns.histplot(test_pred1, kde=True, ax=axes[0], color='#ff7f0e', label='Test Predictions', stat='density', bins=30)
    axes[0].set_title('Phase 1: Train vs Test Target Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Net Power Score ($y$)', fontsize=11)
    axes[0].legend(frameon=True)

    sns.histplot(y2_train, kde=True, ax=axes[1], color='#2ca02c', label='Train Ground Truth', stat='density', bins=30)
    sns.histplot(test_pred2, kde=True, ax=axes[1], color='#9467bd', label='Test Predictions', stat='density', bins=30)
    axes[1].set_title('Phase 2: Train vs Test Target Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Thermal Anomaly Score ($y$)', fontsize=11)
    axes[1].legend(frameon=True)
    plt.tight_layout()
    save_dual(fig, 'train_test_distribution.png')

    print("All figures and predictions updated in figures/, report/figures/, and predictions/!")


if __name__ == '__main__':
    main()
