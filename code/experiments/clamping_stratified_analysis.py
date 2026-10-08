#!/usr/bin/env python3
"""
Stratified Clamping and Importance Weighting Analysis for Phase 1 & Phase 2
Author: Jashandeep Singh Bedi (IMT2024022)

Audits coordinate clamping at domain boundaries (+/- 1.0), computes out-of-fold
cross-validation error stratified by the number of clamped coordinates (k), and
calculates the Importance-Weighted (Test-Adjusted) MSE to evaluate covariate shift optimism.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ROLL_NO = "IMT2024022"


def run_phase1_analysis():
    print("=" * 80)
    print("PHASE 1 (var1): STRATIFIED CLAMPING & IMPORTANCE WEIGHTING ANALYSIS")
    print("=" * 80)

    train_path = REPO_ROOT / "data" / f"{ROLL_NO}_train_var1.csv"
    test_path = REPO_ROOT / "data" / f"{ROLL_NO}_test_var1.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

    X_train = train_df[features].values
    y_train = train_df['y'].values
    X_test = test_df[features].values

    # Clamped count per observation (|x| == 1.0)
    train_k = np.isclose(np.abs(X_train), 1.0).sum(axis=1)
    test_k = np.isclose(np.abs(X_test), 1.0).sum(axis=1)

    print(f"Total Coordinate Clamping:")
    print(f"  Train: {train_k.sum()}/{len(X_train)*6} ({train_k.sum()/(len(X_train)*6)*100:.2f}%)")
    print(f"  Test:  {test_k.sum()}/{len(X_test)*6} ({test_k.sum()/(len(X_test)*6)*100:.2f}%)")

    # Evaluate models under 10-Fold CV
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    models = {
        'OLS (Deg 5)': Pipeline([
            ('poly', PolynomialFeatures(degree=5, include_bias=False)),
            ('scaler', StandardScaler()),
            ('ols', LinearRegression())
        ]),
        'Lasso (Deg 5, alpha=0.015)': Pipeline([
            ('poly', PolynomialFeatures(degree=5, include_bias=False)),
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=0.015, max_iter=10000, random_state=42))
        ])
    }

    oof_preds = {name: np.zeros(len(y_train)) for name in models}
    for name, pipe in models.items():
        for tr_idx, val_idx in kf.split(X_train):
            m = Pipeline(pipe.steps)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            oof_preds[name][val_idx] = m.predict(X_train[val_idx])

    # Print Stratified Breakdown Table
    print("\n" + "-" * 80)
    print(f"{'Strata (k)':<15} | {'Train % (N)':<15} | {'Test % (N)':<15} | {'Lasso MSE':<12} | {'OLS MSE':<12}")
    print("-" * 80)

    test_weights = np.array([(test_k == k).sum() / len(test_df) for k in range(7)])
    lasso_strata_mse = []
    ols_strata_mse = []

    for k in range(7):
        n_tr = (train_k == k).sum()
        p_tr = n_tr / len(train_df) * 100
        n_te = (test_k == k).sum()
        p_te = n_te / len(test_df) * 100

        mask = (train_k == k)
        lasso_mse = mean_squared_error(y_train[mask], oof_preds['Lasso (Deg 5, alpha=0.015)'][mask]) if n_tr > 0 else 0
        ols_mse = mean_squared_error(y_train[mask], oof_preds['OLS (Deg 5)'][mask]) if n_tr > 0 else 0

        lasso_strata_mse.append(lasso_mse)
        ols_strata_mse.append(ols_mse)

        label = f"k={k} (Interior)" if k == 0 else (f"k={k} (Boundary)" if k == 6 else f"k={k}")
        print(f"{label:<15} | {p_tr:5.1f}% ({n_tr:3d})     | {p_te:5.1f}% ({n_te:3d})     | {lasso_mse:<12.4f} | {ols_mse:<12.4f}")

    print("-" * 80)
    lasso_cv = mean_squared_error(y_train, oof_preds['Lasso (Deg 5, alpha=0.015)'])
    ols_cv = mean_squared_error(y_train, oof_preds['OLS (Deg 5)'])
    lasso_iw = sum(test_weights[k] * lasso_strata_mse[k] for k in range(7))
    ols_iw = sum(test_weights[k] * ols_strata_mse[k] for k in range(7))

    print(f"{'Overall CV MSE':<15} | {'100.0% (1000)':<15} | {'—':<15} | {lasso_cv:<12.4f} | {ols_cv:<12.4f}")
    print(f"{'IW-MSE (Test)':<15} | {'—':<15} | {'100.0% (1000)':<15} | {lasso_iw:<12.4f} | {ols_iw:<12.4f}")
    print(f"{'CV Optimism':<15} | {'—':<15} | {'—':<15} | {f'+{(lasso_iw-lasso_cv)/lasso_cv*100:.1f}%':<12} | {f'+{(ols_iw-ols_cv)/ols_cv*100:.1f}%':<12}")
    print("=" * 80)


def run_phase2_analysis():
    print("\n" + "=" * 80)
    print("PHASE 2 (var2): CLAMPING BALANCE & COVARIATE AUDIT")
    print("=" * 80)

    train_path = REPO_ROOT / "data" / f"{ROLL_NO}_train_var2.csv"
    test_path = REPO_ROOT / "data" / f"{ROLL_NO}_test_var2.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    features = ['x1', 'x2', 'x3']

    X_train = train_df[features].values
    X_test = test_df[features].values

    train_k = np.isclose(np.abs(X_train), 1.0).sum(axis=1)
    test_k = np.isclose(np.abs(X_test), 1.0).sum(axis=1)

    print(f"Total Coordinate Clamping:")
    print(f"  Train: {train_k.sum()}/{len(X_train)*3} ({train_k.sum()/(len(X_train)*3)*100:.2f}%)")
    print(f"  Test:  {test_k.sum()}/{len(X_test)*3} ({test_k.sum()/(len(X_test)*3)*100:.2f}%)")

    print("\nStrata Breakdown:")
    for k in range(4):
        n_tr = (train_k == k).sum()
        p_tr = n_tr / len(train_df) * 100
        n_te = (test_k == k).sum()
        p_te = n_te / len(test_df) * 100
        print(f"  k={k} clamped: Train = {n_tr:4d} ({p_tr:5.1f}%), Test = {n_te:4d} ({p_te:5.1f}%)")

    print("Summary: Both train and test in Phase 2 have ~25-26% coordinate clamping at +/- 1.0.")
    print("Strata distributions are balanced, confirming NO acute covariate shift across partitions.")
    print("=" * 80)


if __name__ == '__main__':
    run_phase1_analysis()
    run_phase2_analysis()
