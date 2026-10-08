#!/usr/bin/env python3
"""
Phase 2: Subterranean Thermal Reservoir Mapping (var2)
Polynomial Regression Model Training and Inference Pipeline
Author: Jashandeep Singh Bedi (IMT2024022)
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

ROLL_NO = "IMT2024022"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'code' else SCRIPT_DIR


def resolve_path(relative_candidate):
    """Finds existing file checking multiple standard paths."""
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


def train_and_evaluate(train_path=None, test_path=None, out_path=None, degree=10, alpha=0.7, n_splits=10):
    train_file = Path(train_path) if train_path else resolve_path(f"{ROLL_NO}_train_var2.csv")
    test_file = Path(test_path) if test_path else resolve_path(f"{ROLL_NO}_test_var2.csv")
    out_file = Path(out_path) if out_path else REPO_ROOT / "predictions" / f"{ROLL_NO}_pred_var2.csv"

    print("=" * 70)
    print("PHASE 2: SUBTERRANEAN THERMAL RESERVOIR MAPPING (var2)")
    print("=" * 70)
    print(f"Loading training data from: {train_file}")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    features = ['x1', 'x2', 'x3']
    X_train = train_df[features].values
    y_train = train_df['y'].values
    X_test = test_df[features].values

    print(f"Dataset summary: Train N={len(X_train)}, Test N={len(X_test)}, D={len(features)}")
    print(f"Configuring Pipeline: PolynomialFeatures(degree={degree}) -> StandardScaler() -> Ridge(alpha={alpha})")

    # K-Fold Cross Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_mse = []
    cv_r2 = []
    oof_predictions = np.zeros(len(y_train))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        fold_pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha, random_state=42))
        ])
        fold_pipeline.fit(X_train[train_idx], y_train[train_idx])
        val_preds = fold_pipeline.predict(X_train[val_idx])
        oof_predictions[val_idx] = val_preds

        fold_mse = mean_squared_error(y_train[val_idx], val_preds)
        fold_r2 = r2_score(y_train[val_idx], val_preds)
        cv_mse.append(fold_mse)
        cv_r2.append(fold_r2)

    mean_mse = np.mean(cv_mse)
    std_mse = np.std(cv_mse)
    mean_r2 = np.mean(cv_r2)
    std_r2 = np.std(cv_r2)

    print("-" * 70)
    print(f"10-Fold Cross-Validation Results:")
    print(f"  MSE: {mean_mse:.5f} ± {std_mse:.5f}")
    print(f"  R2:  {mean_r2:.5f} ± {std_r2:.5f}")
    print("-" * 70)

    # Train final model on entire training dataset
    final_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha, random_state=42))
    ])
    final_pipeline.fit(X_train, y_train)

    train_preds = final_pipeline.predict(X_train)
    train_mse = mean_squared_error(y_train, train_preds)
    train_r2 = r2_score(y_train, train_preds)

    total_poly_terms = len(final_pipeline.named_steps['ridge'].coef_)

    print(f"Full Dataset Fit:")
    print(f"  Train MSE: {train_mse:.5f}")
    print(f"  Train R2:  {train_r2:.5f}")
    print(f"  Features:  {total_poly_terms} regularized polynomial terms")

    # Generate Test Predictions
    test_preds = final_pipeline.predict(X_test)
    pred_df = pd.DataFrame({'y': test_preds})

    target_paths = [
        out_file,
        REPO_ROOT / f"{ROLL_NO}_pred_var2.csv",
        REPO_ROOT / "predictions" / f"{ROLL_NO}_pred_var2.csv",
        REPO_ROOT / "data" / f"{ROLL_NO}_pred_var2.csv",
        REPO_ROOT / ROLL_NO / f"{ROLL_NO}_pred_var2.csv"
    ]
    for p in target_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(p, index=False)

    print(f"Predictions saved to: {out_file} (and root / predictions mirrors)")
    print(f"Prediction count: {len(pred_df)} rows")
    print(f"Prediction stats: Mean={test_preds.mean():.4f}, Std={test_preds.std():.4f}, Min={test_preds.min():.4f}, Max={test_preds.max():.4f}")
    print("=" * 70)
    return final_pipeline, mean_mse, mean_r2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Polynomial Regression for Phase 2 (var2)")
    parser.add_argument('--train', type=str, default=None, help="Path to training CSV")
    parser.add_argument('--test', type=str, default=None, help="Path to testing CSV")
    parser.add_argument('--out', type=str, default=None, help="Output prediction CSV path")
    parser.add_argument('--degree', type=int, default=10, help="Polynomial degree")
    parser.add_argument('--alpha', type=float, default=0.7, help="L2 Ridge regularization parameter")
    args = parser.parse_args()

    train_and_evaluate(args.train, args.test, args.out, degree=args.degree, alpha=args.alpha)
