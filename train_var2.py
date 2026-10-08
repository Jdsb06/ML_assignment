#!/usr/bin/env python3
"""
Phase 2: Subterranean Thermal Reservoir Mapping (var2)
Polynomial Regression Model Training and Inference Pipeline
Author: IMT2024022
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

ROLL_NO = "IMT2024022"
DEFAULT_TRAIN_PATH = f"{ROLL_NO}/{ROLL_NO}_train_var2.csv"
DEFAULT_TEST_PATH = f"{ROLL_NO}/{ROLL_NO}_test_var2.csv"
DEFAULT_OUT_PATH = f"{ROLL_NO}_pred_var2.csv"


def train_and_evaluate(train_path, test_path, out_path, degree=10, alpha=0.7, n_splits=10):
    print("=" * 70)
    print("PHASE 2: SUBTERRANEAN THERMAL RESERVOIR MAPPING (var2)")
    print("=" * 70)
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

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
    pred_df.to_csv(out_path, index=False)

    # Also save to roll number subdirectory for completeness
    roll_dir = os.path.dirname(train_path)
    if os.path.exists(roll_dir):
        pred_df.to_csv(os.path.join(roll_dir, os.path.basename(out_path)), index=False)

    print(f"Predictions saved successfully to: {out_path}")
    print(f"Prediction count: {len(pred_df)} rows")
    print(f"Prediction statistics: Mean={test_preds.mean():.4f}, Std={test_preds.std():.4f}, Min={test_preds.min():.4f}, Max={test_preds.max():.4f}")
    print("=" * 70)
    return final_pipeline, mean_mse, mean_r2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Polynomial Regression for Phase 2 (var2)")
    parser.add_argument('--train', type=str, default=DEFAULT_TRAIN_PATH, help="Path to training CSV")
    parser.add_argument('--test', type=str, default=DEFAULT_TEST_PATH, help="Path to testing CSV")
    parser.add_argument('--out', type=str, default=DEFAULT_OUT_PATH, help="Output prediction CSV path")
    parser.add_argument('--degree', type=int, default=10, help="Polynomial degree")
    parser.add_argument('--alpha', type=float, default=0.7, help="L2 Ridge regularization parameter")
    args = parser.parse_args()

    train_and_evaluate(args.train, args.test, args.out, degree=args.degree, alpha=args.alpha)
