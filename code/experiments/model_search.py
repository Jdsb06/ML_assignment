#!/usr/bin/env python3
"""
Experiment 1: Initial Polynomial Degree Search (OLS)
Evaluates unregularized OLS across degrees 1 to 6 for var1 and 1 to 15 for var2.
Author: Jashandeep Singh Bedi (IMT2024022)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.name == 'experiments' else SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
ROLL_NO = "IMT2024022"


def resolve_path(filename):
    for c in [DATA_DIR / filename, REPO_ROOT / ROLL_NO / filename, REPO_ROOT / filename]:
        if c.exists():
            return c
    return DATA_DIR / filename


def main():
    train1 = pd.read_csv(resolve_path(f"{ROLL_NO}_train_var1.csv"))
    X1 = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
    y1 = train1['y'].values

    print("=" * 65)
    print("EXPLORING VAR 1 (6 features) - UNREGULARIZED OLS (10-FOLD CV)")
    print("=" * 65)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for deg in range(1, 7):
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X1_poly = poly.fit_transform(X1)
        n_features = X1_poly.shape[1]

        lr = LinearRegression()
        lr.fit(X1_poly, y1)
        train_pred = lr.predict(X1_poly)
        train_mse = mean_squared_error(y1, train_pred)
        train_r2 = r2_score(y1, train_pred)

        cv_mse_scores, cv_r2_scores = [], []
        for train_idx, val_idx in kf.split(X1_poly):
            lr.fit(X1_poly[train_idx], y1[train_idx])
            val_pred = lr.predict(X1_poly[val_idx])
            cv_mse_scores.append(mean_squared_error(y1[val_idx], val_pred))
            cv_r2_scores.append(r2_score(y1[val_idx], val_pred))

        print(f"Deg {deg}: terms={n_features:4d} | Train MSE={train_mse:.4f}, R2={train_r2:.4f} | CV MSE={np.mean(cv_mse_scores):.4f} (std={np.std(cv_mse_scores):.4f}), CV R2={np.mean(cv_r2_scores):.4f}")

    print("\n" + "=" * 65)
    print("EXPLORING VAR 2 (3 features) - UNREGULARIZED OLS (10-FOLD CV)")
    print("=" * 65)

    train2 = pd.read_csv(resolve_path(f"{ROLL_NO}_train_var2.csv"))
    X2 = train2[['x1', 'x2', 'x3']].values
    y2 = train2['y'].values

    for deg in range(1, 13):
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X2_poly = poly.fit_transform(X2)
        n_features = X2_poly.shape[1]

        lr = LinearRegression()
        lr.fit(X2_poly, y2)
        train_pred = lr.predict(X2_poly)
        train_mse = mean_squared_error(y2, train_pred)
        train_r2 = r2_score(y2, train_pred)

        cv_mse_scores, cv_r2_scores = [], []
        for train_idx, val_idx in kf.split(X2_poly):
            lr.fit(X2_poly[train_idx], y2[train_idx])
            val_pred = lr.predict(X2_poly[val_idx])
            cv_mse_scores.append(mean_squared_error(y2[val_idx], val_pred))
            cv_r2_scores.append(r2_score(y2[val_idx], val_pred))

        print(f"Deg {deg:2d}: terms={n_features:4d} | Train MSE={train_mse:.4f}, R2={train_r2:.4f} | CV MSE={np.mean(cv_mse_scores):.4f}, CV R2={np.mean(cv_r2_scores):.4f}")


if __name__ == '__main__':
    main()
