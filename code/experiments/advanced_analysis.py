#!/usr/bin/env python3
"""
Experiment 2: Regularization Analysis for Phase 1 (var1)
Compares OLS, RidgeCV, and LassoCV with feature standardization.
Author: Jashandeep Singh Bedi (IMT2024022)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
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

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    print("=" * 70)
    print("REGULARIZATION EXPLORATION: VAR 1 (10-FOLD CV)")
    print("=" * 70)

    alphas = np.logspace(-4, 4, 30)

    for deg in range(1, 7):
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X1_poly = poly.fit_transform(X1)
        n_features = X1_poly.shape[1]

        # 1. OLS
        ols_cv_mse, ols_cv_r2 = [], []
        for tr, te in kf.split(X1_poly):
            m = LinearRegression().fit(X1_poly[tr], y1[tr])
            p = m.predict(X1_poly[te])
            ols_cv_mse.append(mean_squared_error(y1[te], p))
            ols_cv_r2.append(r2_score(y1[te], p))
        ols_res = f"OLS CV MSE={np.mean(ols_cv_mse):.4f}, R2={np.mean(ols_cv_r2):.4f}"

        # 2. Ridge (StandardScaler + RidgeCV)
        ridge_cv_mse, ridge_cv_r2 = [], []
        best_alphas = []
        for tr, te in kf.split(X1):
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                ('scaler', StandardScaler()),
                ('ridge', RidgeCV(alphas=alphas))
            ])
            pipe.fit(X1[tr], y1[tr])
            p = pipe.predict(X1[te])
            ridge_cv_mse.append(mean_squared_error(y1[te], p))
            ridge_cv_r2.append(r2_score(y1[te], p))
            best_alphas.append(pipe.named_steps['ridge'].alpha_)
        ridge_res = f"Ridge CV MSE={np.mean(ridge_cv_mse):.4f}, R2={np.mean(ridge_cv_r2):.4f} (median α={np.median(best_alphas):.2e})"

        # 3. Lasso (StandardScaler + LassoCV)
        lasso_cv_mse, lasso_cv_r2, n_nonzero = [], [], []
        for tr, te in kf.split(X1):
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                ('scaler', StandardScaler()),
                ('lasso', LassoCV(cv=3, random_state=42, max_iter=5000, alphas=np.logspace(-4, 1, 25)))
            ])
            pipe.fit(X1[tr], y1[tr])
            p = pipe.predict(X1[te])
            lasso_cv_mse.append(mean_squared_error(y1[te], p))
            lasso_cv_r2.append(r2_score(y1[te], p))
            n_nonzero.append(np.sum(pipe.named_steps['lasso'].coef_ != 0))
        lasso_res = f"Lasso CV MSE={np.mean(lasso_cv_mse):.4f}, R2={np.mean(lasso_cv_r2):.4f} (mean non-zero={np.mean(n_nonzero):.1f})"

        print(f"Deg {deg} (terms={n_features:4d}):")
        print(f"   {ols_res}")
        print(f"   {ridge_res}")
        print(f"   {lasso_res}")


if __name__ == '__main__':
    main()
