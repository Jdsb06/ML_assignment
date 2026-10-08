#!/usr/bin/env python3
"""
Experiment 3: Higher-Degree Exploration for Phase 2 (var2)
Evaluates degrees 4 to 12 with OLS, RidgeCV, and LassoCV.
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
    train2 = pd.read_csv(resolve_path(f"{ROLL_NO}_train_var2.csv"))
    X2 = train2[['x1', 'x2', 'x3']].values
    y2 = train2['y'].values

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    print("=" * 70)
    print("HIGHER-DEGREE EXPLORATION: VAR 2 (10-FOLD CV)")
    print("=" * 70)

    alphas = np.logspace(-4, 3, 20)

    for deg in range(4, 13):
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X2_poly = poly.fit_transform(X2)
        n_features = X2_poly.shape[1]

        # 1. Plain OLS
        cv_mse, cv_r2 = [], []
        for tr, te in kf.split(X2_poly):
            m = LinearRegression().fit(X2_poly[tr], y2[tr])
            p = m.predict(X2_poly[te])
            cv_mse.append(mean_squared_error(y2[te], p))
            cv_r2.append(r2_score(y2[te], p))
        ols_res = f"OLS CV MSE={np.mean(cv_mse):.4f}, R2={np.mean(cv_r2):.4f}"

        # 2. Ridge (StandardScaler + RidgeCV)
        ridge_cv_mse, ridge_cv_r2, best_alphas = [], [], []
        for tr, te in kf.split(X2):
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                ('scaler', StandardScaler()),
                ('ridge', RidgeCV(alphas=alphas))
            ])
            pipe.fit(X2[tr], y2[tr])
            p = pipe.predict(X2[te])
            ridge_cv_mse.append(mean_squared_error(y2[te], p))
            ridge_cv_r2.append(r2_score(y2[te], p))
            best_alphas.append(pipe.named_steps['ridge'].alpha_)
        ridge_res = f"Ridge CV MSE={np.mean(ridge_cv_mse):.4f}, R2={np.mean(ridge_cv_r2):.4f} (median α={np.median(best_alphas):.2e})"

        # 3. Lasso (StandardScaler + LassoCV)
        lasso_cv_mse, lasso_cv_r2, n_nonzero = [], [], []
        for tr, te in kf.split(X2):
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                ('scaler', StandardScaler()),
                ('lasso', LassoCV(cv=3, random_state=42, max_iter=3000, alphas=np.logspace(-4, 1, 20)))
            ])
            pipe.fit(X2[tr], y2[tr])
            p = pipe.predict(X2[te])
            lasso_cv_mse.append(mean_squared_error(y2[te], p))
            lasso_cv_r2.append(r2_score(y2[te], p))
            n_nonzero.append(np.sum(pipe.named_steps['lasso'].coef_ != 0))
        lasso_res = f"Lasso CV MSE={np.mean(lasso_cv_mse):.4f}, R2={np.mean(lasso_cv_r2):.4f} (mean non-zero={np.mean(n_nonzero):.1f})"

        print(f"Deg {deg:2d} (terms={n_features:4d}):")
        print(f"   {ols_res}")
        print(f"   {ridge_res}")
        print(f"   {lasso_res}")


if __name__ == '__main__':
    main()
