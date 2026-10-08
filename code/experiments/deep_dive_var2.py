#!/usr/bin/env python3
"""
Experiment 5: Deep Dive for Phase 2 (var2)
Fine parameter sweeps across degrees 6 to 12 for OLS, Ridge, and Lasso.
Author: Jashandeep Singh Bedi (IMT2024022)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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
    print("PHASE 2 DEEP DIVE: RIDGE & LASSO SENSITIVITY (DEGREES 6 TO 12)")
    print("=" * 70)

    for deg in [7, 8, 9, 10, 11, 12]:
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X2_poly = poly.fit_transform(X2)
        n_feat = X2_poly.shape[1]

        # 1. OLS
        ols_mses = []
        for tr, te in kf.split(X2):
            ols = LinearRegression().fit(X2_poly[tr], y2[tr])
            ols_mses.append(mean_squared_error(y2[te], ols.predict(X2_poly[te])))

        # 2. Ridge fine grid
        best_r_mse = float('inf')
        best_r_alpha = None
        for a in [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            r_mses = []
            for tr, te in kf.split(X2):
                s = StandardScaler()
                X_tr = s.fit_transform(X2_poly[tr])
                X_te = s.transform(X2_poly[te])
                r = Ridge(alpha=a, random_state=42).fit(X_tr, y2[tr])
                r_mses.append(mean_squared_error(y2[te], r.predict(X_te)))
            if np.mean(r_mses) < best_r_mse:
                best_r_mse = np.mean(r_mses)
                best_r_alpha = a

        # 3. Lasso check
        l_mses, nzs = [], []
        for tr, te in kf.split(X2):
            s = StandardScaler()
            X_tr = s.fit_transform(X2_poly[tr])
            X_te = s.transform(X2_poly[te])
            l = Lasso(alpha=0.0003, max_iter=4000, random_state=42).fit(X_tr, y2[tr])
            l_mses.append(mean_squared_error(y2[te], l.predict(X_te)))
            nzs.append(np.sum(l.coef_ != 0))

        print(f"Deg {deg:2d} (terms={n_feat:3d}) | OLS MSE={np.mean(ols_mses):.4f} | Best Ridge MSE={best_r_mse:.4f} (α={best_r_alpha}) | Lasso (α=3e-4) MSE={np.mean(l_mses):.4f} (nz={np.mean(nzs):.1f})")


if __name__ == '__main__':
    main()
