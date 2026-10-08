#!/usr/bin/env python3
"""
Experiment 4: Deep Dive for Phase 1 (var1)
Evaluates Post-Lasso de-biasing, fine alpha sweeps, and boundary holdout testing.
Author: Jashandeep Singh Bedi (IMT2024022)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
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
    feats = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    X1 = train1[feats].values
    y1 = train1['y'].values

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Extreme coordinate clamping check
    clamped_counts = np.sum(np.isclose(np.abs(X1), 1.0), axis=1)
    mask_boundary = (clamped_counts >= 3)
    mask_interior = (clamped_counts <= 2)

    print("=" * 70)
    print("PHASE 1 DEEP DIVE: POST-LASSO & BOUNDARY HOLDOUT SENSITIVITY")
    print(f"Boundary split: {np.sum(mask_interior)} interior train vs {np.sum(mask_boundary)} boundary test")
    print("=" * 70)

    for deg in [4, 5, 6]:
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X1_poly = poly.fit_transform(X1)
        n_feat = X1_poly.shape[1]

        print(f"\n--- Degree {deg} (total terms = {n_feat}) ---")
        for alpha in [0.005, 0.007, 0.010, 0.015, 0.020, 0.030]:
            lasso_mses, post_lasso_mses, nonzero_counts = [], [], []
            for tr, te in kf.split(X1):
                s = StandardScaler()
                X_tr = s.fit_transform(X1_poly[tr])
                X_te = s.transform(X1_poly[te])

                las = Lasso(alpha=alpha, max_iter=5000, random_state=42)
                las.fit(X_tr, y1[tr])
                p_las = las.predict(X_te)
                lasso_mses.append(mean_squared_error(y1[te], p_las))

                mask = (las.coef_ != 0)
                nonzero_counts.append(np.sum(mask))
                if 0 < np.sum(mask) < len(tr):
                    ols = LinearRegression()
                    ols.fit(X1_poly[tr][:, mask], y1[tr])
                    p_post = ols.predict(X1_poly[te][:, mask])
                    post_lasso_mses.append(mean_squared_error(y1[te], p_post))

            # Boundary holdout
            s_b = StandardScaler()
            X_tr_b = s_b.fit_transform(X1_poly[mask_interior])
            X_te_b = s_b.transform(X1_poly[mask_boundary])
            las_b = Lasso(alpha=alpha, max_iter=5000, random_state=42).fit(X_tr_b, y1[mask_interior])
            bound_mse = mean_squared_error(y1[mask_boundary], las_b.predict(X_te_b))

            print(f"alpha={alpha:.3f} | CV Lasso MSE={np.mean(lasso_mses):.4f} | Post-Lasso MSE={np.mean(post_lasso_mses):.4f} | nz={np.mean(nonzero_counts):.1f} | Boundary Holdout MSE={bound_mse:.4f}")


if __name__ == '__main__':
    main()
