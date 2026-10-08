#!/usr/bin/env python3
"""
Systematic 10-Fold Benchmark across Polynomial Degrees
Generates var1_benchmark.csv and var2_benchmark.csv from scratch using identical folds.
Author: Jashandeep Singh Bedi (IMT2024022)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.name == 'experiments' else SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
ROLL_NO = "IMT2024022"


def get_data(phase):
    if phase == 1:
        path = DATA_DIR / f"{ROLL_NO}_train_var1.csv"
        df = pd.read_csv(path)
        X = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
        y = df['y'].values
    else:
        path = DATA_DIR / f"{ROLL_NO}_train_var2.csv"
        df = pd.read_csv(path)
        X = df[['x1', 'x2', 'x3']].values
        y = df['y'].values
    return X, y


def run_benchmark():
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # =========================================================================
    # PHASE 1 BENCHMARK (Degrees 1 to 6)
    # =========================================================================
    print("=" * 75)
    print("RUNNING PHASE 1 BENCHMARK (10-FOLD CV, D=6, DEGREES 1 TO 6)")
    print("=" * 75)
    X1, y1 = get_data(1)
    
    # Boundary holdout mask
    clamped_counts = np.sum(np.isclose(np.abs(X1), 1.0), axis=1)
    mask_boundary = (clamped_counts >= 3)
    mask_interior = (clamped_counts <= 2)

    var1_rows = []

    # Best alphas for Ridge and Lasso across degrees
    ridge_alphas_v1 = {1: 1.0, 2: 5.0, 3: 10.0, 4: 15.0, 5: 20.0, 6: 30.0}
    lasso_alphas_v1 = {1: 0.001, 2: 0.005, 3: 0.007, 4: 0.007, 5: 0.015, 6: 0.02}

    for deg in range(1, 7):
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        n_terms = poly.fit_transform(X1[:2]).shape[1]

        # 1. OLS
        ols_mses, ols_r2s = [], []
        for tr, te in kf.split(X1):
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                ('ols', LinearRegression())
            ])
            pipe.fit(X1[tr], y1[tr])
            p = pipe.predict(X1[te])
            ols_mses.append(mean_squared_error(y1[te], p))
            ols_r2s.append(r2_score(y1[te], p))
        ols_mse_mean, ols_mse_std = np.mean(ols_mses), np.std(ols_mses)
        ols_r2_mean, ols_r2_std = np.mean(ols_r2s), np.std(ols_r2s)

        # 2. Ridge
        r_alpha = ridge_alphas_v1[deg]
        r_mses, r_r2s = [], []
        for tr, te in kf.split(X1):
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=r_alpha, random_state=42))
            ])
            pipe.fit(X1[tr], y1[tr])
            p = pipe.predict(X1[te])
            r_mses.append(mean_squared_error(y1[te], p))
            r_r2s.append(r2_score(y1[te], p))
        r_mse_mean, r_mse_std = np.mean(r_mses), np.std(r_mses)
        r_r2_mean, r_r2_std = np.mean(r_r2s), np.std(r_r2s)

        # 3. Lasso
        l_alpha = lasso_alphas_v1[deg]
        l_mses, l_r2s, l_nzs = [], [], []
        for tr, te in kf.split(X1):
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                ('scaler', StandardScaler()),
                ('lasso', Lasso(alpha=l_alpha, max_iter=10000, random_state=42))
            ])
            pipe.fit(X1[tr], y1[tr])
            p = pipe.predict(X1[te])
            l_mses.append(mean_squared_error(y1[te], p))
            l_r2s.append(r2_score(y1[te], p))
            l_nzs.append(np.sum(pipe.named_steps['lasso'].coef_ != 0))
        l_mse_mean, l_mse_std = np.mean(l_mses), np.std(l_mses)
        l_r2_mean, l_r2_std = np.mean(l_r2s), np.std(l_r2s)
        l_nz_mean = np.mean(l_nzs)

        # Full fit for Lasso to get full active count
        full_pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=l_alpha, max_iter=10000, random_state=42))
        ])
        full_pipe.fit(X1, y1)
        full_nz = np.sum(full_pipe.named_steps['lasso'].coef_ != 0)

        # Boundary holdout for Lasso
        b_pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=l_alpha, max_iter=10000, random_state=42))
        ])
        b_pipe.fit(X1[mask_interior], y1[mask_interior])
        p_b = b_pipe.predict(X1[mask_boundary])
        b_mse = mean_squared_error(y1[mask_boundary], p_b)

        print(f"Deg {deg} ({n_terms:3d} terms): OLS MSE={ols_mse_mean:.4f} | Ridge (a={r_alpha}) MSE={r_mse_mean:.4f} | Lasso (a={l_alpha}) MSE={l_mse_mean:.4f} (CV nz={l_nz_mean:.1f}, Full nz={full_nz}, Bound MSE={b_mse:.4f})")

        var1_rows.append({
            'degree': deg,
            'terms': n_terms,
            'ols_cv_mse': ols_mse_mean,
            'ols_cv_mse_std': ols_mse_std,
            'ols_cv_r2': ols_r2_mean,
            'ridge_alpha': r_alpha,
            'ridge_cv_mse': r_mse_mean,
            'ridge_cv_mse_std': r_mse_std,
            'ridge_cv_r2': r_r2_mean,
            'lasso_alpha': l_alpha,
            'lasso_cv_mse': l_mse_mean,
            'lasso_cv_mse_std': l_mse_std,
            'lasso_cv_r2': l_r2_mean,
            'lasso_cv_active_mean': l_nz_mean,
            'lasso_full_active': full_nz,
            'lasso_boundary_holdout_mse': b_mse
        })

    var1_df = pd.DataFrame(var1_rows)
    var1_df.to_csv(SCRIPT_DIR / "var1_benchmark.csv", index=False)
    print(f"Saved: {SCRIPT_DIR / 'var1_benchmark.csv'}")

    # =========================================================================
    # PHASE 2 BENCHMARK (Degrees 1 to 12)
    # =========================================================================
    print("\n" + "=" * 75)
    print("RUNNING PHASE 2 BENCHMARK (10-FOLD CV, D=3, DEGREES 1 TO 12)")
    print("=" * 75)
    X2, y2 = get_data(2)

    var2_rows = []
    ridge_alphas_v2 = {
        1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1,
        7: 0.1, 8: 0.05, 9: 0.1, 10: 0.7, 11: 1.2, 12: 1.8
    }
    lasso_alphas_v2 = {
        1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001, 5: 0.0005, 6: 0.0003,
        7: 0.0003, 8: 0.0003, 9: 0.0003, 10: 0.0003, 11: 0.0003, 12: 0.0003
    }

    for deg in range(1, 13):
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        n_terms = poly.fit_transform(X2[:2]).shape[1]

        # 1. OLS
        ols_mses, ols_r2s = [], []
        for tr, te in kf.split(X2):
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                ('ols', LinearRegression())
            ])
            pipe.fit(X2[tr], y2[tr])
            p = pipe.predict(X2[te])
            ols_mses.append(mean_squared_error(y2[te], p))
            ols_r2s.append(r2_score(y2[te], p))
        ols_mse_mean, ols_mse_std = np.mean(ols_mses), np.std(ols_mses)
        ols_r2_mean = np.mean(ols_r2s)

        # 2. Ridge
        r_alpha = ridge_alphas_v2[deg]
        r_mses, r_r2s = [], []
        for tr, te in kf.split(X2):
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=r_alpha, random_state=42))
            ])
            pipe.fit(X2[tr], y2[tr])
            p = pipe.predict(X2[te])
            r_mses.append(mean_squared_error(y2[te], p))
            r_r2s.append(r2_score(y2[te], p))
        r_mse_mean, r_mse_std = np.mean(r_mses), np.std(r_mses)
        r_r2_mean = np.mean(r_r2s)

        # 3. Lasso
        l_alpha = lasso_alphas_v2[deg]
        l_mses, l_r2s, l_nzs = [], [], []
        for tr, te in kf.split(X2):
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                ('scaler', StandardScaler()),
                ('lasso', Lasso(alpha=l_alpha, max_iter=5000, random_state=42))
            ])
            pipe.fit(X2[tr], y2[tr])
            p = pipe.predict(X2[te])
            l_mses.append(mean_squared_error(y2[te], p))
            l_r2s.append(r2_score(y2[te], p))
            l_nzs.append(np.sum(pipe.named_steps['lasso'].coef_ != 0))
        l_mse_mean, l_mse_std = np.mean(l_mses), np.std(l_mses)
        l_r2_mean = np.mean(l_r2s)
        l_nz_mean = np.mean(l_nzs)

        full_pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=l_alpha, max_iter=5000, random_state=42))
        ])
        full_pipe.fit(X2, y2)
        full_nz = np.sum(full_pipe.named_steps['lasso'].coef_ != 0)

        print(f"Deg {deg:2d} ({n_terms:3d} terms): OLS MSE={ols_mse_mean:.4f} | Ridge (a={r_alpha:.2f}) MSE={r_mse_mean:.4f} | Lasso (a={l_alpha:.4f}) MSE={l_mse_mean:.4f} (CV nz={l_nz_mean:.1f}, Full nz={full_nz})")

        var2_rows.append({
            'degree': deg,
            'terms': n_terms,
            'ols_cv_mse': ols_mse_mean,
            'ols_cv_mse_std': ols_mse_std,
            'ols_cv_r2': ols_r2_mean,
            'ridge_alpha': r_alpha,
            'ridge_cv_mse': r_mse_mean,
            'ridge_cv_mse_std': r_mse_std,
            'ridge_cv_r2': r_r2_mean,
            'lasso_alpha': l_alpha,
            'lasso_cv_mse': l_mse_mean,
            'lasso_cv_mse_std': l_mse_std,
            'lasso_cv_r2': l_r2_mean,
            'lasso_cv_active_mean': l_nz_mean,
            'lasso_full_active': full_nz
        })

    var2_df = pd.DataFrame(var2_rows)
    var2_df.to_csv(SCRIPT_DIR / "var2_benchmark.csv", index=False)
    print(f"Saved: {SCRIPT_DIR / 'var2_benchmark.csv'}")


if __name__ == '__main__':
    run_benchmark()
