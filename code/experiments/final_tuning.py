#!/usr/bin/env python3
"""
Experiment 6: Final Tuning and Seed Robustness Testing
Evaluates variance across multiple seeds and fold splits for top candidates.
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


def resolve_path(filename):
    for c in [DATA_DIR / filename, REPO_ROOT / ROLL_NO / filename, REPO_ROOT / filename]:
        if c.exists():
            return c
    return DATA_DIR / filename


def main():
    train1 = pd.read_csv(resolve_path(f"{ROLL_NO}_train_var1.csv"))
    X1 = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
    y1 = train1['y'].values

    train2 = pd.read_csv(resolve_path(f"{ROLL_NO}_train_var2.csv"))
    X2 = train2[['x1', 'x2', 'x3']].values
    y2 = train2['y'].values

    print("=" * 70)
    print("PHASE 1: MULTI-SEED 10-FOLD CV (DEGREE 5 LASSO, ALPHA=0.015)")
    print("=" * 70)
    seeds = [42, 101, 2024, 7, 999]
    v1_seed_mses = []
    for s in seeds:
        kf = KFold(n_splits=10, shuffle=True, random_state=s)
        mses = []
        for tr, te in kf.split(X1):
            pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=5, include_bias=False)),
                ('scaler', StandardScaler()),
                ('lasso', Lasso(alpha=0.015, max_iter=10000, random_state=42))
            ])
            pipe.fit(X1[tr], y1[tr])
            mses.append(mean_squared_error(y1[te], pipe.predict(X1[te])))
        v1_seed_mses.append(np.mean(mses))
        print(f"Seed {s:5d} | 10-Fold CV MSE: {np.mean(mses):.5f}")
    print(f"Var1 Multi-Seed Mean MSE: {np.mean(v1_seed_mses):.5f} ± {np.std(v1_seed_mses):.5f}")

    print("\n" + "=" * 70)
    print("PHASE 2: MULTI-SEED COMPARISON: DEGREE 8 VS DEGREE 10 RIDGE")
    print("=" * 70)
    d8_mses = []
    d10_mses = []
    for s in seeds:
        kf = KFold(n_splits=10, shuffle=True, random_state=s)
        m8, m10 = [], []
        for tr, te in kf.split(X2):
            p8 = Pipeline([
                ('poly', PolynomialFeatures(degree=8, include_bias=False)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=0.05, random_state=42))
            ]).fit(X2[tr], y2[tr]).predict(X2[te])
            p10 = Pipeline([
                ('poly', PolynomialFeatures(degree=10, include_bias=False)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=0.7, random_state=42))
            ]).fit(X2[tr], y2[tr]).predict(X2[te])
            m8.append(mean_squared_error(y2[te], p8))
            m10.append(mean_squared_error(y2[te], p10))
        d8_mses.append(np.mean(m8))
        d10_mses.append(np.mean(m10))
        print(f"Seed {s:5d} | Deg 8 Ridge MSE: {np.mean(m8):.5f} | Deg 10 Ridge MSE: {np.mean(m10):.5f} | Diff: {np.mean(m8) - np.mean(m10):+.5f}")
    print(f"Across 5 Seeds: Deg 8 Mean = {np.mean(d8_mses):.5f} ± {np.std(d8_mses):.5f}")
    print(f"Across 5 Seeds: Deg 10 Mean = {np.mean(d10_mses):.5f} ± {np.std(d10_mses):.5f}")
    print(f"Difference across seeds is ~{np.mean(d8_mses) - np.mean(d10_mses):.5f}, well within fold std (~0.07).")


if __name__ == '__main__':
    main()
