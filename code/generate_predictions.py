#!/usr/bin/env python3
"""
End-to-End Prediction Generation and Verification Script
Trains optimal models for var1 and var2 and exports final submission files.
Author: Jashandeep Singh Bedi (IMT2024022)
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add script directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'code' else SCRIPT_DIR
sys.path.insert(0, str(SCRIPT_DIR))

from train_var1 import train_and_evaluate as run_var1
from train_var2 import train_and_evaluate as run_var2

ROLL_NO = "IMT2024022"


def main():
    print("=" * 75)
    print(f"RUNNING COMPLETE PREDICTION PIPELINE FOR {ROLL_NO}")
    print("=" * 75)

    # 1. Run var1 pipeline (Degree 5 Lasso, alpha=0.007)
    run_var1(degree=5, alpha=0.007)

    # 2. Run var2 pipeline (Degree 10 Ridge, alpha=0.7)
    run_var2(degree=10, alpha=0.7)

    # 3. Validation checks
    print("\n" + "=" * 75)
    print("VERIFICATION & INTEGRITY CHECKS")
    print("=" * 75)

    sample_candidates = [
        REPO_ROOT / "data" / "sample_submission.csv",
        REPO_ROOT / "sample_submission.csv"
    ]
    sample_file = next((f for f in sample_candidates if f.exists()), None)
    expected_rows = 1000
    expected_cols = ['y']
    if sample_file:
        s_df = pd.read_csv(sample_file)
        expected_rows = len(s_df)
        expected_cols = list(s_df.columns)

    check_paths = [
        REPO_ROOT / f"{ROLL_NO}_pred_var1.csv",
        REPO_ROOT / f"{ROLL_NO}_pred_var2.csv",
        REPO_ROOT / "predictions" / f"{ROLL_NO}_pred_var1.csv",
        REPO_ROOT / "predictions" / f"{ROLL_NO}_pred_var2.csv"
    ]

    for path in check_paths:
        if path.exists():
            df = pd.read_csv(path)
            assert list(df.columns) == expected_cols, f"Mismatch in columns for {path}: expected {expected_cols}, got {list(df.columns)}"
            assert len(df) == expected_rows, f"Mismatch in row count for {path}: expected {expected_rows}, got {len(df)}"
            assert df['y'].isnull().sum() == 0, f"Found NaN values in {path}!"
            print(f"✓ {path.relative_to(REPO_ROOT)}: OK (shape={df.shape}, nulls={df['y'].isnull().sum()}, header={list(df.columns)})")

    print("=" * 75)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == '__main__':
    main()
