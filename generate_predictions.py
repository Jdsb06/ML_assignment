#!/usr/bin/env python3
"""
End-to-End Prediction Generation and Verification Script
Trains optimal models for var1 and var2 and exports final submission files.
Author: IMT2024022
"""

import os
import pandas as pd
from train_var1 import train_and_evaluate as run_var1
from train_var2 import train_and_evaluate as run_var2

ROLL_NO = "IMT2024022"


def main():
    print("=" * 75)
    print(f"RUNNING COMPLETE PREDICTION PIPELINE FOR {ROLL_NO}")
    print("=" * 75)

    # 1. Run var1 pipeline
    pred1_path = f"{ROLL_NO}_pred_var1.csv"
    run_var1(
        train_path=f"{ROLL_NO}/{ROLL_NO}_train_var1.csv",
        test_path=f"{ROLL_NO}/{ROLL_NO}_test_var1.csv",
        out_path=pred1_path,
        degree=5,
        alpha=0.007
    )

    # 2. Run var2 pipeline
    pred2_path = f"{ROLL_NO}_pred_var2.csv"
    run_var2(
        train_path=f"{ROLL_NO}/{ROLL_NO}_train_var2.csv",
        test_path=f"{ROLL_NO}/{ROLL_NO}_test_var2.csv",
        out_path=pred2_path,
        degree=10,
        alpha=0.7
    )

    # 3. Validation checks
    print("\n" + "=" * 75)
    print("VERIFICATION & INTEGRITY CHECKS")
    print("=" * 75)

    sample_df = pd.read_csv("sample_submission.csv")
    expected_rows = len(sample_df)
    expected_cols = list(sample_df.columns)

    for path in [pred1_path, pred2_path]:
        df = pd.read_csv(path)
        assert list(df.columns) == expected_cols, f"Mismatch in columns for {path}: expected {expected_cols}, got {list(df.columns)}"
        assert len(df) == expected_rows, f"Mismatch in row count for {path}: expected {expected_rows}, got {len(df)}"
        assert df['y'].isnull().sum() == 0, f"Found NaN values in {path}!"
        print(f"✓ {path}: OK (shape={df.shape}, nulls={df['y'].isnull().sum()}, header={list(df.columns)})")

    print("=" * 75)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == '__main__':
    main()
