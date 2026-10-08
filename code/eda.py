#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) Script
Author: Jashandeep Singh Bedi (IMT2024022)
"""

from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'code' else SCRIPT_DIR
ROLL_NO = "IMT2024022"


def resolve_path(relative_candidate):
    candidates = [
        REPO_ROOT / "data" / relative_candidate,
        REPO_ROOT / ROLL_NO / relative_candidate,
        REPO_ROOT / relative_candidate,
        Path(relative_candidate)
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def run_eda():
    train1 = pd.read_csv(resolve_path(f"{ROLL_NO}_train_var1.csv"))
    test1 = pd.read_csv(resolve_path(f"{ROLL_NO}_test_var1.csv"))
    train2 = pd.read_csv(resolve_path(f"{ROLL_NO}_train_var2.csv"))
    test2 = pd.read_csv(resolve_path(f"{ROLL_NO}_test_var2.csv"))

    print("=== PHASE 1 (var1): STEAM TURBINE OPTIMIZATION ===")
    print("Train shape:", train1.shape)
    print("Test shape:", test1.shape)
    print("Missing values train:", train1.isnull().sum().to_dict())
    print("Missing values test:", test1.isnull().sum().to_dict())
    print("\nTrain Summary:\n", train1.describe().T[['mean', 'std', 'min', '50%', 'max']])
    print("\nTest Summary:\n", test1.describe().T[['mean', 'std', 'min', '50%', 'max']])

    print("\n=== PHASE 2 (var2): SUBTERRANEAN THERMAL RESERVOIR ===")
    print("Train shape:", train2.shape)
    print("Test shape:", test2.shape)
    print("Missing values train:", train2.isnull().sum().to_dict())
    print("Missing values test:", test2.isnull().sum().to_dict())
    print("\nTrain Summary:\n", train2.describe().T[['mean', 'std', 'min', '50%', 'max']])
    print("\nTest Summary:\n", test2.describe().T[['mean', 'std', 'min', '50%', 'max']])


if __name__ == '__main__':
    run_eda()
