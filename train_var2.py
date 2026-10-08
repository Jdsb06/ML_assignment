#!/usr/bin/env python3
"""Convenience root wrapper for code/train_var2.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))
import train_var2

if __name__ == '__main__':
    train_var2.train_and_evaluate()
