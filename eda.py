#!/usr/bin/env python3
"""Convenience root wrapper for code/eda.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))
from eda import run_eda

if __name__ == '__main__':
    run_eda()
