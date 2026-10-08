#!/usr/bin/env python3
"""Convenience root wrapper for code/generate_final_models.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))
from generate_final_models import main

if __name__ == '__main__':
    main()
