#!/usr/bin/env python3
"""Run script for AUTOSAR extractor - can be run from outside the package"""
from main import main
import sys
from pathlib import Path

# Add autosar directory to path
autosar_dir = Path(__file__).parent
sys.path.insert(0, str(autosar_dir))


if __name__ == "__main__":
    main()
