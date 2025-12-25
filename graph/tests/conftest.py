"""Pytest configuration and fixtures"""
import sys
from pathlib import Path

# Add autosar directory to path
autosar_dir = Path(__file__).parent.parent
sys.path.insert(0, str(autosar_dir))
