"""Pytest configuration for diffaero_newton tests."""

import sys
import os

# Add source to path
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
source_path = os.path.join(project_root, "source")
sys.path.insert(0, source_path)
