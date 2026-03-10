#!/usr/bin/env python
"""CLI launcher for DiffAero Newton training."""

import sys
import os

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

from diffaero_newton.__main__ import main

if __name__ == "__main__":
    main()
