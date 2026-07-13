"""
conftest.py
───────────
pytest configuration for HealthGuard-XAI.
Ensures the project root is in sys.path so all imports resolve correctly.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
