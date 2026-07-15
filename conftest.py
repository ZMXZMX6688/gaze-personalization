import os
import sys

# Ensure the repo root (where personalized_main_sequence.py lives) is importable
# regardless of pytest's import mode / invocation directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
