import sys
import os

# Ensure the repository root and the src/ directory are on sys.path so tests
# can import application modules (scripts, utils, src.*) regardless of how
# pytest is invoked.
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
