import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.run_all import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(current_dir, run_robustness=True)
