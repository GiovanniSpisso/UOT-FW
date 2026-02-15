import numpy as np
import sys
import os

# Calculate path relative to current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up to UOT_FW
fast_uot_path = os.path.join(parent_dir, 'fast_uot-main')
sys.path.insert(0, fast_uot_path)

from fastuot.uot1d import solve_uot

