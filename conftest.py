# This file makes the project importable when running pytest from the root.
# It adds the project root to sys.path so 'from src.config import ...' works.

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve()))
