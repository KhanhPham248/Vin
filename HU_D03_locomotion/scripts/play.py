"""Play/evaluation entry point for HU_D03 locomotion."""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
# Add our package src
sys.path.insert(0, str(_ROOT / "src"))
# Add mjlab src (sibling folder) so import works without pip install
sys.path.insert(0, str(_ROOT.parent / "mjlab" / "mjlab-main" / "src"))

import importlib
importlib.import_module("hu_d03_locomotion")

from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
    main()
