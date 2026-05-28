"""Điểm bắt đầu chạy đánh giá cho HU_D03."""

import sys
from pathlib import Path

# Cấu hình đường dẫn hệ thống
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT.parent / "mjlab" / "mjlab-main" / "src"))

import importlib
importlib.import_module("hu_d03_03")

from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
    main()
