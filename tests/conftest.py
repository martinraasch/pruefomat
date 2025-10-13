"""Pytest configuration helpers for the pruefomat project."""

from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("PRUEFOMAT_DISABLE_GRADIO", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
SYNTH_SRC = PROJECT_ROOT.parent / "data_synthethizer" / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if SYNTH_SRC.exists() and str(SYNTH_SRC) not in sys.path:
    sys.path.insert(0, str(SYNTH_SRC))
