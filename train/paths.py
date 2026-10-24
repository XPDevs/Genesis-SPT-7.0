from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = ROOT / "training data"
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
LOGS = DATA / "logs"
MODEL = ROOT / "model"
CHECKPOINTS = ROOT / "checkpoints"

MODEL_JSON = MODEL / "Genesis-SPT-7.0.json"
MODEL_BIN = MODEL / "Genesis-SPT-7.0.bin"

for p in (TRAINING_DATA, DATA, PROCESSED, LOGS, MODEL, CHECKPOINTS):
    p.mkdir(parents=True, exist_ok=True)
