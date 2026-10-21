"""Application configuration."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("GOLF_DATA_DIR", str(BASE_DIR / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
ASSETS_DIR = DATA_DIR / "assets"

for directory in (UPLOAD_DIR, RESULTS_DIR, ASSETS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "500"))
MAX_ANALYSIS_FRAMES = int(os.environ.get("MAX_ANALYSIS_FRAMES", "150"))
MIN_POSE_FRAMES = int(os.environ.get("MIN_POSE_FRAMES", "12"))

SWING_PHASES = (
    "address",
    "takeaway",
    "backswing",
    "top",
    "downswing",
    "impact",
    "follow_through",
    "finish",
)

PHASE_DISPLAY_NAMES = {
    "address": "Address",
    "takeaway": "Takeaway",
    "backswing": "Backswing",
    "top": "Top",
    "downswing": "Downswing",
    "impact": "Impact",
    "follow_through": "Follow Through",
    "finish": "Finish",
}
