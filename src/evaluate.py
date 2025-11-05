"""Model evaluation and metrics report utility."""

from __future__ import annotations
import json
from src.config import config


def read_metrics() -> dict[str, float]:
    """Read stored evaluation metrics from disk."""
    if not config.METRICS_PATH.exists():
        raise FileNotFoundError("Metrics file not found. Please run 'python main.py' first.")
    with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
