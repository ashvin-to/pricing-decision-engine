"""Data ingestion and validation for OptiPrice."""

from __future__ import annotations
import pandas as pd
from src.config import config


def load_raw_data() -> pd.DataFrame:
    """Load and validate the raw pricing transaction dataset."""
    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found at: {config.RAW_DATA_PATH}")
    
    df = pd.read_csv(config.RAW_DATA_PATH, parse_dates=["date"])
    
    required_cols = {"date", "product_id", "price", "cost", "quantity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")

    return df.sort_values(["product_id", "date"]).reset_index(drop=True)
