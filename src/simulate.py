"""Scenario simulation engine for what-if shock analysis."""

from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd

from src.config import config
from src.data_prep import load_raw_data
from src.features import build_features


def _align_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


def run_sample_simulation(
    product_id: str = "SKU001",
    price_change_pct: float = 0.05,
    competitor_change_pct: float = -0.03,
    promo_flag: int = 1,
) -> pd.DataFrame:
    """Simulate market shocks and compute predicted revenue, volume, and margins."""
    config.ensure_directories()
    
    raw = load_raw_data()
    base_rows = raw[raw["product_id"] == product_id].tail(30).copy()
    if base_rows.empty:
        raise ValueError(f"No observations found for SKU: {product_id}")

    base_rows["price"] = base_rows["price"] * (1.0 + price_change_pct)
    if "competitor_price" in base_rows.columns:
        base_rows["competitor_price"] = base_rows["competitor_price"] * (1.0 + competitor_change_pct)
    base_rows["promo_flag"] = promo_flag
    
    if "base_price" in base_rows.columns:
        base_rows["discount_pct"] = ((base_rows["base_price"] - base_rows["price"]) / base_rows["base_price"]).clip(lower=0)

    modeling_df, _ = build_features(base_rows)
    with open(config.FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    model = joblib.load(config.MODEL_PATH)
    
    X = _align_features(modeling_df.copy(), feature_cols)
    modeling_df["predicted_quantity"] = np.maximum(0, model.predict(X))
    modeling_df["predicted_revenue"] = modeling_df["predicted_quantity"] * modeling_df["price"]
    
    unit_cost = modeling_df["cost"] if "cost" in modeling_df.columns else modeling_df["price"] * 0.5
    modeling_df["predicted_margin"] = modeling_df["predicted_quantity"] * (modeling_df["price"] - unit_cost)
    modeling_df["product_id"] = product_id

    cols = [
        "product_id", "price", "competitor_price", "promo_flag",
        "predicted_quantity", "predicted_revenue", "predicted_margin",
    ]
    avail_cols = [c for c in cols if c in modeling_df.columns]
    result = modeling_df[avail_cols].copy()
    
    result.to_csv(config.SIMULATION_PATH, index=False)
    return result
