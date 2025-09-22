"""Feature engineering pipeline for demand forecasting."""

from __future__ import annotations
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Transform raw transaction logs into modeling features with lags and price dynamics."""
    data = df.copy().sort_values(["product_id", "date"]).reset_index(drop=True)

    # Calendar features
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    # Price dynamics & competitive intelligence
    if "competitor_price" in data.columns:
        data["price_diff"] = data["price"] - data["competitor_price"]
        data["relative_price_index"] = data["price"] / data["competitor_price"].clip(lower=0.01)
    else:
        data["price_diff"] = 0.0
        data["relative_price_index"] = 1.0

    if "discount_pct" in data.columns:
        data["promo_intensity"] = data["discount_pct"]
    else:
        data["promo_intensity"] = 0.0

    # Margins and operational indicators
    if "cost" in data.columns:
        data["unit_margin"] = data["price"] - data["cost"]
    
    if "stock" in data.columns and "quantity" in data.columns:
        data["stock_cover_ratio"] = data["stock"] / data["quantity"].clip(lower=1)
    else:
        data["stock_cover_ratio"] = 10.0

    # Lags and rolling aggregates (grouped by product_id)
    if "quantity" in data.columns:
        data["lag_quantity_1"] = data.groupby("product_id")["quantity"].shift(1)
        data["lag_quantity_7"] = data.groupby("product_id")["quantity"].shift(7)
        data["rolling_quantity_7"] = (
            data.groupby("product_id")["quantity"]
            .shift(1)
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        data["lag_quantity_1"] = 0.0
        data["lag_quantity_7"] = 0.0
        data["rolling_quantity_7"] = 0.0

    data["rolling_price_7"] = (
        data.groupby("product_id")["price"]
        .shift(1)
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Impute initial rolling missing values with forward/backward fill per group
    data["lag_quantity_1"] = data.groupby("product_id")["lag_quantity_1"].bfill().fillna(0)
    data["lag_quantity_7"] = data.groupby("product_id")["lag_quantity_7"].bfill().fillna(0)
    data["rolling_quantity_7"] = data.groupby("product_id")["rolling_quantity_7"].bfill().fillna(0)
    data["rolling_price_7"] = data.groupby("product_id")["rolling_price_7"].bfill().fillna(data["price"])

    # Categorical encodings
    categorical_cols = [c for c in ["product_id", "category", "region", "store_format"] if c in data.columns]
    model_data = pd.get_dummies(data, columns=categorical_cols, drop_first=False)
    
    # Exclude targets and non-feature metadata
    excluded = {"date", "quantity", "revenue", "margin"}
    feature_cols = [c for c in model_data.columns if c not in excluded]

    return model_data, feature_cols
