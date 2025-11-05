"""Price elasticity estimation using econometric log-log modeling."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.config import config
from src.data_prep import load_raw_data


def classify_elasticity_regime(own_price_coef: float) -> str:
    """Classify product price sensitivity regime based on price elasticity of demand."""
    abs_e = abs(own_price_coef)
    if abs_e > 1.10:
        return "Elastic (High Price Sensitivity)"
    elif abs_e < 0.90:
        return "Inelastic (Volume Insensitive / Pricing Power)"
    else:
        return "Unitary Elastic"


def calculate_elasticity() -> pd.DataFrame:
    """Estimate own-price, cross-price, and promotion elasticities across catalog SKUs."""
    config.ensure_directories()
    
    df = load_raw_data().copy()
    valid_mask = (df["price"] > 0) & (df["quantity"] > 0)
    if "competitor_price" in df.columns:
        valid_mask &= (df["competitor_price"] > 0)
    
    df = df[valid_mask].copy()
    df["log_price"] = np.log(df["price"])
    df["log_quantity"] = np.log(df["quantity"])
    if "competitor_price" in df.columns:
        df["log_competitor_price"] = np.log(df["competitor_price"])
    else:
        df["log_competitor_price"] = df["log_price"]

    rows = []
    for product_id, grp in df.groupby("product_id"):
        if len(grp) < 15:
            continue
        
        feature_subset = ["log_price", "log_competitor_price"]
        if "promo_flag" in grp.columns:
            feature_subset.append("promo_flag")
            
        X = grp[feature_subset]
        y = grp["log_quantity"]
        
        model = LinearRegression()
        model.fit(X, y)
        
        own_coef = float(model.coef_[0])
        cross_coef = float(model.coef_[1]) if len(model.coef_) > 1 else 0.0
        promo_coef = float(model.coef_[2]) if len(model.coef_) > 2 else 0.0
        r2 = float(model.score(X, y))

        rows.append({
            "product_id": product_id,
            "own_price_elasticity": round(own_coef, 4),
            "elasticity_regime": classify_elasticity_regime(own_coef),
            "cross_price_elasticity": round(cross_coef, 4),
            "promo_lift_effect": round(promo_coef, 4),
            "model_r2": round(r2, 4),
            "sample_size": int(len(grp)),
        })

    result = pd.DataFrame(rows).sort_values("product_id")
    result.to_csv(config.ELASTICITY_PATH, index=False)
    return result
