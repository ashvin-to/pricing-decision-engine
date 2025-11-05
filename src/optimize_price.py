"""Price optimization engine for revenue and margin maximization."""

from __future__ import annotations
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import config
from src.data_prep import load_raw_data
from src.features import build_features


def _align_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Ensure dataframe matches trained model feature signature."""
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


def optimize_product_price(
    product_id: str = "SKU001",
    price_range_pct: float = 0.25,
    grid_steps: int = 30,
) -> pd.DataFrame:
    """Simulate candidate prices and derive optimal revenue and margin price points."""
    config.ensure_directories()
    
    raw = load_raw_data()
    product_df = raw[raw["product_id"] == product_id].tail(45).copy()
    if product_df.empty:
        raise ValueError(f"No records found for product: {product_id}")

    with open(config.FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    model = joblib.load(config.MODEL_PATH)

    current_price = float(product_df["price"].median())
    unit_cost = float(product_df["cost"].median()) if "cost" in product_df.columns else 0.5 * current_price

    price_grid = np.linspace(
        current_price * (1.0 - price_range_pct),
        current_price * (1.0 + price_range_pct),
        grid_steps,
    )
    
    rows = []
    for candidate in price_grid:
        scenario = product_df.copy()
        scenario["price"] = candidate
        if "base_price" in scenario.columns:
            scenario["discount_pct"] = ((scenario["base_price"] - scenario["price"]) / scenario["base_price"]).clip(lower=0)
            
        modeling_df, _ = build_features(scenario)
        X = _align_features(modeling_df.copy(), feature_cols)
        pred_qty = np.maximum(0, model.predict(X))
        
        revenue = pred_qty * candidate
        margin = pred_qty * (candidate - unit_cost)
        
        rows.append({
            "product_id": product_id,
            "candidate_price": round(float(candidate), 2),
            "unit_cost": round(float(unit_cost), 2),
            "avg_predicted_quantity": round(float(np.mean(pred_qty)), 2),
            "avg_predicted_revenue": round(float(np.mean(revenue)), 2),
            "avg_predicted_margin": round(float(np.mean(margin)), 2),
        })

    results = pd.DataFrame(rows)
    best_revenue_row = results.loc[results["avg_predicted_revenue"].idxmax()]
    best_margin_row = results.loc[results["avg_predicted_margin"].idxmax()]
    
    baseline_idx = len(results) // 2
    baseline_rev = float(results.iloc[baseline_idx]["avg_predicted_revenue"])
    baseline_mar = float(results.iloc[baseline_idx]["avg_predicted_margin"])

    results["revenue_optimal_price"] = best_revenue_row["candidate_price"]
    results["margin_optimal_price"] = best_margin_row["candidate_price"]
    results["revenue_uplift_pct"] = round(((results["avg_predicted_revenue"] - baseline_rev) / (baseline_rev + 1e-9)) * 100, 2)
    results["margin_uplift_pct"] = round(((results["avg_predicted_margin"] - baseline_mar) / (baseline_mar + 1e-9)) * 100, 2)

    results.to_csv(config.OPTIMIZATION_PATH, index=False)

    # Plot 1: Optimization Value Frontier
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["candidate_price"], results["avg_predicted_revenue"], label="Expected Revenue ($)", color="#2ca02c", lw=2.5)
    ax.plot(results["candidate_price"], results["avg_predicted_margin"], label="Expected Profit Margin ($)", color="#1f77b4", lw=2.5)
    ax.axvline(best_revenue_row["candidate_price"], linestyle="--", color="#2ca02c", alpha=0.8, label=f"Max Rev: ${best_revenue_row['candidate_price']:.2f}")
    ax.axvline(best_margin_row["candidate_price"], linestyle=":", color="#1f77b4", alpha=0.8, label=f"Max Margin: ${best_margin_row['candidate_price']:.2f}")
    ax.set_title(f"OptiPrice Optimization Frontier — {product_id}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Candidate Price ($)")
    ax.set_ylabel("Expected Value ($)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(config.OPTIMIZATION_FIGURE_PATH, dpi=150)
    plt.close(fig)

    # Plot 2: Demand Response Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["candidate_price"], results["avg_predicted_quantity"], color="#d62728", lw=2.5)
    ax.set_title(f"OptiPrice Empirical Demand Response — {product_id}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Candidate Price ($)")
    ax.set_ylabel("Expected Quantity Demand (Units)")
    plt.tight_layout()
    plt.savefig(config.DEMAND_CURVE_PATH, dpi=150)
    plt.close(fig)

    return results
