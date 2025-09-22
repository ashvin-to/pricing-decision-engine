"""Demand forecasting model training and serialization."""

from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import config
from src.data_prep import load_raw_data
from src.features import build_features


def train_demand_model() -> dict[str, float]:
    """Train Gradient Boosting demand forecaster and export metrics & artifacts."""
    config.ensure_directories()
    
    raw = load_raw_data()
    modeling_df, feature_cols = build_features(raw)
    modeling_df.to_csv(config.PROCESSED_DATA_PATH, index=False)

    X = modeling_df[feature_cols]
    y = modeling_df["quantity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "rmse": round(float(mean_squared_error(y_test, preds) ** 0.5), 4),
        "mae": round(float(mean_absolute_error(y_test, preds)), 4),
        "r2": round(float(r2_score(y_test, preds)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # Save artifacts
    with open(config.FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(model, config.MODEL_PATH)

    # Export styled feature importance plot
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.sort_values().plot(kind="barh", color="#1f77b4", ax=ax)
    ax.set_title("OptiPrice — Key Drivers of Demand (Feature Importance)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Relative Importance Score")
    plt.tight_layout()
    plt.savefig(config.FEATURE_IMPORTANCE_PATH, dpi=150)
    plt.close(fig)

    return metrics
