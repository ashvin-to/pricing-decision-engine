"""OptiPrice Configuration and Path Management."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Project directory paths and default parameters."""
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    
    # Data paths
    RAW_DATA_PATH: Path = (
        (BASE_DIR / "data" / "raw" / "pricing_data.csv")
        if (BASE_DIR / "data" / "raw" / "pricing_data.csv").exists()
        else (BASE_DIR / "pricing_data.csv")
    )
    PROCESSED_DATA_PATH: Path = BASE_DIR / "data" / "processed" / "modeling_data.csv"
    
    # Model artifacts
    MODEL_PATH: Path = BASE_DIR / "outputs" / "models" / "demand_model.joblib"
    FEATURES_PATH: Path = BASE_DIR / "outputs" / "models" / "feature_columns.json"
    
    # Results & figures
    METRICS_PATH: Path = BASE_DIR / "outputs" / "results" / "model_metrics.json"
    ELASTICITY_PATH: Path = BASE_DIR / "outputs" / "results" / "elasticity_summary.csv"
    OPTIMIZATION_PATH: Path = BASE_DIR / "outputs" / "results" / "pricing_optimization_results.csv"
    SIMULATION_PATH: Path = BASE_DIR / "outputs" / "results" / "scenario_simulation_sample.csv"
    
    FEATURE_IMPORTANCE_PATH: Path = BASE_DIR / "outputs" / "figures" / "feature_importance.png"
    OPTIMIZATION_FIGURE_PATH: Path = BASE_DIR / "outputs" / "figures" / "optimization_curve.png"
    DEMAND_CURVE_PATH: Path = BASE_DIR / "outputs" / "figures" / "demand_curve_example.png"

    @classmethod
    def ensure_directories(cls) -> None:
        """Create all required directories if they don't exist."""
        cls.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)


config = Config()
