"""OptiPrice Interactive Decision Cockpit (Streamlit)."""

from __future__ import annotations
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import config
from src.features import build_features

st.set_page_config(
    page_title="OptiPrice | Pricing Decision Cockpit",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏷️ OptiPrice — Decision & Optimization Cockpit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Prescriptive pricing, elasticity discovery, and competitor shock simulation</div>', unsafe_allow_html=True)

if not config.MODEL_PATH.exists() or not config.RAW_DATA_PATH.exists():
    st.error("Pre-trained model or data not found. Please execute `python main.py` first to calibrate the models.")
    st.stop()

raw = pd.read_csv(config.RAW_DATA_PATH, parse_dates=["date"])
model = joblib.load(config.MODEL_PATH)
with open(config.FEATURES_PATH, "r", encoding="utf-8") as f:
    feature_cols = json.load(f)
with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)
elasticity_df = pd.read_csv(config.ELASTICITY_PATH) if config.ELASTICITY_PATH.exists() else pd.DataFrame()
opt_df = pd.read_csv(config.OPTIMIZATION_PATH) if config.OPTIMIZATION_PATH.exists() else pd.DataFrame()

# Top metric ribbon
c1, c2, c3, c4 = st.columns(4)
c1.metric("Demand Model RMSE", f"{metrics.get('rmse', 0):.2f}")
c2.metric("Demand Model MAE", f"{metrics.get('mae', 0):.2f}")
c3.metric("Test R² Accuracy", f"{metrics.get('r2', 0):.3f}")
c4.metric("Active Catalog SKUs", f"{raw['product_id'].nunique():,}")

st.divider()

# Sidebar
st.sidebar.header("🕹️ Strategy Parameters")
products = sorted(raw["product_id"].unique())
selected_product = st.sidebar.selectbox("Target Product (SKU)", products, index=0)
base = raw[raw["product_id"] == selected_product].tail(30).copy()

base_price = float(base["price"].median())
base_comp = float(base["competitor_price"].median()) if "competitor_price" in base.columns else base_price
base_cost = float(base["cost"].median()) if "cost" in base.columns else base_price * 0.5

st.sidebar.markdown(f"**Baseline Price:** `${base_price:.2f}` | **Unit Cost:** `${base_cost:.2f}`")

candidate_price = st.sidebar.slider(
    "Set Candidate Price ($)",
    min_value=float(round(base_price * 0.6, 2)),
    max_value=float(round(base_price * 1.4, 2)),
    value=float(round(base_price, 2)),
    step=0.25,
)
competitor_price = st.sidebar.slider(
    "Competitor Reaction ($)",
    min_value=float(round(base_comp * 0.6, 2)),
    max_value=float(round(base_comp * 1.4, 2)),
    value=float(round(base_comp, 2)),
    step=0.25,
)
promo_flag = st.sidebar.selectbox(
    "Promotional Campaign",
    [0, 1],
    format_func=lambda x: "Active Promotion (1)" if x == 1 else "Standard Pricing (0)",
)
scenario_name = st.sidebar.text_input("Scenario Label", value="Strategic Price Shift")

# Run scenario evaluation
scenario = base.copy()
scenario["price"] = candidate_price
scenario["competitor_price"] = competitor_price
scenario["promo_flag"] = promo_flag
if "base_price" in scenario.columns:
    scenario["discount_pct"] = ((scenario["base_price"] - scenario["price"]) / scenario["base_price"]).clip(lower=0.0)

model_df, _ = build_features(scenario)
for col in feature_cols:
    if col not in model_df.columns:
        model_df[col] = 0
X = model_df[feature_cols]
pred_qty = np.maximum(0, model.predict(X))
model_df["predicted_quantity"] = pred_qty
model_df["predicted_revenue"] = model_df["predicted_quantity"] * model_df["price"]
model_df["predicted_margin"] = model_df["predicted_quantity"] * (model_df["price"] - base_cost)

# Compute baseline numbers for comparison
base_model_df, _ = build_features(base)
for col in feature_cols:
    if col not in base_model_df.columns:
        base_model_df[col] = 0
base_pred = np.maximum(0, model.predict(base_model_df[feature_cols]))
b_qty = float(base_pred.mean())
b_rev = float((base_pred * base_model_df["price"].values).mean())
b_mar = float((base_pred * (base_model_df["price"].values - base_cost)).mean())

s_qty = float(model_df["predicted_quantity"].mean())
s_rev = float(model_df["predicted_revenue"].mean())
s_mar = float(model_df["predicted_margin"].mean())

d_qty = ((s_qty - b_qty) / (b_qty + 1e-9)) * 100
d_rev = ((s_rev - b_rev) / (b_rev + 1e-9)) * 100
d_mar = ((s_mar - b_mar) / (b_mar + 1e-9)) * 100

t1, t2, t3 = st.tabs(["🔮 Scenario Simulation", "🎯 Optimization Frontier", "📊 Elasticity Analysis"])

with t1:
    st.subheader(f"Simulation Outcome: {scenario_name} ({selected_product})")
    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted Daily Demand", f"{s_qty:.1f} units", f"{d_qty:+.1f}% vs baseline")
    k2.metric("Predicted Daily Revenue", f"${s_rev:,.2f}", f"{d_rev:+.1f}% vs baseline")
    k3.metric("Predicted Daily Profit", f"${s_mar:,.2f}", f"{d_mar:+.1f}% vs baseline")

    st.markdown("#### Forecast Horizon Trajectory")
    st.line_chart(model_df[["predicted_quantity", "predicted_revenue", "predicted_margin"]])

    with st.expander("🔍 View Simulation Data Table"):
        show_cols = ["date", "price", "competitor_price", "promo_flag", "predicted_quantity", "predicted_revenue", "predicted_margin"]
        avail = [c for c in show_cols if c in model_df.columns]
        st.dataframe(model_df[avail].tail(15), use_container_width=True)

with t2:
    st.subheader(f"Price Optimization & Margin Tradeoff ({selected_product})")
    if not opt_df.empty:
        prod_opt = opt_df[opt_df["product_id"] == selected_product] if "product_id" in opt_df.columns else opt_df
        st.dataframe(prod_opt, use_container_width=True)
    else:
        st.info("Execute 'python main.py' to generate the optimization table.")

with t3:
    st.subheader("Price Elasticity of Demand (PED) Summary")
    if not elasticity_df.empty:
        st.dataframe(elasticity_df, use_container_width=True)
    else:
        st.info("No elasticity data available. Run 'python main.py' first.")
