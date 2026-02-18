"""OptiPrice Command-Line Pipeline Runner."""

from __future__ import annotations
import argparse
import sys
from src.config import config
from src.elasticity import calculate_elasticity
from src.optimize_price import optimize_product_price
from src.simulate import run_sample_simulation
from src.train_model import train_demand_model


def print_banner() -> None:
    print(r"""
   ___       _   _ _____     _          
  / _ \ _ __| |_(_) ___| __ (_) ___ ___ 
 | | | | '_ \ __| | |  _ '_ \| |/ __/ _ \
 | |_| | |_) | |_| | |_| |_) | | (_|  __/
  \___/| .__/ \__|_|\____ .__/|_|\___\___|
       |_|               |_|              
       Dynamic Pricing Decision Engine
    """)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OptiPrice — Enterprise Pricing Optimization & Elasticity Decision Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sku", type=str, default="SKU001", help="Target SKU for optimization and simulation")
    parser.add_argument("--steps", type=int, default=30, help="Grid search resolution steps for optimization")
    parser.add_argument("--price-delta", type=float, default=0.05, help="Simulation price shock delta (+5% = 0.05)")
    args = parser.parse_args()

    print_banner()
    config.ensure_directories()

    print(f"\n[1/4] 🏋️ Training Gradient Boosting Demand Forecaster...")
    metrics = train_demand_model()
    print(f"      ├── Train Samples : {metrics['n_train']:,}")
    print(f"      ├── Test Samples  : {metrics['n_test']:,}")
    print(f"      ├── RMSE          : {metrics['rmse']:.2f}")
    print(f"      ├── MAE           : {metrics['mae']:.2f}")
    print(f"      └── Test R² Score : {metrics['r2']:.4f}")

    print(f"\n[2/4] 📐 Econometric Elasticity & Regime Tagging...")
    elasticity_df = calculate_elasticity()
    print(f"      └── Processed {len(elasticity_df)} SKUs. Sample elasticity:")
    for _, row in elasticity_df.head(3).iterrows():
        print(f"          • {row['product_id']}: PED = {row['own_price_elasticity']:+.2f} ({row['elasticity_regime']})")

    print(f"\n[3/4] 🎯 Pricing Optimization Frontier for [{args.sku}]...")
    opt_df = optimize_product_price(product_id=args.sku, grid_steps=args.steps)
    rev_p = opt_df["revenue_optimal_price"].iloc[0]
    mar_p = opt_df["margin_optimal_price"].iloc[0]
    rev_uplift = opt_df["revenue_uplift_pct"].max()
    mar_uplift = opt_df["margin_uplift_pct"].max()
    print(f"      ├── Revenue-Optimal Price : ${rev_p:.2f} (Max Uplift: {rev_uplift:+.1f}%)")
    print(f"      └── Margin-Optimal Price  : ${mar_p:.2f} (Max Uplift: {mar_uplift:+.1f}%)")

    print(f"\n[4/4] 🔮 Running Market Shock Simulation for [{args.sku}]...")
    sim_df = run_sample_simulation(product_id=args.sku, price_change_pct=args.price_delta)
    avg_qty = sim_df["predicted_quantity"].mean()
    avg_rev = sim_df["predicted_revenue"].mean()
    avg_mar = sim_df["predicted_margin"].mean()
    print(f"      ├── Simulated Mean Daily Volume : {avg_qty:.1f} units")
    print(f"      ├── Simulated Mean Daily Revenue: ${avg_rev:,.2f}")
    print(f"      └── Simulated Mean Daily Margin : ${avg_mar:,.2f}")

    print("\n" + "=" * 65)
    print("  ✅ All pipeline stages executed successfully!")
    print("  🚀 Launch UI: streamlit run app/app.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
