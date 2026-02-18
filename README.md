# 🏷️ OptiPrice — Enterprise Pricing Optimization & Elasticity Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Gradient%20Boosting-orange.svg)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**OptiPrice** is an end-to-end pricing data science platform combining predictive machine learning, econometric elasticity modeling, and constrained optimization to discover revenue- and margin-maximizing price points.

---

## 🚀 Key Modules & Capabilities

1. **Predictive Demand Forecaster**:
   - High-capacity Gradient Boosting Regressor capturing seasonal patterns, calendar effects, promotional intensity, and competitor relative pricing.
   - Automated feature attribution extraction.

2. **Econometric Elasticity Discovery**:
   - Log-log multi-variable regression estimating own-price elasticity of demand (PED), cross-price elasticity, and promotional lift coefficients.
   - Automatic classification into **Elastic**, **Inelastic**, and **Unitary Elastic** pricing regimes.

3. **Optimization Frontier Engine**:
   - Multi-objective grid search determining both **revenue-optimal** and **margin-optimal** price points.
   - Evaluates expected volume and financial delta uplift against historical baselines.

4. **Scenario Simulation & Stress-Testing**:
   - Simulates competitor discounting shocks, promotional campaigns, and supply constraints.

5. **Decision Cockpit (Streamlit)**:
   - Interactive business interface for category managers with KPI delta cards, time-series projections, and data export.

---

## 📁 Project Architecture

```text
OptiPrice/
├── data/
│   ├── raw/                 <- Raw transactions dataset (pricing_data.csv)
│   └── processed/           <- Processed feature-engineered data
├── src/
│   ├── config.py            <- Centralized configuration & directory orchestrator
│   ├── data_prep.py         <- Ingestion and data validation
│   ├── features.py          <- Lags, rolling windows, and competitive ratios
│   ├── train_model.py       <- Demand model training, validation, & feature importance
│   ├── elasticity.py        <- Econometric elasticity modeling & regime tagging
│   ├── optimize_price.py    <- Price optimization frontier & curve generator
│   ├── simulate.py          <- Competitor shock and what-if scenario engine
│   └── evaluate.py          <- Model performance evaluation
├── app/
│   └── app.py               <- Streamlit interactive decision cockpit
├── outputs/
│   ├── models/              <- Serialized model artifacts (joblib) & feature schemas
│   ├── figures/             <- Feature importance, demand curves, and frontiers
│   └── results/             <- CSV summaries of elasticity and optimal prices
├── main.py                  <- CLI pipeline runner
├── requirements.txt
└── README.md
```

---

## ⚡ Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline via CLI
```bash
python main.py --sku SKU001 --steps 30 --price-delta 0.05
```

### 3. Launch Interactive Cockpit
```bash
streamlit run app/app.py
```

---

## 📊 Objective Formulations

- **Revenue Objective**:
  $$\max_p R(p) = p \cdot \hat{Q}(p)$$
- **Margin Objective**:
  $$\max_p M(p) = (p - c) \cdot \hat{Q}(p)$$
- **Log-Log Econometric Elasticity**:
  $$\ln(Q) = \alpha + \beta_{\text{own}} \ln(P) + \beta_{\text{cross}} \ln(P_{\text{comp}}) + \beta_{\text{promo}} \cdot \text{Promo} + \epsilon$$

---

## 📜 License
MIT License.
