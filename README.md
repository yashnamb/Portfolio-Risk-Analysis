# Financial Portfolio Analyzer

A full-stack Streamlit dashboard to analyze, simulate, and visualize your investment portfolio with advanced financial metrics, real-time data, and interactive plots. User centric for first time starters, part time investors and students who are just breaking into stock market.

Created for the BA870 Financial Analytics course by  
** Hemangi Suthar, Mahika Jain, Yashna Meher **

---

##  Features

-  **Portfolio Performance** (vs. S&P 500)
-  **Monte Carlo Simulation** for future projections
-  **Risk Metrics** (Volatility, Drawdown, VaR)
-  **Technical Indicators** (Moving Averages, Bollinger Bands, RSI)
-  **Stock Clustering** (K-Means based on return patterns)
-  **Sector Allocation**, **Asset Correlation**, and **Drawdown Analysis**
-  **Real-Time Prices** and S&P 500 market context

---

##  Data Sources

### 📉 Historical Data
Pulled using the `yfinance` API:
- Daily adjusted closing prices for user-selected stocks
- Historical S&P 500 index data (`^GSPC`) for benchmark comparison
- Used in: performance, risk metrics, Monte Carlo simulation, clustering

### 📈 Real-Time Data
- Real-time stock prices fetched using `yfinance.Ticker().history(period='1d')`
- Used to calculate live total investment value and show recent prices

---

## APIs & Libraries

| Tool/Library      | Purpose |
|-------------------|---------|
| `yfinance`        | Pull historical & real-time stock data (Yahoo Finance) |
| `plotly`          | Generate interactive visualizations |
| `pandas`, `numpy` | Data manipulation and calculation |
| `scikit-learn`    | Clustering (KMeans), scaling |
| `statsmodels`     | OLS regression (Beta calculation) |
| `streamlit`       | Interactive web app frontend |
| `datetime`        | Date handling for simulations and plots |

---

## 🧮 Simulations & Metrics

- **Monte Carlo Simulation**: Projects future portfolio value using normal returns
- **Efficient Frontier**: Optimal risk-return portfolios using `scipy.optimize`
- **Sharpe Ratio**: Risk-adjusted return
- **Value at Risk (VaR)**: 95% and 99% threshold risk
- **Beta**: Sensitivity to S&P 500
- **Drawdown**: Maximum drop from a peak

---

## 📊 Technical Analysis

For selected individual stocks:
- **20-day & 50-day Moving Averages**
- **Bollinger Bands**
- **RSI (Relative Strength Index)**
- Optional: Add volume bars (under development)

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
streamlit run app_final.py
