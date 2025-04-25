# dynamic_portfolio_app.py
# =============================================================================
# Dynamic Portfolio Optimizer & Interactive Risk Dashboard
# 
# Contributors:
#   - Alice Kumar    (UI/UX & caching)
#   - Bob Li         (Data ingestion & analytics)
#   - Charlie Smith  (Optimization & advanced plotting)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# 1. Page & Sidebar Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Portfolio Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("💼 Interactive Portfolio Optimizer & Risk Dashboard")
st.markdown(
    """
    Adjust your portfolio universe and risk-free rate. All charts and
    metrics update instantly when you modify inputs!
    """
)

# Sidebar inputs
st.sidebar.header("Universe & Data Range")
assets = st.sidebar.multiselect(
    "Select tickers:",
    options=["AAPL","MSFT","GOOGL","AMZN","TSLA","META","JPM","V","JNJ","WMT"],
    default=["AAPL","MSFT","GOOGL"]
)
start = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end   = st.sidebar.date_input("End date",   pd.to_datetime("today"))

st.sidebar.header("Risk-Free Rate")
rf_rate = st.sidebar.number_input(
    "Risk-free rate (annual):", min_value=0.0, max_value=0.10, value=0.01, step=0.001
)

if not assets:
    st.sidebar.error("Select at least one asset.")
    st.stop()
if start >= end:
    st.sidebar.error("Start must be before end.")
    st.stop()

# Fixed simulation & risk parameters
n_portfolios = 1000  # fixed Monte Carlo samples
conf_level = 95      # fixed VaR confidence level (%)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Loading & Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_prices(tickers, start, end):
    try:
        df = yf.download(tickers, start=start, end=end)["Adj Close"]
        if df.dropna(how="all").empty:
            raise ValueError
    except Exception:
        df = pdr.DataReader(tickers, "stooq", start=start, end=end)["Close"]
        df = df.sort_index()
    return df.dropna()

prices = load_prices(assets, start, end)
returns = prices.pct_change().dropna()

# Data preview
st.subheader("Price & Return Preview")
st.dataframe(prices.tail(5))
st.line_chart(prices)

# Correlation heatmap
st.subheader("Return Correlation")
fig_corr, ax_corr = plt.subplots(figsize=(4,4))
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
st.pyplot(fig_corr)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Portfolio Simulation & Efficient Frontier
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Monte Carlo Simulation & Efficient Frontier")
mean_ret = returns.mean() * 252
cov_mat  = returns.cov() * 252

# simulate random portfolios
np.random.seed(42)
results = np.zeros((n_portfolios, 3 + len(assets)))
for i in range(n_portfolios):
    w = np.random.random(len(assets))
    w /= np.sum(w)
    port_ret = np.dot(w, mean_ret)
    port_vol = np.sqrt(w.T @ cov_mat @ w)
    sharpe   = (port_ret - rf_rate) / port_vol
    results[i,:3] = [port_ret, port_vol, sharpe]
    results[i,3:] = w
cols = ["Return","Volatility","Sharpe"] + assets
sim_df = pd.DataFrame(results, columns=cols)

# efficient frontier: min vol for return bucket
ef = sim_df.loc[
    sim_df.groupby(sim_df['Return'].round(3))['Volatility']
           .idxmin()
]

# plot scatter + frontier
fig, ax = plt.subplots()
ax.scatter(
    sim_df['Volatility'], sim_df['Return'],
    c=sim_df['Sharpe'], cmap='viridis', alpha=0.5
)
ax.plot(ef['Volatility'], ef['Return'], 'r--', lw=2)
ax.set_xlabel('Annual Volatility')
ax.set_ylabel('Annual Return')
ax.set_title('Simulated Portfolios & Efficient Frontier')
st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Optimal Max-Sharpe Calculation & Metrics
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Optimal Portfolio (Max Sharpe)")

def neg_sharpe(w, mean_ret, cov, rf):
    ret = w.dot(mean_ret)
    vol = np.sqrt(w.T @ cov @ w)
    return -(ret - rf) / vol

cons = ({'type':'eq','fun':lambda w: np.sum(w)-1})
bnds = tuple((0,1) for _ in assets)
init = np.repeat(1/len(assets), len(assets))

opt = minimize(neg_sharpe, init, args=(mean_ret, cov_mat, rf_rate),
               method='SLSQP', bounds=bnds, constraints=cons)
opt_w = pd.Series(opt.x, index=assets)
opt_ret = np.dot(opt.x, mean_ret)
opt_vol = np.sqrt(opt.x.T @ cov_mat @ opt.x)
opt_sh  = (opt_ret - rf_rate) / opt_vol

# Display metrics as cards
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Expected Return", f"{opt_ret:.2%}")
with c2:
    st.metric("Volatility", f"{opt_vol:.2%}")
with c3:
    st.metric("Sharpe Ratio", f"{opt_sh:.2f}")

# Pie chart for weights
st.subheader("Optimal Weights Allocation")
fig_pie, ax_pie = plt.subplots()
ax_pie.pie(opt_w, labels=assets, autopct='%1.1f%%', startangle=140)
ax_pie.axis('equal')
st.pyplot(fig_pie)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Historical Performance & Risk Metrics
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Historical Performance & Risk Metrics")

# compute portfolio returns
df_port = returns.dot(opt_w)
cum_ret = (1 + df_port).cumprod()

# cumulative return chart
st.line_chart(cum_ret.rename('Portfolio Cumulative Return'))

# VaR & CVaR
alpha = 1 - conf_level/100
var = np.percentile(df_port, 100*alpha)
cvar = df_port[df_port <= var].mean()

# Show as cards
rc1, rc2 = st.columns(2)
with rc1:
    st.metric(f"{conf_level}% Historical VaR", f"{var:.2%}")
with rc2:
    st.metric(f"{conf_level}% Historical CVaR", f"{cvar:.2%}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Download Outputs
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Download Data")
csv_sim = sim_df.to_csv(index=False).encode('utf-8')
st.download_button("Simulations CSV", csv_sim, "simulations.csv", "text/csv")

csv_w = opt_w.to_csv(header=True).encode('utf-8')
st.download_button("Optimal Weights CSV", csv_w, "weights.csv", "text/csv")

st.markdown("---")
st.caption("Built with ❤️ by Alice, Bob & Charlie | Streamlit • yfinance • pandas_datareader • SciPy")

