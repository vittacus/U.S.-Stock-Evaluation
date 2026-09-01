import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backtest import buyhold_cumulative, rf_cumulative, log_cumulative, daily_return_test, rf_daily_return, log_daily_return
from backtest import sharpe_ratio, max_drawdown, drawdown_series
from train_model import importances, log_coefficients

st.set_page_config(page_title="VOO Quant Strategy", layout="wide")

st.title("VOO Quant/ML Trading Strategy")
st.write(
    "Comparing a Random Forest and Logistic Regression classifier, trained on "
    "return, moving average, volume, volatility, and RSI features, against simply "
    "holding VOO over the same held-out test period."
)

with st.expander("About this project"):
    st.write("""
    This model classifies whether VOO's price will be higher 5 trading days out,
    using five engineered features: multi-timeframe returns, moving averages
    (including the golden/death cross signal), volume ratio, rolling volatility,
    and RSI. Two classifiers (Logistic Regression, Random Forest) were trained
    on 2016–2024 data and backtested on a fully held-out 2024–2026 period,
    then converted into simple long/cash trading strategies and compared
    against buy and hold on total return, Sharpe ratio, and max drawdown.
    """)

# Equity curve chart
st.subheader("Cumulative Return: Strategy vs. Buy & Hold")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=buyhold_cumulative.index, y=buyhold_cumulative,
    name="Buy & Hold", line=dict(color="#4C9AFF", width=2.5)
))
fig.add_trace(go.Scatter(
    x=rf_cumulative.index, y=rf_cumulative,
    name="Random Forest Strategy", line=dict(color="#FF8A5C", width=2.5)
))
fig.add_trace(go.Scatter(
    x=log_cumulative.index, y=log_cumulative,
    name="Logistic Regression Strategy", line=dict(color="#4CD97B", width=2.5)
))

fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    height=450,
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis_title="Growth of $1",
    xaxis_title=None,
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Note: the Logistic Regression strategy was invested only 85 of 535 days, "
    "explaining its comparatively flat curve. That flatness reflects its much "
    "lower drawdown, not inactivity."
)

# Drawdown Chart
st.subheader("Drawdown Over Time")
dd_fig = go.Figure()
dd_fig.add_trace(go.Scatter(x=buyhold_cumulative.index, y=drawdown_series(buyhold_cumulative) * 100,
    name="Buy & Hold", line=dict(color="#4C9AFF", width=2), fill="tozeroy"))
dd_fig.add_trace(go.Scatter(x=rf_cumulative.index, y=drawdown_series(rf_cumulative) * 100,
    name="Random Forest Strategy", line=dict(color="#FF8A5C", width=2)))
dd_fig.add_trace(go.Scatter(x=log_cumulative.index, y=drawdown_series(log_cumulative) * 100,
    name="Logistic Regression Strategy", line=dict(color="#4CD97B", width=2)))

dd_fig.update_layout(
    template="plotly_dark", hovermode="x unified", height=300,
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis_title="Drawdown (%)", xaxis_title=None,
)
st.plotly_chart(dd_fig, use_container_width=True)

# Key Metrics
st.subheader("Performance Summary")

col1, col2, col3 = st.columns(3)

def show_metrics(col, label, cumulative, daily_returns, is_baseline=False):
    total_return = (cumulative.iloc[-1] - 1) * 100
    sharpe = sharpe_ratio(daily_returns)
    drawdown = max_drawdown(cumulative) * 100

    with col:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.metric("Total Return", f"{total_return:.2f}%")
            st.metric("Sharpe Ratio", f"{sharpe:.3f}")
            st.metric("Max Drawdown", f"{drawdown:.2f}%")

show_metrics(col1, "Buy & Hold", buyhold_cumulative, daily_return_test)
show_metrics(col2, "Random Forest Strategy", rf_cumulative, rf_daily_return)
show_metrics(col3, "Logistic Regression Strategy", log_cumulative, log_daily_return)

st.caption(
    "Backtested on out of sample data from 2024-07-05 to 2026-08-21. "
    "Past performance does not guarantee future results."
)

# Feature Importance Chart
st.subheader("What Each Model Relied On")

fi_col1, fi_col2 = st.columns(2)

with fi_col1:
    st.markdown("**Random Forest — Feature Importance**")
    imp_sorted = importances.sort_values(ascending=True)
    imp_fig = go.Figure(go.Bar(
        x=imp_sorted.values, y=imp_sorted.index, orientation="h",
        marker=dict(color="#FF8A5C")
    ))
    imp_fig.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Importance", yaxis_title=None,
    )
    st.plotly_chart(imp_fig, use_container_width=True)
    st.caption("Higher = more influence on the model's decisions. No direction implied.")

with fi_col2:
    st.markdown("**Logistic Regression — Coefficients**")
    coef_sorted = log_coefficients.sort_values(ascending=True)
    coef_colors = ["#FF6B6B" if v < 0 else "#4CD97B" for v in coef_sorted.values]
    coef_fig = go.Figure(go.Bar(
        x=coef_sorted.values, y=coef_sorted.index, orientation="h",
        marker=dict(color=coef_colors)
    ))
    coef_fig.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Coefficient", yaxis_title=None,
    )
    st.plotly_chart(coef_fig, use_container_width=True)
    st.caption("Green = pushes prediction toward 'up' trend. Red = pushes toward 'down' trend.")



st.caption(
    "Moving average features (sma50_vs_sma200, sma_50, sma_20, sma_200) dominated "
    "the model's decisions, while short-term signals like return_5d and volume_ratio "
    "mattered least, suggesting 5 day price direction is driven more by prevailing "
    "trend than by short-term noise."
)