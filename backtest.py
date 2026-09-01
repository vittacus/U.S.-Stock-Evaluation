import numpy as np
import pandas as pd
from features import data_final, X_test
from train_model import rf_predictions, log_predictions

# Day over day returns for test
daily_return_test = data_final.loc[X_test.index, "daily_return"]

def sharpe_ratio(daily_returns, risk_free_rate=0.0):
    excess_returns = daily_returns - risk_free_rate / 252
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

def max_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()

def run_backtest(predictions, label):
    signal = pd.Series(predictions, index=X_test.index)
    position = signal.shift(1).fillna(0)
    strategy_daily_return = position * daily_return_test
    strategy_cumulative = (1 + strategy_daily_return).cumprod()

    total_return = (strategy_cumulative.iloc[-1] - 1) * 100
    sharpe = sharpe_ratio(strategy_daily_return)
    drawdown = max_drawdown(strategy_cumulative) * 100
    days_in_market = position.sum()

    print(f"\n{label}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Max Drawdown: {drawdown:.2f}%")
    print(f"Days in market: {int(days_in_market)} / {len(position)}")

    return strategy_cumulative

# Buy & Hold
buyhold_cumulative = (1 + daily_return_test).cumprod()
print("Buy & Hold")
print(f"Total return: {(buyhold_cumulative.iloc[-1] - 1) * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio(daily_return_test):.3f}")
print(f"Max Drawdown: {max_drawdown(buyhold_cumulative) * 100:.2f}%")
print(f"Days in market: {len(daily_return_test)} / {len(daily_return_test)}")

# Both model strategies
rf_cumulative = run_backtest(rf_predictions, "Random Forest Strategy")
log_cumulative = run_backtest(log_predictions, "Logistic Regression Strategy")