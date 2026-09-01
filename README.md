# VOO Quant/ML Trading Strategy

A machine learning pipeline that predicts short-term price direction for VOO (Vanguard S&P 500 ETF), converts those predictions into a trading strategy, and backtests it against simply holding the ETF — measuring not just returns, but risk-adjusted performance.

**[Live Dashboard](#)** *(https://voo-quant-strategy.streamlit.app/)*

![Dashboard preview](#) *(optional: drop a screenshot of your equity curve chart here)*

---

## The Question

Can a set of well-known technical indicators (Moving Averages, Relative Strength Index, Volatility, Volume, and Momentum) meaningfully predict whether VOO will be higher 5 trading days from now? And if so, does trading on that prediction actually beat just buying and holding the ETF?

## Data

- **Source:** Yahoo Finance via the `yfinance` API
- **Range:** 2015–2026, ~2,900 trading days
- **Ticker:** VOO (chosen as a broad, diversified index ETF rather than a single stock, to reduce single-company noise)

## Features

Five feature families, each engineered to look only backward in time (no lookahead bias):

| Feature | What it captures |
|---|---|
| **Multi-timeframe return** | 5/20/60/252 day % price change: Looking at short, medium, quarterly, and yearly momentum |
| **Moving averages** | 20/50/200 day Simple Moving Averages (SMA): Adds price-vs-SMA20 and the SMA50-vs-SMA200 "golden/death cross" signal |
| **Volume ratio** | Today's volume relative to its 20 day average: Serving as a proxy for conviction behind a price move |
| **Volatility** | 20 day rolling standard deviation of daily returns: Indicates how choppy recent trading has been |
| **RSI (14-day)** | Relative Strength Index: Whether recent gains have outpaced recent losses and flags overbought/oversold conditions |

## Target

Binary classification: will VOO's closing price be higher 5 trading days from now than it is today? Labels are built by shifting price forward in time, which is necessary since forward-looking data must *only* be used to build the label, never as a feature (doing so would leak the answer into the inputs).

## Modeling

Two classifiers were trained on a **chronological** 80/20 split (2016–2024 train, 2024–2026 test). These were never randomly shuffled, since a random split would let the model "peek" at data adjacent to what it's being tested on.

- **Logistic Regression** : A linear baseline, with features being standardized first since logistic regression is sensitive to feature scale
- **Random Forest** (200 trees, max depth 5) : A non-linear model capable of capturing feature interactions a linear model can't

Both were trained with `class_weight="balanced"` to counteract class imbalance (~59% "up" days vs. ~41% "down" days in the test period), which otherwise causes a model to default to always predicting the majority class.

## Backtest

Model predictions were converted into a simple long/cash strategy: hold VOO on days the model predicts "up," sit in cash otherwise. Predictions are shifted forward one day before being acted on, since a signal generated from today's closing data can only be traded on starting tomorrow.

Three metrics were compared against buy-and-hold:
- **Total return** : raw profitability
- **Sharpe ratio** : return earned per unit of volatility (risk-adjusted performance)
- **Max drawdown** : the worst peak decline over the period

## Results

| | Buy & Hold | Random Forest | Logistic Regression |
|---|---|---|---|
| Total Return | 42.31% | 34.03% | 16.19% |
| Sharpe Ratio | 1.106 | 0.943 | **1.371** |
| Max Drawdown | -18.69% | -18.69% | **-4.48%** |
| Days Invested | 535/535 | 521/535 | 85/535 |

**Neither model beat buy-and-hold on raw return.** Random Forest predicted "up" on 97% of test days, nearly identical to always being invested, so it took on the same risk as buy-and-hold while capturing less upside, the worst outcome of the three.

**Logistic Regression told a different story.** By trading far more selectively (only 85 of 535 days), it earned less total return but achieved the *highest* Sharpe ratio and by far the lowest drawdown of any approach — including buy-and-hold. This demonstrates a real, general tradeoff in quantitative finance: total return and risk-adjusted return are not the same question, and a model that "loses" on one metric can meaningfully "win" on the other.

**Feature importance was consistent with financial intuition.** Both models leaned most heavily on moving-average-based features (particularly the golden/death cross signal, `sma50_vs_sma200`), while short-term signals like 5-day return and volume ratio mattered least — suggesting 5-day direction is driven more by prevailing trend than by short-term noise. One nuance worth flagging: Logistic Regression's `sma_200` coefficient came out slightly negative despite the feature's high overall importance — a known effect of **multicollinearity**, since `sma_20`, `sma_50`, `sma_200`, and `price_vs_sma20` are all derived from overlapping price windows and are highly correlated with each other, which can cause a linear model to assign an unintuitively-signed weight to one correlated feature even when the group as a whole matters.

## Limitations
