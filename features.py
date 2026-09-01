import pandas as pd

# Because the CSV has two header rows (Price and Ticker), this tells pandas
# to spand over two rows

data = pd.read_csv("voo_data.csv", header = [0, 1], index_col = 0, parse_dates = True)

# Taking away additional two-row headers (ex: 'Close' + 'VOO' now 'Close')
data.columns = data.columns.get_level_values(0)

# Feature 1: Recent Stock Return
# "Return" = % change in price with percentage change (N) looking back at N rows
# Will calculate the difference in % from N days ago to most recent date (only looking backwards)
data["return_5d"] = data["Close"].pct_change(5)
data["return_20d"] = data["Close"].pct_change(20)
data["return_60d"] = data["Close"].pct_change(60)
data["return_252d"] = data["Close"].pct_change(252)

print(data[["Close", "return_5d", "return_20d", "return_60d", "return_252d"]].tail(10))

# Feature 2: Moving Average
# Just a Simple Moving Average, which is average closing price over last N days
# Going by month, quarter, and institutional year
data["sma_20"] = data["Close"].rolling(window=20).mean()
data["sma_50"] = data["Close"].rolling(window=50).mean()
data["sma_200"] = data["Close"].rolling(window=200).mean()

# Asking: how far is price from its own trend line? (short term)
# Positive = price is ABOVE its average (therefore in an uptrend)
# Negative = price is BELOW its average (downtrend)
data["price_vs_sma20"] = (data["Close"] - data["sma_20"]) / data["sma_20"]

# Indication of bull or bear market
# Positive = 50-day is above the 200-day (bullish), Negative = 50-day is below (bearish)
data["sma50_vs_sma200"] = (data["sma_50"] - data["sma_200"]) / data["sma_200"]

print(data[["Close", "sma_20", "sma_50", "sma_200", "price_vs_sma20", "sma50_vs_sma200"]].tail(10))

# Feature 3: Trading Volume
# Raw volume in itself is noisy data. Doesn't really say much because it will increase/decrease
# Instead, comparing today's colume to its recent average, telling us about any unusual activity
data["volume_sma_20"] = data["Volume"].rolling(window=20).mean()

# Ratio > 1 = today's volume is above most recent normal (higher conviction)
# Ratio < 1 = today's volume is below normal (low conviction / quiet day)
data["volume_ratio"] = data["Volume"] / data["volume_sma_20"]

print(data[["Volume", "volume_sma_20", "volume_ratio"]].tail(10))

# Feature 4: Volatility
# Get daily returns from % change
data["daily_return"] = data["Close"].pct_change(1)

# Volatility = std of daily returns over a rolling window
# Std measures how spread values are from average, regardless of direction
data["volatility_20d"] = data["daily_return"].rolling(window=20).std()

print(data[["Close", "daily_return", "volatility_20d"]].tail(10))

# Feature 5: Relative Strength Index (RSI)
# Separates price changes into up and down days, then compares avg size of
# recent gains to avg size of recent losses
delta = data["Close"].diff()

# keeps up/down day changes, but zeroes out down days
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
data["rsi_14"] = 100 - (100 / (1 + rs))

print(data[["Close", "rsi_14"]].tail(15))

# First 200 days of 2015 should be NaN, so this will drop any features still NaN
data_clean = data.dropna()

print(f"Original rows: {len(data)}")
print(f"Rows after dropping NaN: {len(data_clean)}")

# Now, we will look FOWARD, determining if price will be higher in 5 trading days from now?
future_price = data_clean["Close"].shift(-5)

# Comparing first but casting to a float value so NaN values can move through
target = (future_price > data_clean["Close"]).astype(float)
target[future_price.isna()] = float("nan")

# Assigning back on an explicit copy
data_clean = data_clean.copy()
data_clean["target"] = target

print(data_clean[["Close", "target"]].tail(15))

# Drop rows with unknown price targets
data_final = data_clean.dropna(subset=["target"])
data_final = data_final.copy()
data_final["target"] = data_final["target"].astype(int)

print(f"Final dataset ready for modeling: {len(data_final)} rows")

# Train/test split. Making sure to do chronologically w/o shuffling
feature_cols = [
    "return_5d", "return_20d", "return_60d", "return_252d",
    "sma_20", "sma_50", "sma_200", "price_vs_sma20", "sma50_vs_sma200",
    "volume_ratio", "volatility_20d", "rsi_14"
]

X = data_final[feature_cols] # inputs
y = data_final["target"]     # outputs

split_index = int(len(data_final) * 0.8) #80% train, 20% test

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Training rows: {len(X_train)} (earliest data)")
print(f"Testing rows: {len(X_test)} (most recent, held out)")
print(f"Training date range: {X_train.index.min()} to {X_train.index.max()}")
print(f"Testing date range: {X_test.index.min()} to {X_test.index.max()}")