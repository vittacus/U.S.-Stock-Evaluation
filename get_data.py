import yfinance as yf

# Pulling VOO's daily historical data
ticker = "VOO"
data = yf.download(ticker, start = "2015-01-01", end = "2026-08-31")

# Saving locally
data.to_csv("voo_data.csv")

print(data.head())
print(f"\nTotal rows: {len(data)}")