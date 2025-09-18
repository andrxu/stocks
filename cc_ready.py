import yfinance as yf
import pandas as pd

# -----------------------------
# Color output
# -----------------------------
def color_text(text, color):
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "reset": "\033[0m"
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

# -----------------------------
# Helper: make timestamp tz-naive safely
# -----------------------------
def make_tz_naive(ts):
    if isinstance(ts, pd.Timestamp) and ts.tz is not None:
        return ts.tz_localize(None)
    if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
        return ts.replace(tzinfo=None)
    return ts

# -----------------------------
# Covered Call Scanner
# -----------------------------
def scan_covered_calls(symbols, expiries_days=[7, 14], min_premium_percent=0.5, max_delta=0.3, min_iv=0.2, min_recent_return=0.05):
    results = []

    for sym in symbols:
        try:
            stock = yf.Ticker(sym)
            hist = stock.history(period="1mo")
            if hist.empty or len(hist) < 10:
                continue

            latest = hist.iloc[-1]

            recent_return = (latest["Close"] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]
            if recent_return < min_recent_return:
                continue

            # RSI 14
            delta_price = hist["Close"].diff()
            gain = delta_price.clip(lower=0)
            loss = -delta_price.clip(upper=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_latest = rsi.iloc[-1]

            # Overbought + slowing momentum
            overbought = rsi_latest > 65
            recent_slope = (latest["Close"] - hist["Close"].iloc[-5]) / hist["Close"].iloc[-5]
            momentum_slowing = recent_slope < 0.02

            if not (overbought and momentum_slowing):
                continue

            if not stock.options:
                continue

            for d in expiries_days:
                expiry = min(stock.options, key=lambda x: abs((pd.to_datetime(x) - make_tz_naive(pd.to_datetime(latest.name))).days - d))
                opt_chain = stock.option_chain(expiry)
                calls = opt_chain.calls

                otm_calls = calls[calls['strike'] > latest['Close']].copy()  # <-- make copy to avoid SettingWithCopyWarning
                if otm_calls.empty:
                    continue

                # Implied volatility
                if 'impliedVolatility' in otm_calls.columns:
                    otm_calls.loc[:, 'iv'] = otm_calls['impliedVolatility']
                else:
                    otm_calls.loc[:, 'iv'] = 0.2

                otm_calls.loc[:, 'premium_percent'] = otm_calls['lastPrice'] / latest['Close'] * 100

                otm_calls_filtered = otm_calls[otm_calls['iv'] >= min_iv]
                if 'delta' in otm_calls_filtered.columns:
                    otm_calls_filtered = otm_calls_filtered[otm_calls_filtered['delta'] <= max_delta]

                if otm_calls_filtered.empty:
                    continue

                best_call = otm_calls_filtered.loc[otm_calls_filtered['premium_percent'].idxmax()]

                expiry_dt = pd.to_datetime(expiry)
                latest_dt = make_tz_naive(pd.to_datetime(latest.name))
                days_to_expiry_actual = (expiry_dt - latest_dt).days
                premium_per_day = best_call['premium_percent'] / days_to_expiry_actual if days_to_expiry_actual > 0 else 0

                # Color coding
                if best_call['premium_percent'] >= 1.0 and best_call.get('delta', 0.2) <= 0.2:
                    color = "green"
                elif best_call['premium_percent'] >= 0.7:
                    color = "yellow"
                else:
                    color = "red"

                results.append({
                    "symbol": sym,
                    "stock_price": round(latest["Close"], 2),
                    "expiry": expiry,
                    "call_strike": best_call['strike'],
                    "premium": best_call['lastPrice'],
                    "premium_percent": round(best_call['premium_percent'], 2),
                    "premium_per_day": round(premium_per_day, 2),
                    "iv": round(best_call['iv'], 2),
                    "delta": best_call.get('delta', 'N/A'),
                    "recent_return": round(recent_return*100, 2),
                    "rsi": round(rsi_latest, 2),
                    "color": color
                })

        except Exception as e:
            print(f"Error processing {sym}: {e}")

    df = pd.DataFrame(results)

    if not df.empty:
        # Keep only the row with the highest premium_percent per symbol
        df = df.loc[df.groupby('symbol')['premium_percent'].idxmax()]
        df = df.sort_values(by="premium_percent", ascending=False)

    for col in ['premium_percent', 'call_strike', 'expiry', 'stock_price']:
        if col not in df.columns:
            df[col] = None

    df = df.dropna(subset=['premium_percent', 'call_strike', 'expiry', 'stock_price'])
    df = df.sort_values(by="premium_percent", ascending=False)
    return df

# -----------------------------
# Run scanner
# -----------------------------
if __name__ == "__main__":
    tickers = [
        "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "NFLX","AVGO", "CRSP",
               "INTC", "AMD", "PYPL", "ADBE", "CRM", "ORCL", "SOFI", "UBER",
               "HOOD", "GRAB", "JPM", "V", "CRWD", "TSM", "SNOW",  "QQQ", "PLTR"
    ]

    df = scan_covered_calls(
        tickers, expiries_days=[7, 14],
        min_premium_percent=0.5, max_delta=0.3, min_iv=0.2,
        min_recent_return=0.05
    )

    if df.empty:
        print("No suitable covered calls found.")
    else:
        for _, row in df.iterrows():
            line = (f"{row['symbol']:6} | Price: {row['stock_price']:>6} | Expiry: {row['expiry']} | "
                    f"Strike: {row['call_strike']:>6} | Premium: {row['premium']:>5} "
                    f"({row['premium_percent']}%) | /day: {row['premium_per_day']} | IV: {row['iv']} | Delta: {row['delta']} | "
                    f"Return: {row['recent_return']}% | RSI: {row['rsi']}")
            print(color_text(line, row['color']))

