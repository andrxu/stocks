import yfinance as yf
import pandas as pd
from colorama import Fore, Style, init
from datetime import datetime, timedelta

init(autoreset=True)

# -----------------------------
# Technical Analysis (Long-term + Timing)
# -----------------------------
def check_leap_candidate(symbol: str, vix_ok: bool):
    df = yf.download(symbol, period="2y", interval="1d", auto_adjust=True, progress=False)

    if df.empty:
        return {"symbol": symbol, "error": "No data returned"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    if "Close" not in df.columns or "Volume" not in df.columns:
        return {"symbol": symbol, "error": f"Missing columns: {df.columns.tolist()}"}

    # Moving averages
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["Vol50"] = df["Volume"].rolling(50).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df = df.dropna(subset=["MA50", "MA200", "RSI", "Vol50"])
    if df.empty:
        return {"symbol": symbol, "error": "Not enough history"}

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Long-term trend filters
    golden_cross = (prev["MA50"] < prev["MA200"]) and (latest["MA50"] >= latest["MA200"])
    long_trend_up = latest["MA50"] > latest["MA200"]
    above_mas = latest["Close"] > latest["MA50"] and latest["Close"] > latest["MA200"]
    volume_ok = latest["Volume"] > latest["Vol50"]
    rsi_ok = latest["RSI"] < 85  # avoid extreme overbought

    # Timing / entry filters
    pullback_to_ma50 = latest["Close"] <= latest["MA50"] * 1.03  # within 3% above MA50
    pullback_from_recent_high = latest["Close"] <= df["Close"].rolling(50).max().iloc[-1] * 0.9
    timing_ok = pullback_to_ma50 or pullback_from_recent_high

    ready_technicals = all([above_mas, volume_ok, rsi_ok, vix_ok, long_trend_up]) and timing_ok

    return {
        "symbol": symbol,
        "ready_technicals": ready_technicals,
        "golden_cross_today": golden_cross,
        "above_mas": above_mas,
        "volume_ok": volume_ok,
        "rsi_ok": rsi_ok,
        "long_trend_up": long_trend_up,
        "vix_ok": vix_ok,
        "timing_ok": timing_ok,
        "latest_price": round(float(latest["Close"]), 2),
        "ma50": round(float(latest["MA50"]), 2),
        "ma200": round(float(latest["MA200"]), 2),
        "rsi": round(float(latest["RSI"]), 2)
    }

# -----------------------------
# Fundamentals (Improved LEAP filters without ticker.info)
# -----------------------------
def add_fundamentals(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.get_info()  # safer than ticker.info

        pe_ratio = info.get("forwardPE")
        peg_ratio = info.get("pegRatio")
        revenue_growth = info.get("revenueGrowth") or info.get("revenue_growth")
        profit_margin = info.get("profitMargins") or info.get("netMargins")
        debt_to_equity = info.get("debtToEquity")
        market_cap = info.get("marketCap")

        fundamentals_ok = True
        if pe_ratio is not None and pe_ratio > 50:
            fundamentals_ok = False
        if peg_ratio is not None and peg_ratio > 2:
            fundamentals_ok = False
        if revenue_growth is not None and revenue_growth < 0:
            fundamentals_ok = False
        if profit_margin is not None and profit_margin < 0.1:
            fundamentals_ok = False
        if debt_to_equity is not None and debt_to_equity > 200:
            fundamentals_ok = False
        if market_cap is not None and market_cap < 1e9:
            fundamentals_ok = False

        return {
            "pe_ratio": pe_ratio,
            "peg_ratio": peg_ratio,
            "revenue_growth": revenue_growth,
            "profit_margin": profit_margin,
            "debt_to_equity": debt_to_equity,
            "market_cap": market_cap,
            "fundamentals_ok": fundamentals_ok
        }

    except Exception as e:
        return {"error": f"Fundamentals fetch failed: {e}", "fundamentals_ok": False}


# -----------------------------
# LEAP Expiry Helper
# -----------------------------
def get_longest_expiry(symbol: str):
    """Return the furthest available expiry from Yahoo (usually a LEAP)."""
    try:
        ticker = yf.Ticker(symbol)
        expiries = ticker.options
        if not expiries:
            return None
        return expiries[-1]  # select the last expiry = longest
    except Exception as e:
        print(f"Error fetching expiries for {symbol}: {e}")
        return None

# -----------------------------
# Option Metrics
# -----------------------------
def get_leap_option_metrics(symbol: str, expiry: str):
    ticker = yf.Ticker(symbol)
    try:
        calls = ticker.option_chain(expiry).calls
        calls = calls.sort_values(by="strike")
        price = ticker.history(period="1d")["Close"].iloc[-1]

        # ATM call
        calls["diff"] = abs(calls["strike"] - price)
        atm_call = calls.iloc[calls["diff"].argmin()]

        strike = atm_call["strike"]
        premium = atm_call["lastPrice"]
        percent = 100 * premium / price
        days_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days
        per_day = premium / days_to_expiry if days_to_expiry > 0 else 0
        iv = atm_call.get("impliedVolatility", 0)
        delta = atm_call.get("delta", "N/A")
        expected_return = 100 * (strike - price + premium) / premium if premium > 0 else 0

        return {
            "expiry": expiry,
            "strike": strike,
            "premium": round(premium, 2),
            "percent": round(percent, 2),
            "per_day": round(per_day, 2),
            "iv": round(iv, 2),
            "delta": delta,
            "expected_return": round(expected_return, 2)
        }

    except Exception as e:
        return {"expiry": expiry, "error": f"Option fetch failed: {e}"}

# -----------------------------
# Screening
# -----------------------------
def screen_stocks(symbols):
    vix_data = yf.download("^VIX", period="2mo", interval="1d", auto_adjust=True, progress=False)
    latest_vix = None
    vix_ok = True
    if vix_data.empty:
        print("Failed to fetch VIX data.")
        vix_ok = False
    else:
        latest_vix = float(vix_data["Close"].iloc[-1].item())
        vix_ok = latest_vix < 30
        print(f"Latest VIX: {latest_vix:.2f}, VIX OK: {vix_ok}")

    results = []
    for sym in symbols:
        selected_expiry = get_longest_expiry(sym)
        if not selected_expiry:
            print(Fore.RED + f"{sym}: No option expiries found.")
            continue

        tech_result = check_leap_candidate(sym, vix_ok)
        fundamentals = add_fundamentals(sym)
        tech_result.update(fundamentals)
        tech_result["ready_for_leap"] = tech_result["ready_technicals"] and fundamentals["fundamentals_ok"]

        option_metrics = get_leap_option_metrics(sym, selected_expiry)
        tech_result.update(option_metrics)
        results.append(tech_result)

    return pd.DataFrame(results), latest_vix

# -----------------------------
# Colored Output
# -----------------------------
def print_colored_results(df: pd.DataFrame):
    for _, row in df.iterrows():
        if "error" in row:
            print(Fore.RED + f"{row['symbol']}: ERROR - {row['error']}")
            continue

        color = Fore.GREEN if row["ready_for_leap"] else Fore.YELLOW if row["ready_technicals"] else Fore.WHITE

        fundamentals_summary = (
            f"PE: {row.get('pe_ratio', 'N/A')}, "
            f"RevGr: {row.get('revenue_growth', 'N/A')}, "
            f"Margin: {row.get('profit_margin', 'N/A')}, "
            f"D/E: {row.get('debt_to_equity', 'N/A')}, "
            f"Cap: {row.get('market_cap', 'N/A')}"
        )

        option_summary = (
            f"Expiry: {row.get('expiry', 'N/A')}, "
            f"Strike: {row.get('strike', 'N/A')}, "
            f"Premium: {row.get('premium', 'N/A')} ({row.get('percent', 'N/A')}%), "
            f"/day: {row.get('per_day', 'N/A')}, "
            f"IV: {row.get('iv', 'N/A')}, "
            f"Delta: {row.get('delta', 'N/A')}, "
            f"Return: {row.get('expected_return', 'N/A')}%"
        )

        print(color + (
            f"{row['symbol']} | Price: {row.get('latest_price', 'N/A')} | "
            f"Ready: {row['ready_for_leap']} | {option_summary} | RSI: {row.get('rsi', 'N/A')} | "
            + fundamentals_summary + Style.RESET_ALL
        ))

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    tickers = [
        "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "NFLX", "AVGO", "CRSP", "GS",
        "INTC", "AMD", "PYPL", "ADBE", "CRM", "ORCL", "SOFI", "UBER",
        "HOOD", "GRAB", "JPM", "V", "CRWD", "TSM", "SNOW", "QQQ", "PLTR"
    ]

    df, latest_vix = screen_stocks(sorted(tickers))
    print_colored_results(df)
