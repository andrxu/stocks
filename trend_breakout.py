import yfinance as yf
import pandas as pd
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# -----------------------------
# Format metrics
# -----------------------------
def format_metric(value, decimals=2):
    if value is None or value == 'N/A':
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except:
        return "N/A"

# -----------------------------
# Technical Analysis
# -----------------------------
# Plain-English descriptions for the signals and numeric fields used below.
# These can be used for inline help, documentation, or to annotate output.
SIGNAL_DESCRIPTIONS = {
    "ready_momentum": "All key conditions for a momentum breakout are met - price above MA50 & MA200, volume confirms, RSI healthy, MA50 rising, MA20>MA50, golden cross and breakout.",
    "above_mas": "Price is above both the 50-day and 200-day moving averages.",
    "volume_confirm": "Today's volume exceeds the 20-day average volume.",
    "rsi_ok": "RSI (14) is between 50 and 70 — upward momentum without overbought conditions.",
    "ma50_trending_up": "50-day moving average is higher than it was 10 trading days ago.",
    "breakout": "Today's close is above the previous 20-day high — breakout above recent range.",
    "early_momentum": "20-day moving average is above the 50-day moving average (short-term > medium-term).",
    "golden_cross": "50-day moving average is above the 200-day moving average (bullish crossover).",
    "latest_price": "Latest closing price.",
    "ma20": "20-day moving average (short-term trend level).",
    "ma50": "50-day moving average (medium-term trend level).",
    "ma200": "200-day moving average (long-term trend level).",
    "rsi": "14-day Relative Strength Index (momentum oscillator).",
}

def check_trend_breakout(symbol: str):
    try:
        df = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, progress=False)
        if df.empty:
            return {"symbol": symbol, "error": "No data returned"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        if "Close" not in df.columns or "Volume" not in df.columns:
            return {"symbol": symbol, "error": f"Missing columns: {df.columns.tolist()}"}

        # Moving averages
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df["MA200"] = df["Close"].rolling(200).mean()
        df["Vol20"] = df["Volume"].rolling(20).mean()

        # RSI (14-day)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        df = df.dropna(subset=["MA20", "MA50", "MA200", "RSI", "Vol20"])
        if df.empty:
            # Not enough history: return sensible defaults instead of an error so callers
            # (and the one-line printer) can continue processing tickers like CRWV.
            return {
                "symbol": symbol,
                "ready_momentum": False,
                "above_mas": False,
                "volume_confirm": False,
                "rsi_ok": False,
                "ma50_trending_up": False,
                "breakout": False,
                "early_momentum": False,
                "golden_cross": False,
                "latest_price": "N/A",
                "ma20": "N/A",
                "ma50": "N/A",
                "ma200": "N/A",
                "rsi": "N/A",
            }

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Technical signals
        above_mas = (latest["Close"] > latest["MA50"]) and (latest["Close"] > latest["MA200"])
        volume_confirm = latest["Volume"] > latest["Vol20"]
        rsi_ok = 50 < latest["RSI"] < 70  # trending, not overbought
        ma50_trending_up = latest["MA50"] > df["MA50"].iloc[-10]
        breakout = latest["Close"] > df["Close"].rolling(20).max().iloc[-2]
        early_momentum = latest["MA20"] > latest["MA50"]
        golden_cross = latest["MA50"] > latest["MA200"]

        # Ready for momentum breakout
        ready_momentum = all([above_mas, volume_confirm, rsi_ok, ma50_trending_up, early_momentum, golden_cross, breakout])

        return {
            "symbol": symbol,
            "ready_momentum": ready_momentum,
            "above_mas": above_mas,
            "volume_confirm": volume_confirm,
            "rsi_ok": rsi_ok,
            "ma50_trending_up": ma50_trending_up,
            "breakout": breakout,
            "early_momentum": early_momentum,
            "golden_cross": golden_cross,
            "latest_price": round(float(latest["Close"]), 2),
            "ma20": round(float(latest["MA20"]), 2),
            "ma50": round(float(latest["MA50"]), 2),
            "ma200": round(float(latest["MA200"]), 2),
            "rsi": round(float(latest["RSI"]), 2)
        }

    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

# -----------------------------
# Print one-line summary
# -----------------------------
def print_signals_one_line(row):
    error_msg = row.get("error")
    if error_msg is not None and str(error_msg) != 'nan':
        print(Fore.RED + f"{row['symbol']}: ERROR - {error_msg}")
        return

    # Color coding
    if row["ready_momentum"]:
        color = Fore.GREEN
    elif row.get("above_mas", False) :
        color = Fore.YELLOW
    else:
        color = Fore.WHITE

    print(color + (
        f"{row['symbol']}: "
        f"MomentumReady: {row['ready_momentum']}, "
        f"AboveMA: {row['above_mas']}, "
        f"VolConfirm: {row['volume_confirm']}, "
        f"RSI_OK: {row['rsi_ok']}, "
        f"MA50Up: {row['ma50_trending_up']}, "
        f"Breakout: {row['breakout']}, "
        f"EarlyMom: {row['early_momentum']}, "
        f"GoldenCross: {row['golden_cross']}, "
        f"Price: {row['latest_price']}, "
        f"MA20: {row['ma20']}, MA50: {row['ma50']}, MA200: {row['ma200']}, "
        f"RSI: {row['rsi']}"
    ) + Style.RESET_ALL)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    tickers = [
        "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "NFLX", "AVGO", "ELF", "CELH",
        "INTC", "AMD", "PYPL", "ADBE", "CRM", "ORCL", "SOFI", "UBER",
        "COIN", "HOOD", "GRAB", "JPM", "V", "CRWD", "TSM", "SNOW", "WMT", "HIMS", "ASTS", "OKLO", "TEM", "AFRM", "GS", "VOO", "COST", "VGT", "IWY", "IYW",
        "CRSP", "PLTR", "SHOP", "PINS", "DDOG", "MRNA","MELI", "CRCL", "GTLB", "QBTS", "QUBT", "CRWV", "IONQ", "APP", "UPST", "AI", "CVNA", "SMCI", 
    ]

    all_rows = []
    for sym in sorted(tickers):
        row = check_trend_breakout(sym)
        all_rows.append(row)
        print_signals_one_line(row)

