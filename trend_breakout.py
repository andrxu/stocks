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

# Weights for each signal to compute a final score (0-100). Adjust these to tune
# how strict the scoring is. Total should sum to 100 for intuitive percent scores.
SIGNAL_WEIGHTS = {
    "above_mas": 20,
    "volume_confirm": 15,
    "rsi_ok": 10,
    "ma50_trending_up": 10,
    "early_momentum": 10,
    "golden_cross": 15,
    "breakout": 20,
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

        # Compute weighted score (0-100)
        signals = {
            "above_mas": bool(above_mas),
            "volume_confirm": bool(volume_confirm),
            "rsi_ok": bool(rsi_ok),
            "ma50_trending_up": bool(ma50_trending_up),
            "early_momentum": bool(early_momentum),
            "golden_cross": bool(golden_cross),
            "breakout": bool(breakout),
        }

        total_weight = sum(SIGNAL_WEIGHTS.values())
        score_raw = sum(SIGNAL_WEIGHTS[k] for k, v in signals.items() if v)
        score = int(round((score_raw / total_weight) * 100)) if total_weight > 0 else 0

        # Determine readiness using a threshold; tune this threshold to be more/less strict
        READY_THRESHOLD = 70
        ready_momentum = score >= READY_THRESHOLD

        result = {
            "symbol": symbol,
            "ready_momentum": ready_momentum,
            "score": score,
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

        return result

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

    # Color coding by numeric score (if present) to provide more granularity
    score = row.get("score")
    if isinstance(score, int):
        if score >= 80:
            color = Fore.GREEN
        elif score >= 60:
            color = Fore.YELLOW
        elif score >= 40:
            color = Fore.CYAN
        else:
            color = Fore.WHITE
    else:
        # Fallback to previous behavior
        if row.get("ready_momentum"):
            color = Fore.GREEN
        elif row.get("above_mas", False):
            color = Fore.YELLOW
        else:
            color = Fore.WHITE

    print(color + (
        f"{row['symbol']}: "
        f"Score: {score if score is not None else 'N/A'}, "
        f"MomentumReady: {row.get('ready_momentum')}, "
        f"AboveMA: {row.get('above_mas')}, "
        f"VolConfirm: {row.get('volume_confirm')}, "
        f"RSI_OK: {row.get('rsi_ok')}, "
        f"MA50Up: {row.get('ma50_trending_up')}, "
        f"Breakout: {row.get('breakout')}, "
        f"EarlyMom: {row.get('early_momentum')}, "
        f"GoldenCross: {row.get('golden_cross')}, "
        f"Price: {row.get('latest_price')}, "
        f"MA20: {row.get('ma20')}, MA50: {row.get('ma50')}, MA200: {row.get('ma200')}, "
        f"RSI: {row.get('rsi')}"
    ) + Style.RESET_ALL, flush=True)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    tickers = [
        "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "NFLX", "AVGO", "ELF", "CELH","NET",
        "INTC", "AMD", "PYPL", "ADBE", "CRM", "ORCL", "SOFI", "UBER",
        "COIN", "HOOD", "GRAB", "JPM", "V", "CRWD", "TSM", "SNOW", "WMT", "HIMS", "ASTS", "OKLO", "TEM", "AFRM", "GS", "VOO", "COST", "VGT", "IWY", "IYW",
        "CRSP", "PLTR", "SHOP", "PINS", "DDOG", "MRNA","MELI", "CRCL", "GTLB", "QBTS", "QUBT", "CRWV", "IONQ", "APP", "UPST", "AI", "CVNA", "SMCI", 
    ]

    all_rows = []
    for sym in sorted(tickers):
        row = check_trend_breakout(sym)
        all_rows.append(row)

    # Sort by score descending (missing scores go last), then by symbol ascending
    def sort_key(r):
        s = r.get('score')
        # Use -1 for missing/NA to send them to the bottom
        return (-(s if isinstance(s, int) else -1), r.get('symbol', ''))

    all_rows_sorted = sorted(all_rows, key=sort_key)
    for row in all_rows_sorted:
        print_signals_one_line(row)

