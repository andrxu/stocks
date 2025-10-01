import yfinance as yf
import pandas as pd
from colorama import Fore, Style, init
import re

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
    "pe": "Price-to-Earnings ratio (trailing or forward if available).",
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

# Readiness threshold used to mark 'ready_momentum' when score >= threshold
READY_THRESHOLD = 70


def print_rules_summary():
    """Print a short summary of scoring rules and signal descriptions."""
    print("Scoring rules:", flush=True)
    print(f"  - Ready threshold: score >= {READY_THRESHOLD}", flush=True)
    print("  - Signal weights (contribute to 0-100 score):", flush=True)
    for sig, w in SIGNAL_WEIGHTS.items():
        desc = SIGNAL_DESCRIPTIONS.get(sig, "")
        # Print the description in white for readability
        print(Fore.WHITE + f"    - {sig}: weight={w} -> {desc}", flush=True)
    print("  - Numeric fields: latest_price, ma20, ma50, ma200, rsi", flush=True)
    print("", flush=True)


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

        # Attempt to get P/E ratio from yfinance Ticker.info
        pe_value = "N/A"
        try:
            t = yf.Ticker(symbol)
            info = t.info if hasattr(t, 'info') else {}
            pe = info.get('trailingPE') or info.get('forwardPE')
            if pe is not None:
                pe_value = round(float(pe), 2)
        except Exception:
            # leave pe_value as 'N/A' on any error
            pe_value = "N/A"

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
            "rsi": round(float(latest["RSI"]), 2),
            "pe": pe_value,
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
        f", P/E: {format_metric(row.get('pe'), 2)}"
    ) + Style.RESET_ALL, flush=True)


def _bool_short(v):
    return 'Y' if v else 'N'


def print_table(rows):
    """Print rows in a vertically-aligned table for easier reading."""
    # Define columns and how to extract/format each field
    headers = [
        'Symbol', 'Score', 'Ready', 'AboveMA', 'Vol', 'RSI_OK', 'MA50Up',
        'Breakout', 'Early', 'Golden', 'Price', 'MA20', 'MA50', 'MA200', 'RSI', 'P/E'
    ]

    # ANSI escape sequence stripper to compute visible lengths for alignment
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    def visible_len(s: str):
        return len(ansi_re.sub('', str(s)))

    def pad_cell(s: str, width: int, align_right: bool = False):
        s = str(s)
        v = visible_len(s)
        if v >= width:
            return s
        pad = ' ' * (width - v)
        return (pad + s) if align_right else (s + pad)

    # Helper to pick a score color consistent with one-line output
    def score_color(score):
        if not isinstance(score, int):
            return Style.RESET_ALL
        if score >= 80:
            return Fore.GREEN
        elif score >= 60:
            return Fore.YELLOW
        elif score >= 40:
            return Fore.CYAN
        else:
            return Fore.WHITE

    table_rows = []
    for r in rows:
        # Ensure keys exist and format numbers
        score = r.get('score')
        score_raw = str(score) if isinstance(score, int) else 'N/A'
        # Color the score cell according to ranges
        sc_color = score_color(score)
        score_s = sc_color + score_raw + Style.RESET_ALL if score_raw != 'N/A' else 'N/A'

        # Color boolean flags: Y (green), N (white)
        def bool_cell(val):
            short = _bool_short(val)
            if short == 'Y':
                return Fore.GREEN + 'Y' + Style.RESET_ALL
            else:
                return Fore.WHITE + 'N' + Style.RESET_ALL

        ready = bool_cell(r.get('ready_momentum', False))
        above = bool_cell(r.get('above_mas', False))
        vol = bool_cell(r.get('volume_confirm', False))
        rsi_ok = bool_cell(r.get('rsi_ok', False))
        ma50up = bool_cell(r.get('ma50_trending_up', False))
        breakout = bool_cell(r.get('breakout', False))
        early = bool_cell(r.get('early_momentum', False))
        golden = bool_cell(r.get('golden_cross', False))

        price = format_metric(r.get('latest_price'))
        ma20 = format_metric(r.get('ma20'))
        ma50 = format_metric(r.get('ma50'))
        ma200 = format_metric(r.get('ma200'))
        rsi = format_metric(r.get('rsi'))
        pe = format_metric(r.get('pe'))

        table_rows.append([
            str(r.get('symbol', '')),
            score_s,
            ready,
            above,
            vol,
            rsi_ok,
            ma50up,
            breakout,
            early,
            golden,
            price,
            ma20,
            ma50,
            ma200,
            rsi,
            pe,
        ])

    # Compute column widths
    # Compute column widths using visible lengths (strip ANSI codes)
    all_for_width = [headers] + [[ansi_re.sub('', c) for c in row] for row in table_rows]
    cols = list(zip(*all_for_width)) if table_rows else [headers]
    widths = [max(len(str(cell)) for cell in col) for col in cols]

    # Print header
    header_line = '  '.join(pad_cell(h, w, align_right=False) for h, w in zip(headers, widths))
    print(header_line)
    print('-' * visible_len(header_line))

    # Print rows with alignment: left for symbol, right for numeric-ish
    for tr in table_rows:
        pieces = []
        for i, cell in enumerate(tr):
            # Determine alignment: symbol left, others right
            if i == 0:
                pieces.append(pad_cell(cell, widths[i], align_right=False))
            else:
                pieces.append(pad_cell(cell, widths[i], align_right=True))
        print('  '.join(pieces), flush=True)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Print rules summary before running
    print_rules_summary()

    # Require tickers.yml to exist and be valid; abort on any error to avoid
    # running with an unintended default set.
    import sys, os
    try:
        import yaml
    except Exception:
        print("Missing dependency: PyYAML is required to load tickers.yml. Install with: pip install PyYAML", flush=True)
        sys.exit(2)

    yml_path = os.path.join(os.path.dirname(__file__), "tickers.yml")
    if not os.path.exists(yml_path):
        print(f"Missing configuration file: {yml_path}", flush=True)
        sys.exit(2)

    with open(yml_path, "r") as f:
        data = yaml.safe_load(f)

    if not (isinstance(data, dict) and "tickers" in data and isinstance(data["tickers"], list)):
        print("Invalid tickers.yml: expected a mapping with a 'tickers' list", flush=True)
        sys.exit(2)

    # Clean and validate tickers
    tickers = [str(x).strip() for x in data["tickers"] if x is not None and str(x).strip()]
    if not tickers:
        print("tickers.yml contains no valid tickers", flush=True)
        sys.exit(2)

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
    # Print a vertically-aligned table for easier reading
    print_table(all_rows_sorted)

