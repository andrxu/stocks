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
        # If technical check returned an error, report and skip
        if 'error' in tech_result:
            print(Fore.RED + f"{sym}: {tech_result['error']}")
            continue
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
    # Build rows and compute visible widths while preserving ANSI color codes
    import re
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    def visible_len(s: str) -> int:
        return len(ansi_re.sub('', str(s)))

    def pad(s: str, width: int, right: bool = True) -> str:
        s = str(s)
        v = visible_len(s)
        if v >= width:
            return s
        pad_str = ' ' * (width - v)
        return (pad_str + s) if right else (s + pad_str)

    rows = []
    for _, row in df.iterrows():
        if "error" in row:
            rows.append({"symbol": row.get('symbol', 'N/A'), "error": f"ERROR - {row['error']}"})
            continue

        color = Fore.GREEN if row["ready_for_leap"] else Fore.YELLOW if row.get("ready_technicals") else Fore.WHITE

        # Reuse the same formatting helpers
        def fmt_num(v, decimals=2):
            try:
                if v is None:
                    return 'N/A'
                return f"{float(v):.{decimals}f}"
            except Exception:
                return 'N/A'

        def fmt_percent(v, decimals=3):
            try:
                if v is None:
                    return 'N/A'
                return f"{float(v) * 100:.{decimals}f}%"
            except Exception:
                return 'N/A'

        def fmt_cap(v):
            try:
                if v is None:
                    return 'N/A'
                return f"{int(v):,}"
            except Exception:
                return 'N/A'

        row_data = {
            "symbol": row.get('symbol', 'N/A'),
            "price": fmt_num(row.get('latest_price')),
            "ready": 'Y' if row.get('ready_for_leap') else 'N',
            "expiry": row.get('expiry', 'N/A'),
            "strike": fmt_num(row.get('strike')),
            "premium": fmt_num(row.get('premium')),
            "percent": fmt_num(row.get('percent')),
            "/day": fmt_num(row.get('per_day')),
            "iv": fmt_num(row.get('iv')),
            "delta": row.get('delta', 'N/A'),
            "return": fmt_num(row.get('expected_return')) + '%',
            "rsi": fmt_num(row.get('rsi')),
            "pe": fmt_num(row.get('pe_ratio')),
            "revgr": fmt_percent(row.get('revenue_growth')),
            "margin": fmt_num(row.get('profit_margin')),
            "de": fmt_num(row.get('debt_to_equity')),
            "cap": fmt_cap(row.get('market_cap')),
            "color": color
        }
        rows.append(row_data)

    if not rows:
        print("No results to show.")
        return

    headers = ["Symbol", "Price", "Ready", "Expiry", "Strike", "Premium", "%", "/day", "IV", "Delta", "Return", "RSI", "PE", "RevGr", "Margin", "D/E", "Cap"]

    # Compute visible widths
    cols = {h: [] for h in headers}
    for r in rows:
        # Skip error-only rows for column width calculation
        if 'error' in r:
            continue
        cols['Symbol'].append(r.get('symbol', 'N/A'))
        cols['Price'].append(r.get('price', 'N/A'))
        cols['Ready'].append(r.get('ready', 'N'))
        cols['Expiry'].append(r.get('expiry', 'N/A'))
        cols['Strike'].append(r.get('strike', 'N/A'))
        cols['Premium'].append(r.get('premium', 'N/A'))
        cols['%'].append(r.get('percent', 'N/A'))
        cols['/day'].append(r.get('/day', 'N/A'))
        cols['IV'].append(r.get('iv', 'N/A'))
        cols['Delta'].append(r.get('delta', 'N/A'))
        cols['Return'].append(r.get('return', 'N/A'))
        cols['RSI'].append(r.get('rsi', 'N/A'))
        cols['PE'].append(r.get('pe', 'N/A'))
        cols['RevGr'].append(r.get('revgr', 'N/A'))
        cols['Margin'].append(r.get('margin', 'N/A'))
        cols['D/E'].append(r.get('de', 'N/A'))
        cols['Cap'].append(r.get('cap', 'N/A'))

    widths = {}
    for h in headers:
        sample = [h] + cols[h]
        widths[h] = max(visible_len(str(x)) for x in sample)

    # Print header (right-align numeric headers to match column alignment)
    right_align = set(["Price", "Ready", "Strike", "Premium", "%", "/day", "IV", "Delta", "Return", "RSI", "PE", "RevGr", "Margin", "D/E", "Cap"])
    header_line = '  '.join((h.rjust(widths[h]) if h in right_align else h.ljust(widths[h])) for h in headers)
    print(header_line)
    print('-' * visible_len(header_line))

    # Print rows with color per row
    for r in rows:
        if 'error' in r:
            print(Fore.RED + f"{r['symbol']}: {r['error']}")
            continue
        color = r['color']
        parts = [
            r['symbol'].ljust(widths['Symbol']),
            r['price'].rjust(widths['Price']),
            r['ready'].center(widths['Ready']),
            str(r['expiry']).ljust(widths['Expiry']),
            r['strike'].rjust(widths['Strike']),
            r['premium'].rjust(widths['Premium']),
            r['percent'].rjust(widths['%']),
            r['/day'].rjust(widths['/day']),
            r['iv'].rjust(widths['IV']),
            str(r['delta']).rjust(widths['Delta']),
            r['return'].rjust(widths['Return']),
            r['rsi'].rjust(widths['RSI']),
            r['pe'].rjust(widths['PE']),
            r['revgr'].rjust(widths['RevGr']),
            r['margin'].rjust(widths['Margin']),
            r['de'].rjust(widths['D/E']),
            r['cap'].rjust(widths['Cap']),
        ]
        print(color + '  '.join(parts) + Style.RESET_ALL)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Short description of purpose and rules
    print("LEAP readiness scanner")
    print("Goal: identify long-term LEAP candidates that combine healthy fundamentals with a favorable long-term technical setup and reasonable option metrics.")
    print("Rules summary: positive long-term trend (MA50>MA200), price above key MAs, reasonable RSI (<85), timing via pullback to MA50 or pullback from recent highs, and fundamental filters: PE not excessive, positive rev growth, margin >= 10%, D/E reasonable, market cap >= $1B.")
    print("")

    # Load tickers from tickers.yml (required)
    import os, sys
    try:
        import yaml
    except Exception:
        print("Missing dependency: PyYAML is required to load tickers.yml. Install with: pip install PyYAML")
        sys.exit(2)

    yml_path = os.path.join(os.path.dirname(__file__), 'tickers.yml')
    if not os.path.exists(yml_path):
        print(f"Missing configuration file: {yml_path}")
        sys.exit(2)

    with open(yml_path, 'r') as f:
        data = yaml.safe_load(f)

    if not (isinstance(data, dict) and 'tickers' in data and isinstance(data['tickers'], list)):
        print('Invalid tickers.yml: expected a mapping with a "tickers" list')
        sys.exit(2)

    tickers = [str(x).strip() for x in data['tickers'] if x is not None and str(x).strip()]

    df, latest_vix = screen_stocks(sorted(tickers))
    print_colored_results(df)
