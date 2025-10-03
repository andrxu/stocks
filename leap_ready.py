import yfinance as yf
import pandas as pd
from colorama import Fore, Style, init
from datetime import datetime, timedelta
from math import log, sqrt, exp, erf

init(autoreset=True)

# IV gating threshold (only mark ready when option IV is <= this)
# IV_THRESHOLD is expressed as a percentage (e.g., 40.0 means 40%)
IV_THRESHOLD = 40.0

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

    # Long-term trend filters
    long_trend_up = latest["MA50"] > latest["MA200"]
    rsi_ok = latest["RSI"] <= 40  # avoid extreme overbought

    # # Timing / entry filters
    # pullback_to_ma50 = latest["Close"] <= latest["MA50"] * 1.03  # within 3% above MA50
    # pullback_from_recent_high = latest["Close"] <= df["Close"].rolling(50).max().iloc[-1] * 0.9
    # timing_ok = pullback_to_ma50 or pullback_from_recent_high

    ready_technicals = all([rsi_ok, vix_ok, long_trend_up]) ## and timing_ok

    return {
        "symbol": symbol,
        "ready_technicals": ready_technicals,
        "rsi_ok": rsi_ok,
        "long_trend_up": long_trend_up,
        "vix_ok": vix_ok,
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
        # yfinance usually returns impliedVolatility as a decimal (e.g., 0.20 for 20%).
        # But some rows contain tiny placeholder values (1e-05) or NaNs when market data is sparse.
        # We'll try yfinance first, but fall back to solving for implied vol from the market premium
        # using a lightweight Black-Scholes bisection solver when the returned IV is implausibly small.
        raw_iv = atm_call.get("impliedVolatility", None)
        iv = None
        if raw_iv is not None:
            try:
                iv = float(raw_iv) * 100.0
            except Exception:
                iv = None

        # Helper: normal CDF
        def _norm_cdf(x):
            return 0.5 * (1.0 + erf(x / sqrt(2.0)))

        # Black-Scholes call price
        def _bs_call_price(S, K, T, r, sigma):
            if T <= 0 or sigma <= 0:
                return max(0.0, S - K * exp(-r * T))
            d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)
            return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)

        # Implied vol via bisection
        def _implied_vol_from_price(mkt_price, S, K, T, r):
            if mkt_price <= 0 or T <= 0:
                return None
            lo, hi = 1e-6, 5.0
            p_lo = _bs_call_price(S, K, T, r, lo)
            p_hi = _bs_call_price(S, K, T, r, hi)
            # If market price is outside [p_lo, p_hi], bisection will fail; guard against that
            if mkt_price < p_lo or mkt_price > p_hi:
                # try expanding hi
                hi = 10.0
                p_hi = _bs_call_price(S, K, T, r, hi)
                if mkt_price > p_hi:
                    return None
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                p_mid = _bs_call_price(S, K, T, r, mid)
                if abs(p_mid - mkt_price) < 1e-4:
                    return mid
                if p_mid > mkt_price:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi)
        delta = atm_call.get("delta", "N/A")
        expected_return = 100 * (strike - price + premium) / premium if premium > 0 else 0

        # days to expiry in years for BS solver
        T_years = max(days_to_expiry / 365.0, 0.0)

        # If yfinance returned an implausibly small IV (or none), attempt to compute it from price
        computed_iv = None
        try:
            if (iv is None) or (iv < 0.5):
                # require a sensible premium to compute implied vol
                if premium and premium > 0.01 and T_years > 0:
                    # use a small risk-free rate guess (3%)
                    r = 0.03
                    vol_frac = _implied_vol_from_price(float(premium), float(price), float(strike), T_years, r)
                    if vol_frac is not None:
                        computed_iv = float(vol_frac) * 100.0
        except Exception:
            computed_iv = None

        if computed_iv is not None and (iv is None or computed_iv > iv):
            iv = computed_iv

        if iv is None:
            iv = 0.0
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
        # preliminary readiness based on technicals + fundamentals (final readiness will include IV)
        tech_result["ready_for_leap"] = tech_result["ready_technicals"] and fundamentals["fundamentals_ok"]
        # initialize notes list to collect skip/reject reasons
        tech_result.setdefault('notes', [])

        option_metrics = get_leap_option_metrics(sym, selected_expiry)
        tech_result.update(option_metrics)

        # Enforce IV gating: only ready if option IV is available and <= IV_THRESHOLD
        iv_val = option_metrics.get('iv') if isinstance(option_metrics, dict) else None
        iv_ok = False
        if iv_val is not None:
            try:
                iv_ok = float(iv_val) <= IV_THRESHOLD
            except Exception:
                iv_ok = False

        tech_result['iv_ok'] = iv_ok
        # collect reasons
        if not tech_result.get('ready_technicals'):
            # which technical checks failed? include booleans available from check_leap_candidate
            failed = []
            if not tech_result.get('long_trend_up', True):
                failed.append('trend_down')
            if not tech_result.get('rsi_ok', True):
                failed.append('rsi_bad')
            if not tech_result.get('vix_ok', True):
                failed.append('vix_high')
            tech_result['notes'].append('tech:' + ','.join(failed) if failed else 'tech:fail')
        if not fundamentals.get('fundamentals_ok'):
            # list basic fundamental failures
            ffail = []
            if fundamentals.get('pe_ratio') is not None and fundamentals.get('pe_ratio') > 50:
                ffail.append('pe_high')
            if fundamentals.get('revenue_growth') is not None and fundamentals.get('revenue_growth') < 0:
                ffail.append('rev_down')
            if fundamentals.get('profit_margin') is not None and fundamentals.get('profit_margin') < 0.1:
                ffail.append('low_margin')
            if fundamentals.get('debt_to_equity') is not None and fundamentals.get('debt_to_equity') > 200:
                ffail.append('de_high')
            if fundamentals.get('market_cap') is not None and fundamentals.get('market_cap') < 1e9:
                ffail.append('small_cap')
            tech_result['notes'].append('fund:' + ','.join(ffail) if ffail else 'fund:fail')
        if not iv_ok:
            tech_result['notes'].append('iv:high_or_missing')

        # final readiness requires iv_ok as well
        tech_result['ready_for_leap'] = bool(tech_result.get('ready_technicals') and fundamentals.get('fundamentals_ok') and iv_ok)
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

        # Build human-readable notes string
        notes_str = ' ; '.join([str(x) for x in (row.get('notes') or [])]) or ''

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
            "notes": notes_str,
            "color": color
        }
        rows.append(row_data)

    if not rows:
        print("No results to show.")
        return

    headers = ["Symbol", "Price", "Ready", "Expiry", "Strike", "Premium", "%", "/day", "IV", "Delta", "Return", "RSI", "PE", "RevGr", "Margin", "D/E", "Cap", "Notes"]

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
    cols['Notes'].append(r.get('notes', ''))

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
            r['notes'].ljust(widths['Notes']),
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
