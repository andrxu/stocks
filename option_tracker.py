import yfinance as yf
import pandas as pd
from colorama import Fore, Style, init
from datetime import datetime

# Initialize colorama
init(autoreset=True)

# -----------------------------
# Define options you want to track
# -----------------------------
# Format: { "SYMBOL": [ (expiry, strike, type), ... ] }
# type: "call" or "put"
OPTIONS_TO_TRACK = {
    "NVDA": [("2026-01-16", 200, "call"), ("2026-01-16", 200, "put")],
}

# -----------------------------
# Helper: format number safely
# -----------------------------
def safe_round(value, decimals=2):
    if value is None or value == 0 or pd.isna(value):
        return None
    try:
        return round(float(value), decimals)
    except:
        return None

# -----------------------------
# Helper: fetch option premium
# -----------------------------
def fetch_option_data(symbol: str, expiry: str, strike: float, option_type: str = "call"):
    try:
        ticker = yf.Ticker(symbol)
        if expiry not in ticker.options:
            return {"symbol": symbol, "expiry": expiry, "strike": strike, "type": option_type, "error": "Expiry not available"}

        opt_chain = ticker.option_chain(expiry)
        options = opt_chain.calls if option_type.lower() == "call" else opt_chain.puts

        row = options[options["strike"] == strike]
        if row.empty:
            return {"symbol": symbol, "expiry": expiry, "strike": strike, "type": option_type, "error": "Strike not found"}

        row = row.iloc[0]

        last_price = safe_round(row.get("lastPrice"))
        bid = safe_round(row.get("bid"))
        ask = safe_round(row.get("ask"))
        mid = safe_round(((row.get("bid") or 0) + (row.get("ask") or 0)) / 2) if row.get("bid") and row.get("ask") else None
        iv = safe_round(row.get("impliedVolatility"), 3)
        oi = int(row.get("openInterest", 0)) if not pd.isna(row.get("openInterest")) else None

        return {
            "symbol": symbol,
            "expiry": expiry,
            "strike": strike,
            "type": option_type,
            "last_price": last_price,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "iv": iv,
            "open_interest": oi,
        }
    except Exception as e:
        return {"symbol": symbol, "expiry": expiry, "strike": strike, "type": option_type, "error": str(e)}

# -----------------------------
# Main loop
# -----------------------------
if __name__ == "__main__":
    results = []
    for sym, contracts in OPTIONS_TO_TRACK.items():
        for expiry, strike, option_type in contracts:
            row = fetch_option_data(sym, expiry, strike, option_type)
            results.append(row)

            if "error" in row:
                print(Fore.RED + f"{row['symbol']} {row['expiry']} {row['strike']} ({row['type']}): ERROR - {row['error']}")
            else:
                print(
                    Fore.GREEN
                    + f"{row['symbol']:5} | Exp: {row['expiry']} | Strike: {row['strike']:6} | "
                    f"Type: {row['type']:4} | Last: {row['last_price'] or 'N/A'} | "
                    f"Bid: {row['bid'] or 'N/A'} | Ask: {row['ask'] or 'N/A'} | "
                    f"Mid: {row['mid'] or 'N/A'} | IV: {row['iv'] or 'N/A'} | OI: {row['open_interest'] or 'N/A'}"
                )

    # Save to CSV for history with timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"options_tracker_output_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(Style.BRIGHT + Fore.YELLOW + f"\nSaved results to {filename}")
