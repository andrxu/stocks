import yfinance as yf
import pandas as pd
from colorama import Fore, Style, init
from datetime import datetime
import os

# Initialize colorama
init(autoreset=True)

# -----------------------------
# Define options to track
# -----------------------------
# Format: { "SYMBOL": [ (expiry, strike, type), ... ] }
# type: "call" or "put"
OPTIONS_TO_TRACK = {
    "NVDA": [("2026-01-16", 200, "call"), ("2026-01-16", 200, "put")],
    "AAPL": [("2026-01-16", 200, "put")]
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
# Fetch option premium
# -----------------------------
def fetch_option_data(symbol: str, expiry: str, strike: float, option_type: str = "call"):
    try:
        ticker = yf.Ticker(symbol)
        if expiry not in ticker.options:
            return None

        opt_chain = ticker.option_chain(expiry)
        options = opt_chain.calls if option_type.lower() == "call" else opt_chain.puts
        row = options[options["strike"] == strike]

        if row.empty:
            return None

        row = row.iloc[0]
        last_price = safe_round(row.get("lastPrice"))
        return last_price
    except:
        return None

# -----------------------------
# Main loop
# -----------------------------
if __name__ == "__main__":
    folder = "__options_tracker"
    os.makedirs(folder, exist_ok=True)
    csv_file = os.path.join(folder, "options_tracker_history.csv")

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Prepare new row data
    new_data = {}
    for sym, contracts in OPTIONS_TO_TRACK.items():
        for expiry, strike, option_type in contracts:
            key = f"{sym}-{option_type.upper()}-{expiry.replace('-', '')}"
            price = fetch_option_data(sym, expiry, strike, option_type)
            new_data[key] = price
            if price is None:
                print(Fore.RED + f"{key}: PRICE NOT FOUND")
            else:
                print(Fore.GREEN + f"{key}: {price}")

    # Load existing CSV or create new
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, index_col=0)
    else:
        df = pd.DataFrame()

    # Add new column for today
    df[today_str] = pd.Series(new_data)

    # Save back to CSV
    df.to_csv(csv_file)
    print(Style.BRIGHT + Fore.YELLOW + f"\nUpdated historical tracker: {csv_file}")
