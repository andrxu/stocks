import yfinance as yf
import pandas as pd
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# -----------------------------
# Define options you want to track
# -----------------------------
# Format: { "SYMBOL": [ (expiry, strike, "call/put") ] }
options_to_track = {
    "AAPL": [("2027-01-15", 220, "call")],
    "MSFT": [("2027-01-15", 500, "call")],
    "INTC": [("2027-01-15", 25, "call")],
    "NVDA": [("2027-01-15", 180, "call")],
}

# -----------------------------
# Color output function
# -----------------------------
def color_text(text, color):
    colors = {
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "red": Fore.RED,
        "reset": Style.RESET_ALL
    }
    return f"{colors.get(color, '')}{text}{Style.RESET_ALL}"

# -----------------------------
# IV Rank Helpers
# -----------------------------
def calculate_ivr(iv_series):
    """Calculate IV Rank from historical IV proxy series"""
    iv_series = iv_series.dropna()
    if iv_series.empty:
        return None

    current_iv = iv_series.iloc[-1] * 100  # FIXED: no more FutureWarning
    iv_min = iv_series.min() * 100
    iv_max = iv_series.max() * 100

    if iv_max == iv_min:  # avoid divide by zero
        return None

    ivr = (current_iv - iv_min) / (iv_max - iv_min) * 100
    # clamp 0–100
    ivr = max(0, min(100, ivr))
    return ivr

def ivr_label(ivr):
    """Return label for IVR"""
    if ivr is None:
        return "N/A"
    if ivr < 20:
        return color_text(f"{ivr:.1f}% (Cheap)", "green")
    elif ivr > 50:
        return color_text(f"{ivr:.1f}% (Expensive)", "red")
    else:
        return color_text(f"{ivr:.1f}% (Fair)", "yellow")

# -----------------------------
# Main loop
# -----------------------------
for symbol, opts in options_to_track.items():
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y")
    current_price = hist["Close"].iloc[-1]

    print(color_text(f"\n--- {symbol} @ {current_price:.2f} ---", "yellow"))

    # Rough IV proxy from historical volatility
    try:
        iv_hist = hist["Close"].pct_change().rolling(20).std() * (252**0.5)
        ivr = calculate_ivr(iv_hist)
    except Exception:
        ivr = None

    for expiry, strike, opt_type in opts:
        try:
            chain = ticker.option_chain(expiry)
            if opt_type.lower() == "call":
                opt = chain.calls[chain.calls["strike"] == strike]
            else:
                opt = chain.puts[chain.puts["strike"] == strike]

            if not opt.empty:
                last_price = opt["lastPrice"].values[0]
                bid = opt["bid"].values[0]
                ask = opt["ask"].values[0]

                ivr_text = ivr_label(ivr)

                print(
                    f"{opt_type.upper()} {strike} {expiry}: "
                    f"Last={last_price:.2f}, Bid={bid:.2f}, Ask={ask:.2f}, "
                    f"IVR={ivr_text}"
                )
            else:
                print(f"No {opt_type} found at strike {strike} for {expiry}")
        except Exception as e:
            print(f"Error fetching {symbol} {expiry} {strike} {opt_type}: {e}")

