import yfinance as yf
import pandas as pd
import numpy as np
from math import log, sqrt, exp, erf
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# -----------------------------
# Startup description and load tickers
# -----------------------------
print("LEAP watch - scans tickers and proposes LEAP call strikes and expirations")
print("Rules: pick an ideal call strike near a small OTM level (default +5%), and choose expiries ~12 and ~16 months out when available.")

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
    cfg = yaml.safe_load(f)

if not (isinstance(cfg, dict) and 'tickers' in cfg and isinstance(cfg['tickers'], list)):
    print('Invalid tickers.yml: expected a mapping with a "tickers" list')
    sys.exit(2)

options_to_track = {}
for t in cfg['tickers']:
    sym = str(t).strip()
    if sym:
        options_to_track[sym] = []

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
from datetime import datetime
from dateutil.relativedelta import relativedelta

def _round_strike(price):
    # Round strike to sensible increments: use 1 for <20, 2.5 for <100, 5 for <500, 10 for >=500
    if price < 20:
        step = 1
    elif price < 100:
        step = 2.5
    elif price < 500:
        step = 5
    else:
        step = 10
    return round(round(price / step) * step)

today = datetime.today().date()
# accumulator for best-candidates across all tickers
BEST_CANDIDATES = []
for symbol in sorted(options_to_track.keys()):
    ticker = yf.Ticker(symbol)
    try:
        hist = ticker.history(period="1y")
        current_price = float(hist["Close"].iloc[-1])
    except Exception as e:
        print(Fore.RED + f"{symbol}: failed to fetch price/history: {e}")
        continue

    print(color_text(f"\n--- {symbol} @ {current_price:.2f} ---", "yellow"))

    # compute IV proxy
    try:
        iv_hist = hist["Close"].pct_change().rolling(20).std() * (252**0.5)
        ivr = calculate_ivr(iv_hist)
    except Exception:
        ivr = None

    # ideal strike ~ +5% OTM by default
    ideal = current_price * 1.05
    ideal_strike = _round_strike(ideal)

    # target expirations roughly 12 and 16 months out
    target_12 = today + relativedelta(months=12)
    target_16 = today + relativedelta(months=16)

    # find available expiries and pick nearest to targets
    expiries = []
    try:
        expiries = [datetime.strptime(s, "%Y-%m-%d").date() for s in ticker.options]
    except Exception:
        expiries = []

    def _preferred_expiry_on_or_after(target):
        if not expiries:
            return None
        # prefer expiries on-or-after target; if none, pick the nearest earlier (latest before target)
        after = [d for d in expiries if d >= target]
        if after:
            # choose the soonest expiry on-or-after the target
            chosen = min(after, key=lambda d: (d - target).days)
        else:
            # choose the latest expiry before the target (closest earlier expiry)
            before = [d for d in expiries if d < target]
            if before:
                chosen = max(before)
            else:
                # fallback: pick the nearest expiry available
                chosen = min(expiries, key=lambda d: abs((d - target).days))
        return chosen.strftime("%Y-%m-%d")

    e12 = _preferred_expiry_on_or_after(target_12)
    e16 = _preferred_expiry_on_or_after(target_16)

    # Display proposals
    print(f"Proposed strike: {ideal_strike} (approx +5%)")
    print(f"Proposed expiries: 12m -> {e12}, 16m -> {e16}")

    ivr_text = ivr_label(ivr)

    # iterate unique expiries only (avoid duplicates when e12 == e16)
    # skip expiries that are too close (not LEAP-ish)
    min_leap_days = 300
    unique_expiries = sorted(set([e for e in (e12, e16) if e]))
    for expiry in unique_expiries:
        try:
            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            days_to_exp = (exp_date - today).days
        except Exception:
            days_to_exp = 0
        if days_to_exp < min_leap_days:
            print(f"Skipping expiry {expiry} ({days_to_exp} days) — too close to be a LEAP")
            continue
        if not expiry:
            print(f"No expiry available for target {expiry}")
            continue
        try:
            chain = ticker.option_chain(expiry)
            # find nearest available strike to our ideal
            calls = chain.calls.copy()
            if calls.empty:
                print(f"No calls available for expiry {expiry}")
                continue
            # prefer strikes that are OTM and liquid when scoring
            calls['is_otm'] = calls['strike'] >= current_price
            calls['liquid'] = ((calls.get('openInterest', 0) > 0) | ((calls.get('bid', 0) > 0) & (calls.get('ask', 0) > 0)))

            # Black-Scholes helpers for fallback implied vol (use mid price when available)
            def _norm_cdf(x):
                return 0.5 * (1.0 + erf(x / sqrt(2.0)))

            def _bs_call_price(S, K, T, r, sigma):
                if T <= 0 or sigma <= 0:
                    return max(0.0, S - K * exp(-r * T))
                d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
                d2 = d1 - sigma * sqrt(T)
                return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)

            def _implied_vol_from_price(mkt_price, S, K, T, r=0.03):
                if mkt_price is None or mkt_price <= 0 or T <= 0:
                    return None
                lo, hi = 1e-6, 5.0
                p_lo = _bs_call_price(S, K, T, r, lo)
                p_hi = _bs_call_price(S, K, T, r, hi)
                if mkt_price < p_lo or mkt_price > p_hi:
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

            # compute implied-vol percent values, using yfinance IV when present,
            # otherwise fall back to mid-price BS solve when sensible
            def compute_iv_pct(row):
                raw_iv = row.get('impliedVolatility', None)
                if raw_iv is not None and not pd.isna(raw_iv) and float(raw_iv) > 1e-4:
                    return float(raw_iv) * 100.0
                # fallback: try mid price
                bid = row.get('bid', None)
                ask = row.get('ask', None)
                last = row.get('lastPrice', None)
                mid = None
                try:
                    if bid is not None and ask is not None and not pd.isna(bid) and not pd.isna(ask) and (bid > 0 or ask > 0):
                        mid = (float(bid) + float(ask)) / 2.0
                    elif last is not None and not pd.isna(last) and last > 0:
                        mid = float(last)
                except Exception:
                    mid = None
                # time to expiry in years
                try:
                    exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
                    days = (exp_date - today).days
                    T = max(days / 365.0, 0.0)
                except Exception:
                    T = 0.0
                if mid is not None and T > 0:
                    try:
                        vol = _implied_vol_from_price(float(mid), float(current_price), float(row['strike']), T)
                        if vol is not None:
                            return float(vol) * 100.0
                    except Exception:
                        pass
                return np.nan

            calls['iv_pct'] = calls.apply(compute_iv_pct, axis=1)

            # For IV percentile, prefer using liquid strikes' IVs to avoid illiquid noise
            ivs_liquid = calls.loc[calls['liquid'] & calls['iv_pct'].notna(), 'iv_pct']
            ivs_all = calls.loc[calls['iv_pct'].notna(), 'iv_pct']
            ivs = ivs_liquid if len(ivs_liquid) >= 3 else ivs_all

            def iv_percentile(iv_value):
                try:
                    if ivs.empty or pd.isna(iv_value):
                        return None
                    return float((ivs < iv_value).sum()) / float(len(ivs)) * 100.0
                except Exception:
                    return None
            # prefer strikes that are OTM (strike >= price) and have some liquidity
            calls['is_otm'] = calls['strike'] >= current_price
            calls['liquid'] = ((calls.get('openInterest', 0) > 0) | ((calls.get('bid', 0) > 0) & (calls.get('ask', 0) > 0)))
            calls['diff'] = abs(calls['strike'] - ideal_strike)
            # scoring: prefer OTM (weight 2), liquid (weight 1), but keep closeness important
            calls['score'] = calls['is_otm'].astype(int) * 20 + calls['liquid'].astype(int) * 10 - calls['diff']
            # filter out strikes that are very far from ideal (e.g., > 3*step)
            step = 1 if ideal_strike < 20 else 2.5 if ideal_strike < 100 else 5 if ideal_strike < 500 else 10
            max_dist = step * 6
            candidates = calls[calls['diff'] <= max_dist].copy()
            if candidates.empty:
                # fallback to any strike within a wider range
                candidates = calls[calls['diff'] <= step * 12].copy()
            if candidates.empty:
                # last resort: use the closest strike
                best = calls.iloc[calls['diff'].argmin()]
                candidates = pd.DataFrame([best])
            # pick top 3 candidates ordered by score then closeness
            candidates = candidates.sort_values(by=['score', 'diff'], ascending=[False, True])
            top = candidates.head(3)
            for _, best in top.iterrows():
                last_price = best.get('lastPrice', None)
                bid = best.get('bid', None)
                ask = best.get('ask', None)
                iv = best.get('impliedVolatility', None)
                iv_display = f"{float(iv)*100:.2f}%" if iv is not None else 'N/A'
                ivp = None
                try:
                    ivp = iv_percentile(float(iv)*100) if iv is not None else None
                except Exception:
                    ivp = None
                ivp_text = f"IVP={ivp:.1f}%" if ivp is not None else "IVP=N/A"
                illiquid_flag = '' if best.get('liquid') else ' (ILLIQ)'
                print(f"CALL {best['strike']} {expiry}: Last={last_price}, Bid={bid}, Ask={ask}, IV={iv_display}, {ivp_text}, IVR={ivr_text}{illiquid_flag}")

                # collect candidate for end-of-run summary
                try:
                    BEST_CANDIDATES.append({
                        'symbol': symbol,
                        'expiry': expiry,
                        'strike': best['strike'],
                        'last': float(last_price) if last_price is not None and not pd.isna(last_price) else 0.0,
                        'bid': float(bid) if bid is not None and not pd.isna(bid) else 0.0,
                        'ask': float(ask) if ask is not None and not pd.isna(ask) else 0.0,
                        'iv': float(iv)*100.0 if iv is not None and not pd.isna(iv) else None,
                        'ivp': float(ivp) if ivp is not None else None,
                        'ivr': float(ivr) if ivr is not None else None,
                        'liquid': bool(best.get('liquid'))
                    })
                except Exception:
                    pass
        except Exception as e:
            print(f"Error fetching {symbol} {expiry}: {e}")

# End of per-ticker loop: print best candidates summary
def _score_candidate(c):
    # score: prefer liquid (+100), lower IVP (+ inverse), lower IV (+ inverse), higher last (+), closer to OTM (not implemented here)
    score = 0.0
    if c.get('liquid'):
        score += 100.0
    # prefer lower IVP (cheaper relative to peers)
    if c.get('ivp') is not None:
        score += max(0.0, 50.0 - c['ivp'])
    if c.get('iv') is not None:
        score += max(0.0, 30.0 - c['iv']) / 2.0
    score += min(50.0, c.get('last', 0.0)) / 10.0
    return score

def print_best_candidates(all_candidates, top_n=20, ivp_cutoff=60.0):
    if not all_candidates:
        print("No candidates collected.")
        return
    # filter out expensive ones by IVP (if IVP missing, keep)
    filtered = [c for c in all_candidates if (c.get('ivp') is None) or (c.get('ivp') < ivp_cutoff)]
    if not filtered:
        print("No candidates under IVP cutoff; relaxing filter to include all.")
        filtered = all_candidates
    # compute scores
    for c in filtered:
        c['_score'] = _score_candidate(c)
    ranked = sorted(filtered, key=lambda x: x['_score'], reverse=True)

    print('\n=== Best options to consider (excluding expensive IVP >= {:.0f}%) ==='.format(ivp_cutoff))
    print('Rank  Score  Symbol  Expiry       Strike  Last   IV    IVP   IVR   Liquid')
    print('----  -----  ------  ----------   ------  -----  ----  -----  ----  ------')
    for i, c in enumerate(ranked[:top_n], start=1):
        ivs = f"{c['iv']:.2f}%" if c.get('iv') is not None else 'N/A'
        ivp = f"{c['ivp']:.1f}%" if c.get('ivp') is not None else 'N/A'
        ivr = f"{c['ivr']:.1f}%" if c.get('ivr') is not None else 'N/A'
        liquid = 'Y' if c.get('liquid') else 'N'
        print(f"{i:>3}   {c['_score']:.1f}   {c['symbol']:6}  {c['expiry']}  {c['strike']:6}  {c['last']:5.2f}  {ivs:6}  {ivp:5}  {ivr:5}  {liquid}")

print_best_candidates(BEST_CANDIDATES, top_n=30, ivp_cutoff=60.0)

