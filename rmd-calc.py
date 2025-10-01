import numpy as np
from datetime import datetime
import math

# -----------------------------
# Tax brackets (2025 Single filer)
# -----------------------------
TAX_BRACKETS = [
    (0, 11000, 0.10),
    (11001, 44725, 0.12),
    (44726, 95375, 0.22),
    (95376, 182100, 0.24),
    (182101, 231250, 0.32),
    (231251, 578125, 0.35),
    (578126, float('inf'), 0.37),
]

WITHDRAWAL_STEP = 2000  # minimal step for DP table

# -----------------------------
# IRS Uniform Lifetime Table
# -----------------------------
IRS_UNIFORM_LIFETIME = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9,
    78: 22.0, 79: 21.1, 80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7,
    84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9,
    90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9,
    96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4, 101: 6.0,
    102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1,
    108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1,
    114: 3.0, 115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3,
    120: 2.0
}

def get_life_factor(age: int) -> float:
    """Return IRS Uniform Lifetime factor, fallback gracefully if missing."""
    if age in IRS_UNIFORM_LIFETIME:
        return IRS_UNIFORM_LIFETIME[age]
    elif age < 72:
        return IRS_UNIFORM_LIFETIME[72]
    else:
        return 2.0

# -----------------------------
# Helpers
# -----------------------------
def calculate_tax(amount):
    """Return tax and bracket number for given withdrawal."""
    for i, (low, high, rate) in enumerate(TAX_BRACKETS, 1):
        if amount <= high:
            tax = amount * rate
            return tax, i
    return 0, 1

def life_factor(age, birth_year, birth_month, birth_day):
    """Life expectancy factor from IRS table."""
    return get_life_factor(age)

def round_balance(x):
    """Round to nearest WITHDRAWAL_STEP for DP indexing."""
    return int(round(x / WITHDRAWAL_STEP) * WITHDRAWAL_STEP)

# -----------------------------
# Dynamic Programming Optimization
# -----------------------------
def optimize_withdrawals(start_balance, current_age, life_exp, growth_rate, target_bracket):
    years = life_exp - current_age + 1
    dp = [{} for _ in range(years + 1)]

    dp[-1][0] = (0, 0)

    for y in reversed(range(years)):
        age = current_age + y
        RMD = start_balance / get_life_factor(age)  # IRS factor
        for balance in np.arange(0, start_balance*3, WITHDRAWAL_STEP):
            min_withdraw = max(RMD, 0)  # enforce RMD
            max_withdraw = balance
            best_total_tax = float('inf')
            best_withdraw = 0
            w = min_withdraw
            while w <= max_withdraw:
                next_balance = balance - w
                next_balance = max(0, round_balance(next_balance * (1 + growth_rate)))
                future_tax = dp[y+1].get(next_balance, (0, 0))[0]
                tax, bracket = calculate_tax(w)
                if target_bracket and bracket > target_bracket:
                    w += WITHDRAWAL_STEP
                    continue
                total_tax = tax + future_tax
                if total_tax < best_total_tax:
                    best_total_tax = total_tax
                    best_withdraw = w
                w += WITHDRAWAL_STEP
            # Ensure withdrawal >= RMD
            if best_withdraw < RMD:
                best_withdraw = RMD
            dp[y][round_balance(balance)] = (best_total_tax, best_withdraw)

    withdrawals = []
    balance = round_balance(start_balance)
    for y in range(years):
        total_tax, w = dp[y][balance]
        tax, bracket = calculate_tax(w)
        withdrawals.append((current_age+y, w, tax, bracket, balance))
        balance = round_balance(balance - w)
        balance = round_balance(balance * (1 + growth_rate))
    return withdrawals

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    start_balance = float(input("Enter starting IRA balance: "))
    current_age = int(input("Enter your current age: "))
    life_exp = int(input("Enter expected age (life expectancy): "))
    growth_rate = float(input("Expected annual return rate (e.g., 0.05 for 5%): "))

    print("\n2025 Single Filer Brackets:")
    for i, (low, high, rate) in enumerate(TAX_BRACKETS, 1):
        print(f"{i}: {low}-{high if high != float('inf') else '+'} at {rate*100:.1f}%")
    target_bracket = int(input("Choose a bracket number to target (or 0 for no target): "))

    withdrawals = optimize_withdrawals(start_balance, current_age, life_exp, growth_rate, target_bracket)

    print("\nAge    LifeFactor     RMD Required   Withdrawal       Tax   Bracket     EndBalance")
    print("-"*95)
    total_tax = 0
    for age, w, tax, bracket, bal in withdrawals:
        lf = life_factor(age, 0, 0, 0)
        rmd_required = bal / lf
        end_balance = round_balance(bal - w + w*(1+growth_rate))
        total_tax += tax
        print(f"{age:3}    {lf:7}     {rmd_required:12,.2f}   {w:10,.2f}   {tax:10,.2f}    {bracket:3}    {end_balance:10,.2f}")
    print(f"\nTotal tax paid over life expectancy: {total_tax:,.2f}")
    total_withdrawals = sum(w for _, w, _, _, _ in withdrawals)
    final_balance = withdrawals[-1][4]
    print(f"Total distributions over life expectancy: {total_withdrawals:,.2f}")
    print(f"Final ending balance: {final_balance:,.2f}")
