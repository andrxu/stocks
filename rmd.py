from datetime import datetime

# 2025 IRS Uniform Lifetime Table factors (ages 72–120)
life_expectancy_factors = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0,
    79: 21.1, 80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0,
    86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8,
    93: 10.1, 94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4,
    101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1,
    108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1, 114: 3.0,
    115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0,
}

# 2025 single filer tax brackets
tax_brackets = [
    (0, 11000, 0.10),
    (11001, 44725, 0.12),
    (44726, 95375, 0.22),
    (95376, 182100, 0.24),
    (182101, 231250, 0.32),
    (231251, 578125, 0.35),
    (578126, float("inf"), 0.37),
]

def get_tax(rmd):
    """Compute progressive (marginal) tax on the RMD and return (tax, bracket_index).

    The function walks the tax_brackets and taxes each portion of the RMD at the
    marginal rate for that bracket. It returns the total tax and the index of the
    highest bracket that applies.
    """
    tax = 0.0
    highest_bracket = 0
    for i, (low, high, rate) in enumerate(tax_brackets, start=1):
        if rmd <= low:
            break
        # taxable amount in this bracket
        taxable = max(0.0, min(rmd, high) - low)
        tax += taxable * rate
        if rmd > low:
            highest_bracket = i
        if rmd <= high:
            break
    return tax, highest_bracket if highest_bracket > 0 else len(tax_brackets)

def main():
    def print_description():
        print("RMD projection tool - columns:")
        print("  Age: taxable age for the year")
        print("  Factor: IRS life expectancy divisor for that age")
        print("  Prior Balance: starting account balance for the year")
        print("  RMD: required minimum distribution for the year")
        print("  % of Bal: RMD as percent of starting balance")
        print("  Tax: estimated tax on the RMD using single-filer brackets")
        print("  Bracket: tax bracket index used")
        print("  End Balance: balance after withdrawal and growth")
        print("")

    # User input
    # Defaults if user hits Enter
    default_birth = "1952-09-10"
    default_balance = 1000000.0
    default_return = 0.05

    try:
        birth_date_str = input(f"Enter your birth date (YYYY-MM-DD) [default: {default_birth}]: ") or default_birth
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
    except Exception as e:
        print(f"Invalid birth date: {e}")
        return

    try:
        bal_in = input(f"Enter your account balance today [default: {int(default_balance)}]: ")
        balance = float(bal_in) if bal_in.strip() else default_balance
    except Exception as e:
        print(f"Invalid balance: {e}")
        return

    try:
        ret_in = input(f"Enter expected annual return rate (e.g., 0.05 for 5%) [default: {default_return}]: ")
        annual_return = float(ret_in) if ret_in.strip() else default_return
    except Exception as e:
        print(f"Invalid annual return: {e}")
        return

    if balance <= 0:
        print("Balance must be positive")
        return

    print_description()

    # Current year and age
    current_year = datetime.today().year
    age = current_year - birth_date.year
    if (datetime.today().month, datetime.today().day) < (birth_date.month, birth_date.day):
        age -= 1

    print(f"\nYou are age {age} this year.\n")

    # Print header
    # Right-aligned numeric columns for better readability
    print(
        f"{'Age':>5} {'Factor':>10} {'Prior Balance':>18} {'RMD':>15}"
        f" {'% of Bal':>10} {'Tax':>12} {'Bracket':>8} {'End Balance':>15}"
    )
    print("-" * 115)

    cumulative_tax = 0.0
    cumulative_rmd = 0.0

    for yr in range(age, 121):
        if yr not in life_expectancy_factors:
            break

        factor = life_expectancy_factors[yr]
        if factor <= 0:
            print(f"Skipping age {yr} due to non-positive factor: {factor}")
            continue
        rmd = balance / factor
        pct_of_bal = rmd / balance * 100
        tax, bracket = get_tax(rmd)
        cumulative_tax += tax
        cumulative_rmd += rmd

        # End balance after withdrawal and growth
        end_balance = (balance - rmd) * (1 + annual_return)

        print(
            f"{yr:>5} {factor:>10.1f} {balance:>18,.2f} {rmd:>15,.2f}"
            f" {pct_of_bal:>10.2f} {tax:>12,.2f} {bracket:>8} {end_balance:>15,.2f}"
        )

        balance = end_balance  # carry forward

    print("-" * 115)
    print(f"Cumulative tax paid over period: {cumulative_tax:,.2f}")
    print(f"Cumulative RMD withdrawn over period: {cumulative_rmd:,.2f}")

if __name__ == "__main__":
    main()
