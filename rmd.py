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
    """Return tax amount and bracket number for a given RMD."""
    for i, (low, high, rate) in enumerate(tax_brackets, start=1):
        if low <= rmd <= high:
            return rmd * rate, i
    return rmd * tax_brackets[-1][2], len(tax_brackets)

def main():
    # User input
    birth_date_str = input("Enter your birth date (YYYY-MM-DD): ")
    birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
    balance = float(input("Enter your account balance today: "))
    annual_return = float(input("Enter expected annual return rate (e.g., 0.05 for 5%): "))

    # Current year and age
    current_year = datetime.today().year
    age = current_year - birth_date.year
    if (datetime.today().month, datetime.today().day) < (birth_date.month, birth_date.day):
        age -= 1

    print(f"\nYou are age {age} this year.\n")

    # Print header
    print(
        f"{'Age':<5}{'Factor':<10}{'Prior Balance':<18}{'RMD':<15}"
        f"{'% of Bal':<12}{'Tax':<12}{'Bracket':<10}{'End Balance':<15}"
    )
    print("-" * 95)

    for yr in range(age, 121):
        if yr not in life_expectancy_factors:
            break

        factor = life_expectancy_factors[yr]
        rmd = balance / factor
        pct_of_bal = rmd / balance * 100
        tax, bracket = get_tax(rmd)

        # End balance after withdrawal and growth
        end_balance = (balance - rmd) * (1 + annual_return)

        print(
            f"{yr:<5}{factor:<10.1f}{balance:<18,.2f}{rmd:<15,.2f}"
            f"{pct_of_bal:<12.2f}{tax:<12,.2f}{bracket:<10}{end_balance:<15,.2f}"
        )

        balance = end_balance  # carry forward

if __name__ == "__main__":
    main()
