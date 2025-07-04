import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.stats import norm
import numpy as np

def black_scholes_delta(S, K, T, r, sigma, option_type='call'):
    """Calculate delta using the Black-Scholes model."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def main():
    print("Welcome to the Options Scanner!")

    symbol = input("Ticker symbol (e.g., QQQ): ").upper()
    option_type = input("Option type? ('call' or 'put'): ").lower()
    delta_target_input = input("What delta are you targeting? (or press Enter to skip): ")
    delta_target = float(delta_target_input) if delta_target_input else None
    delta_bound_type = input("Is that your minimum or maximum delta? (type 'min' or 'max'): ").lower() if delta_target else None
    min_weekly_yield = float(input("What’s the minimum weekly yield you're willing to accept? (e.g., 0.01 for 1%): "))
    max_exp_date = datetime.strptime(input("How far out are you willing to go? (Max expiration date in YYYY-MM-DD format): "), "%Y-%m-%d")
    max_bid_ask_spread = float(input("How tight do you want the bid-ask spread? (e.g., 0.25): "))

    ticker = yf.Ticker(symbol)
    try:
        expirations = [datetime.strptime(date, "%Y-%m-%d") for date in ticker.options if datetime.strptime(date, "%Y-%m-%d") <= max_exp_date]
    except Exception as e:
        print("Error getting expiration dates:", e)
        return

    try:
        current_price = ticker.history(period='1d')['Close'].iloc[-1]
    except:
        print("Error getting current price.")
        return

    r = 0.05  # Risk-free rate (static for now)

    all_options = []

    for exp in expirations:
        try:
            chain = ticker.option_chain(exp.strftime('%Y-%m-%d'))
            df = chain.calls if option_type == 'call' else chain.puts
            df = df.copy()
            df['expiration'] = exp
            df['days_to_exp'] = (exp.date() - datetime.now().date()).days
            df = df[df['days_to_exp'] > 0]
            df['T'] = df['days_to_exp'] / 365.0
            df['mid'] = (df['bid'] + df['ask']) / 2
            df['bid_ask_spread'] = df['ask'] - df['bid']
            df = df[df['bid_ask_spread'] <= max_bid_ask_spread]
            df = df[df['impliedVolatility'] > 0]
            df['delta_calc'] = df.apply(
                lambda row: black_scholes_delta(
                    S=current_price,
                    K=row['strike'],
                    T=row['T'],
                    r=r,
                    sigma=row['impliedVolatility'],
                    option_type=option_type
                ),
                axis=1
            )

            # Apply delta filtering
            if delta_target:
                if option_type == 'put':
                    if delta_bound_type == 'min':
                        df = df[df['delta_calc'] <= -abs(delta_target)]
                    elif delta_bound_type == 'max':
                        df = df[df['delta_calc'] >= -abs(delta_target)]
                else:
                    if delta_bound_type == 'min':
                        df = df[df['delta_calc'] >= delta_target]
                    elif delta_bound_type == 'max':
                        df = df[df['delta_calc'] <= delta_target]

            # Calculate extrinsic premium
            if option_type == 'call':
                df['intrinsic'] = (current_price - df['strike']).clip(lower=0)
            else:
                df['intrinsic'] = (df['strike'] - current_price).clip(lower=0)
            df['extrinsic_premium'] = df['mid'] - df['intrinsic']

            # Calculate yields
            df['total_yield'] = df['extrinsic_premium'] / df['strike']
            df['weekly_yield'] = df['total_yield'] / (df['days_to_exp'] / 7)

            # CORRECTED: Expected utility yield
            if option_type == 'put':
                df['expected_utility_yield'] = df['total_yield'] * (1 - abs(df['delta_calc']))
            else:
                df['expected_utility_yield'] = df['total_yield'] * abs(df['delta_calc'])

            df['expected_weekly_yield'] = df['expected_utility_yield'] / (df['days_to_exp'] / 7)

            df = df[df['weekly_yield'] >= min_weekly_yield]

            all_options.append(df)

        except Exception as e:
            print(f"Error processing expiration {exp.strftime('%Y-%m-%d')}: {e}")
            continue

    if not all_options:
        print("No options matched your criteria.")
        return

    final_df = pd.concat(all_options)

    final_df['strike'] = final_df['strike'].map(lambda x: f"{x:.2f}")

    cols = [
        'contractSymbol', 'expiration', 'strike', 'delta_calc',
        'total_yield', 'weekly_yield', 'expected_utility_yield',
        'expected_weekly_yield', 'extrinsic_premium', 'bid_ask_spread'
    ]
    final_df = final_df[cols]

    final_df = final_df.sort_values(by=['weekly_yield', 'expected_weekly_yield'], ascending=False)

    print("\nTop Option Candidates:\n")
    print(final_df.head(10).to_string(index=False))

    best = final_df.iloc[0]

    print("\nBest Trade Candidate:")
    print(f"Expiration: {best['expiration'].strftime('%Y-%m-%d')}, Strike: {best['strike']}")
    print(f"Delta: {best['delta_calc']:.4f}, Weekly Yield: {best['weekly_yield']*100:.2f}%")
    print(f"Expected Utility Yield (EUY): {best['expected_utility_yield']*100:.4f}%")
    print(f"Expected Weekly Yield: {best['expected_weekly_yield']*100:.4f}%")
    print(f"Extrinsic Premium: ${best['extrinsic_premium']:.2f}")
    print(f"Bid-Ask Spread: ${best['bid_ask_spread']:.4f}")

if __name__ == "__main__":
    main()
