import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
from scipy.stats import norm

def black_scholes_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

st.title("📈 Options Yield Scanner")
st.markdown("Find high-yield option candidates based on delta and return criteria.")

symbol = st.text_input("Ticker Symbol", value="TQQQ").upper()
option_type = st.selectbox("Option Type", ["call", "put"])
delta_target = st.number_input("Target Delta (optional)", value=0.0, step=0.05)
delta_bound_type = st.selectbox("Is this your minimum or maximum delta?", ["min", "max"])
min_weekly_yield = st.number_input("Minimum Weekly Yield (e.g. 0.01 for 1%)", value=0.01)
max_exp_date = st.date_input("Max Expiration Date", value=pd.Timestamp("2025-12-31"))
max_bid_ask_spread = st.number_input("Maximum Bid-Ask Spread", value=0.25)

if st.button("🔍 Scan Options"):
    ticker = yf.Ticker(symbol)
    try:
        expirations = [
            datetime.strptime(date, "%Y-%m-%d") for date in ticker.options
            if datetime.strptime(date, "%Y-%m-%d").date() <= max_exp_date
        ]
    except Exception as e:
        st.error(f"Error getting expiration dates: {e}")
        st.stop()

    try:
        current_price = ticker.history(period='1d')['Close'].iloc[-1]
    except:
        st.error("Error fetching current price.")
        st.stop()

    r = 0.05  # Risk-free rate
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

            if delta_target:
                if option_type == 'put':
                    if delta_bound_type == 'min':
                        df = df[df['delta_calc'] <= -abs(delta_target)]
                    else:
                        df = df[df['delta_calc'] >= -abs(delta_target)]
                else:
                    if delta_bound_type == 'min':
                        df = df[df['delta_calc'] >= delta_target]
                    else:
                        df = df[df['delta_calc'] <= delta_target]

            if option_type == 'call':
                df['intrinsic'] = (current_price - df['strike']).clip(lower=0)
            else:
                df['intrinsic'] = (df['strike'] - current_price).clip(lower=0)

            df['extrinsic_premium'] = df['mid'] - df['intrinsic']
            df['total_yield'] = df['extrinsic_premium'] / df['strike']
            df['weekly_yield'] = df['total_yield'] / (df['days_to_exp'] / 7)

            if option_type == 'put':
                df['expected_utility_yield'] = df['total_yield'] * (1 - abs(df['delta_calc']))
            else:
                df['expected_utility_yield'] = df['total_yield'] * abs(df['delta_calc'])

            df['expected_weekly_yield'] = df['expected_utility_yield'] / (df['days_to_exp'] / 7)
            df = df[df['weekly_yield'] >= min_weekly_yield]
            all_options.append(df)

        except Exception as e:
            st.warning(f"Skipping {exp.strftime('%Y-%m-%d')}: {e}")
            continue

    if not all_options:
        st.warning("No options matched your criteria.")
        st.stop()

    final_df = pd.concat(all_options)
    final_df['strike'] = final_df['strike'].map(lambda x: f"{x:.2f}")

    final_df = final_df.sort_values(by=['weekly_yield', 'expected_weekly_yield'], ascending=False)
    display_cols = [
        'contractSymbol', 'expiration', 'strike', 'delta_calc',
        'total_yield', 'weekly_yield', 'expected_utility_yield',
        'expected_weekly_yield', 'extrinsic_premium', 'bid_ask_spread'
    ]
    st.dataframe(final_df[display_cols].head(20).reset_index(drop=True))

    best = final_df.iloc[0]
    st.markdown("### 💎 Best Trade Candidate")
    st.markdown(f"""
    - **Expiration**: {best['expiration'].strftime('%Y-%m-%d')}
    - **Strike**: {best['strike']}
    - **Delta**: {best['delta_calc']:.4f}
    - **Weekly Yield**: {best['weekly_yield']*100:.2f}%
    - **Expected Utility Yield (EUY)**: {best['expected_utility_yield']*100:.4f}%
    - **Expected Weekly Yield**: {best['expected_weekly_yield']*100:.4f}%
    - **Extrinsic Premium**: ${best['extrinsic_premium']:.2f}
    - **Bid-Ask Spread**: ${best['bid_ask_spread']:.4f}
    """)
