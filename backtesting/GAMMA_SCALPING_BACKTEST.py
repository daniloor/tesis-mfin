"""
GAMMA SCALPING STRATEGY - GGAL OPTIONS
======================================

Implements a true gamma scalping strategy that maintains high gamma exposure
by rolling options when gamma decays below 50% of ATM gamma.

Key Features:
- Enters ATM options (highest gamma)
- Monitors gamma daily
- Rolls to new ATM when gamma drops below 50% of current ATM gamma
- Re-evaluates signal direction on each roll
- Continuous delta hedging with daily rebalancing
- Financing costs included

Author: GGAL Volatility Forecasting Thesis
Date: March 2026
"""

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime, timedelta
from glob import glob
from scipy.stats import norm, ttest_1samp
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
np.random.seed(42)

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTRADAY_DIR = os.path.join(SCRIPT_DIR, '..', 'intraday')
PROCESS_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'process_data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

print("=" * 100)
print("GAMMA SCALPING STRATEGY - GGAL OPTIONS")
print("Maintains high gamma exposure through rolling")
print("=" * 100)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

OUTPUT_DIR = RESULTS_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

COMMISSION_OPTIONS = 0.002  # 0.2% per option trade
COMMISSION_STOCK = 0.0005 * 1.21  # 0.05% + 21% VAT
MIN_DAYS_TO_EXPIRY = 7  # Roll to next OPEX if less than this
MIN_VOLUME = 500
LOTE_SIZE = 100
GAMMA_ROLL_THRESHOLD = 0.50  # Roll when gamma < 50% of ATM gamma
DELTA_UPPER_BOUND = 0.85  # Roll when delta > this (option too ITM)
DELTA_LOWER_BOUND = 0.15  # Roll when delta < this (option too OTM)
MARGIN_RATE = 1.00  # 100% margin requirement for short selling (conservative)

# HAR model prediction horizon
PREDICTION_HORIZON = 5

# Annualization factor for RV
ANNUALIZATION_FACTOR = np.sqrt(252)

# ============================================================================
# BLACK-SCHOLES FUNCTIONS
# ============================================================================

def bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_delta_call(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    return norm.cdf(bs_d1(S, K, T, r, sigma))

def bs_gamma(S, K, T, r, sigma):
    """Gamma is the same for calls and puts."""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_theta_call(S, K, T, r, sigma):
    """Theta for call option (per day)."""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    return theta / 365  # Per day

# ============================================================================
# LOAD DATA
# ============================================================================

print("[STEP 1] Loading data...")

df_stock = pd.read_csv(os.path.join(PROCESS_DATA_DIR, 'data.dat'))
df_stock['Date'] = pd.to_datetime(df_stock['Date'])
df_stock = df_stock.sort_values('Date').reset_index(drop=True)

options_files = sorted(glob(os.path.join(PROCESS_DATA_DIR, 'options_data_*.dat')))
options_list = []
for f in options_files:
    try:
        opt_df = pd.read_csv(f)
        opt_df['Date'] = pd.to_datetime(opt_df['Date'])
        basename = os.path.basename(f)
        parts = basename.replace('options_data_', '').replace('.dat', '').split('_')
        opt_df['OPEX'] = f"{parts[0]}-{parts[1]}"
        options_list.append(opt_df)
    except:
        pass

df_options = pd.concat(options_list, ignore_index=True)
print(f"  Stock: {len(df_stock)} days, Options: {len(df_options)} records")

# Load intraday data
print("  Loading intraday data for realized volatility...")
df_intraday = pd.read_csv(os.path.join(INTRADAY_DIR, 'BCBA_DLY_GGAL, 10 (1).csv'))
df_intraday['time'] = pd.to_datetime(df_intraday['time'])
df_intraday = df_intraday.sort_values('time').reset_index(drop=True)
print(f"  Intraday: {len(df_intraday)} bars")

# ============================================================================
# CALCULATE REALIZED VOLATILITY (Kumar 2010)
# ============================================================================

print("\n[STEP 2] Calculating realized volatility measures (Kumar 2010)...")

def calc_kumar_volatility(df, min_bars=10):
    df = df.copy()
    df['date'] = df['time'].dt.date
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['prev_date'] = df['date'].shift(1)
    df.loc[df['date'] != df['prev_date'], 'log_return'] = np.nan
    df['log_return_lag1'] = df['log_return'].shift(1)
    df.loc[df['date'] != df['prev_date'], 'log_return_lag1'] = np.nan

    daily_list = []
    mu_1 = np.sqrt(2 / np.pi)

    for date, group in df.groupby('date'):
        returns = group['log_return'].dropna()
        if len(returns) < min_bars:
            continue
        RV = (returns ** 2).sum()
        abs_returns = group['log_return'].abs()
        abs_returns_lag1 = group['log_return_lag1'].abs()
        bipower_products = (abs_returns * abs_returns_lag1).dropna()
        if len(bipower_products) > 0:
            RBV = (mu_1 ** -2) * bipower_products.sum()
        else:
            RBV = RV

        daily_list.append({
            'date': pd.to_datetime(date),
            'RV': np.sqrt(RV),
            'RBV': np.sqrt(RBV),
        })

    return pd.DataFrame(daily_list)

df_rv = calc_kumar_volatility(df_intraday)
print(f"  Realized volatility: {len(df_rv)} days")

# ============================================================================
# MERGE AND ANNUALIZE
# ============================================================================

print("\n[STEP 3] Merging and annualizing RV...")

df = df_stock.copy()
df['Date_date'] = df['Date'].dt.date
df_rv['date_key'] = df_rv['date'].dt.date
df = df.merge(df_rv[['date_key', 'RV', 'RBV']], left_on='Date_date', right_on='date_key', how='left')

# Calculate IV average
df['IV_Avg'] = (df['IV_Call_Avg'] + df['IV_Put_Avg']) / 2

# Annualize RV to match IV scale
df['RV'] = df['RV'] * ANNUALIZATION_FACTOR
df['RBV'] = df['RBV'] * ANNUALIZATION_FACTOR

df = df.dropna(subset=['RV', 'RBV'])
print(f"  Merged data: {len(df)} days")

# ============================================================================
# HAR-RBV-IV MODEL WITH DUAL SIGNALS
# ============================================================================
# Updated to use HAR-RBV-IV (the best model at 5-day horizon with 10min data)
# Features: RBV_d, RBV_w, RBV_m, IV_d (log of daily IV)

print("\n[STEP 4] Building HAR-RBV-IV model with DUAL signal generation...")

df['log_RBV'] = np.log(df['RBV'])
df['RBV_d'] = df['log_RBV'].shift(1)
df['RBV_w'] = df['log_RBV'].rolling(5).mean().shift(1)
df['RBV_m'] = df['log_RBV'].rolling(22).mean().shift(1)
df['IV_d'] = np.log(df['IV_Avg']).shift(1)  # Log of daily IV (lagged)
df['y_5d'] = df['log_RBV'].rolling(PREDICTION_HORIZON).mean().shift(-PREDICTION_HORIZON)

def fit_har_rbv_iv_dual_signals(df, window=120, horizon=5):
    """Generate predictions using HAR-RBV-IV model (RBV_d, RBV_w, RBV_m, IV_d)."""
    predictions = []
    feature_cols = ['RBV_d', 'RBV_w', 'RBV_m', 'IV_d']

    for i in range(len(df)):
        if i < window:
            predictions.append({
                'RV_Predicted': np.nan,
                'RV_vs_IV': np.nan,
                'Signal_A': None,
            })
            continue

        train = df.iloc[i-window:i].dropna(subset=feature_cols + ['y_5d'])
        if len(train) < 80:
            predictions.append({
                'RV_Predicted': np.nan,
                'RV_vs_IV': np.nan,
                'Signal_A': None,
            })
            continue

        X = train[feature_cols].values
        y = train['y_5d'].values
        model = LinearRegression().fit(X, y)

        today = df.iloc[i]
        X_today = today[feature_cols].values.astype(float).reshape(1, -1)
        if np.any(np.isnan(X_today)):
            predictions.append({
                'RV_Predicted': np.nan,
                'RV_vs_IV': np.nan,
                'Signal_A': None,
            })
            continue

        log_rv_pred = model.predict(X_today)[0]
        rv_predicted = np.exp(log_rv_pred)
        iv_current = today['IV_Avg']

        # MODEL A: RV vs IV (Variance Risk Premium)
        rv_vs_iv = (rv_predicted - iv_current) / iv_current

        if rv_vs_iv > 0.05:
            signal_A = 'UNDERPRICED'
        elif rv_vs_iv < -0.05:
            signal_A = 'OVERPRICED'
        else:
            signal_A = None

        predictions.append({
            'RV_Predicted': rv_predicted,
            'RV_vs_IV': rv_vs_iv,
            'Signal_A': signal_A,
        })

    return pd.DataFrame(predictions)

print("  Fitting HAR-RBV-IV model...")
har_predictions = fit_har_rbv_iv_dual_signals(df, window=120, horizon=PREDICTION_HORIZON)

df['RV_Predicted'] = har_predictions['RV_Predicted'].values
df['RV_vs_IV'] = har_predictions['RV_vs_IV'].values
df['Signal_A'] = har_predictions['Signal_A'].values

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_opex(date, df_options):
    """Get all OPEX cycles available on a given date."""
    day_options = df_options[df_options['Date'] == date]
    if len(day_options) == 0:
        return []
    return sorted(day_options['OPEX'].unique())

def get_opex_end_date(opex, df_options):
    """Get the last trading date for an OPEX."""
    opex_data = df_options[df_options['OPEX'] == opex]
    if len(opex_data) == 0:
        return None
    return opex_data['Date'].max()

def get_atm_option(date, stock_price, opex, df_options, option_type='Call'):
    """Get the ATM option for a given date and OPEX."""
    day_options = df_options[
        (df_options['Date'] == date) &
        (df_options['OPEX'] == opex) &
        (df_options['Type'] == option_type) &
        (df_options['Volume_Units'] >= MIN_VOLUME)
    ].copy()

    if len(day_options) == 0:
        # Try without volume filter
        day_options = df_options[
            (df_options['Date'] == date) &
            (df_options['OPEX'] == opex) &
            (df_options['Type'] == option_type)
        ].copy()

    if len(day_options) == 0:
        return None

    # Find closest to ATM
    day_options['distance'] = abs(day_options['Strike'] - stock_price)
    atm = day_options.loc[day_options['distance'].idxmin()]

    return {
        'strike': atm['Strike'],
        'price': atm['Last_Price'],
        'iv': atm['Implied_Volatility'],
        'opex': opex
    }

def get_option_price(date, strike, opex, df_options, option_type='Call'):
    """Get option price for a specific strike and date."""
    opt = df_options[
        (df_options['Date'] == date) &
        (abs(df_options['Strike'] - strike) < 1) &
        (df_options['OPEX'] == opex) &
        (df_options['Type'] == option_type)
    ]
    if len(opt) == 0:
        return None, None
    return opt.iloc[0]['Last_Price'], opt.iloc[0]['Implied_Volatility']

def calculate_atm_gamma(stock_price, risk_free, iv, days_to_expiry):
    """Calculate gamma for an ATM option."""
    T = max(days_to_expiry, 1) / 365
    return bs_gamma(stock_price, stock_price, T, risk_free, iv)

# ============================================================================
# GAMMA SCALPING STRATEGY
# ============================================================================

def run_gamma_scalping_strategy(df, df_options, option_type='CALL'):
    """
    Run gamma scalping strategy with rolling.

    Rules:
    1. Enter ATM option based on signal (LONG if UNDERPRICED, SHORT if OVERPRICED)
    2. Delta hedge daily
    3. Monitor gamma - if current gamma < 50% of ATM gamma, roll to new ATM
    4. Re-evaluate signal on each roll
    5. Continue until OPEX ends, then start fresh with next OPEX
    """

    # Define OPEX cycles
    opex_cycles = sorted(df_options['OPEX'].unique())
    opex_cycles = [o for o in opex_cycles if o >= '2025-04']  # Start from test period

    all_trades = []
    hedge_log = []

    for opex in opex_cycles:
        print(f"\n  Processing OPEX {opex}...")

        # Get trading days for this OPEX
        opex_options = df_options[df_options['OPEX'] == opex]
        opex_dates = sorted(opex_options['Date'].unique())

        if len(opex_dates) == 0:
            continue

        opex_start = opex_dates[0]
        opex_end = opex_dates[-1]

        # Filter stock data for this OPEX
        opex_df = df[(df['Date'] >= opex_start) & (df['Date'] <= opex_end)].copy()

        if len(opex_df) == 0:
            continue

        # State variables for this OPEX
        current_position = None  # 'LONG' or 'SHORT'
        current_strike = None
        current_entry_price = None
        current_entry_date = None
        current_iv = None
        cash_position = 0
        hedge_shares = 0
        financing_pnl = 0
        hedge_pnl = 0
        option_pnl = 0
        total_commission = 0
        prev_date = None
        prev_stock = None
        trade_count = 0

        for idx, (_, row) in enumerate(opex_df.iterrows()):
            current_date = row['Date']
            stock_price = row['Last_Price']
            risk_free = row.get('Risk_Free_Rate', 0.4)
            signal = row.get('Signal_A', None)

            days_to_expiry = (opex_end - current_date).days
            T = max(days_to_expiry, 1) / 365

            # Calculate ATM gamma for reference
            avg_iv = row.get('IV_Avg', 0.5)
            atm_gamma = calculate_atm_gamma(stock_price, risk_free, avg_iv, days_to_expiry)

            # ================================================================
            # CASE 1: No position - check if we should enter
            # ================================================================
            if current_position is None:
                if signal is None:
                    continue

                # Don't enter new positions if too close to expiry
                if days_to_expiry < MIN_DAYS_TO_EXPIRY:
                    continue

                # Get ATM option
                atm = get_atm_option(current_date, stock_price, opex, df_options,
                                     'Call' if option_type == 'CALL' else 'Put')
                if atm is None:
                    continue

                # Enter position
                current_position = 'LONG' if signal == 'UNDERPRICED' else 'SHORT'
                current_strike = atm['strike']
                current_entry_price = atm['price']
                current_entry_date = current_date
                current_iv = atm['iv'] if atm['iv'] and not pd.isna(atm['iv']) else avg_iv

                # Cash flow from option
                if current_position == 'LONG':
                    cash_position -= current_entry_price * LOTE_SIZE
                else:
                    cash_position += current_entry_price * LOTE_SIZE

                # Calculate initial delta and hedge
                if option_type == 'CALL':
                    delta = bs_delta_call(stock_price, current_strike, T, risk_free, current_iv)
                else:
                    delta = bs_delta_call(stock_price, current_strike, T, risk_free, current_iv) - 1

                if current_position == 'SHORT':
                    delta = -delta

                target_hedge = round(-delta * LOTE_SIZE)
                hedge_shares = target_hedge
                cash_position -= hedge_shares * stock_price

                total_commission += current_entry_price * LOTE_SIZE * COMMISSION_OPTIONS
                total_commission += abs(hedge_shares) * stock_price * COMMISSION_STOCK  # Initial hedge commission

                hedge_log.append({
                    'Date': current_date,
                    'OPEX': opex,
                    'Action': 'ENTER',
                    'Position': current_position,
                    'Strike': current_strike,
                    'Option_Price': current_entry_price,
                    'Stock': stock_price,
                    'Delta': delta,
                    'Gamma': bs_gamma(stock_price, current_strike, T, risk_free, current_iv),
                    'ATM_Gamma': atm_gamma,
                    'Hedge': hedge_shares,
                    'Cash': cash_position,
                    'Option_PnL': option_pnl,
                    'Hedge_PnL': hedge_pnl,
                    'Financing_PnL': financing_pnl,
                    'Commission': total_commission,
                    'Net_PnL': option_pnl + hedge_pnl + financing_pnl - total_commission
                })

                prev_date = current_date
                prev_stock = stock_price
                trade_count += 1
                continue

            # ================================================================
            # CASE 2: Have position - calculate financing and hedge P&L
            # ================================================================

            # Financing
            if prev_date is not None:
                calendar_days = (current_date - prev_date).days
                daily_rate = risk_free / 365
                period_interest = cash_position * daily_rate * calendar_days
                financing_pnl += period_interest
                cash_position += period_interest

            # Hedge P&L
            if prev_stock is not None and hedge_shares != 0:
                hedge_pnl += hedge_shares * (stock_price - prev_stock)

            # Get current option price and greeks
            opt_price, opt_iv = get_option_price(current_date, current_strike, opex, df_options,
                                                  'Call' if option_type == 'CALL' else 'Put')

            if opt_price is None:
                prev_date = current_date
                prev_stock = stock_price
                continue

            if opt_iv is None or pd.isna(opt_iv):
                opt_iv = current_iv

            # Calculate current greeks
            if option_type == 'CALL':
                delta = bs_delta_call(stock_price, current_strike, T, risk_free, opt_iv)
            else:
                delta = bs_delta_call(stock_price, current_strike, T, risk_free, opt_iv) - 1

            current_gamma = bs_gamma(stock_price, current_strike, T, risk_free, opt_iv)

            if current_position == 'SHORT':
                position_delta = -delta
            else:
                position_delta = delta

            # ================================================================
            # CASE 3: Check if we should ROLL (gamma decay or end of OPEX)
            # ================================================================
            should_roll = False
            roll_reason = None

            # Check gamma decay
            if atm_gamma > 0 and current_gamma < GAMMA_ROLL_THRESHOLD * atm_gamma:
                should_roll = True
                roll_reason = f"Gamma decay ({current_gamma:.6f} < {GAMMA_ROLL_THRESHOLD * atm_gamma:.6f})"

            # Check delta bounds (option too ITM or OTM)
            abs_delta = abs(delta)
            if abs_delta > DELTA_UPPER_BOUND:
                should_roll = True
                roll_reason = f"Delta too high ({abs_delta:.2f} > {DELTA_UPPER_BOUND})"
            elif abs_delta < DELTA_LOWER_BOUND:
                should_roll = True
                roll_reason = f"Delta too low ({abs_delta:.2f} < {DELTA_LOWER_BOUND})"

            # Check if near expiry
            if days_to_expiry <= MIN_DAYS_TO_EXPIRY:
                should_roll = True
                roll_reason = f"Near expiry ({days_to_expiry} days left)"

            if should_roll:
                # Close current position
                if current_position == 'LONG':
                    option_pnl += (opt_price - current_entry_price) * LOTE_SIZE
                    cash_position += opt_price * LOTE_SIZE
                else:
                    option_pnl += (current_entry_price - opt_price) * LOTE_SIZE
                    cash_position -= opt_price * LOTE_SIZE

                total_commission += opt_price * LOTE_SIZE * COMMISSION_OPTIONS

                # Close hedge
                cash_position += hedge_shares * stock_price
                total_commission += abs(hedge_shares) * stock_price * COMMISSION_STOCK

                hedge_log.append({
                    'Date': current_date,
                    'OPEX': opex,
                    'Action': f'ROLL_CLOSE ({roll_reason})',
                    'Position': current_position,
                    'Strike': current_strike,
                    'Option_Price': opt_price,
                    'Stock': stock_price,
                    'Delta': delta,
                    'Gamma': current_gamma,
                    'ATM_Gamma': atm_gamma,
                    'Hedge': 0,
                    'Cash': cash_position,
                    'Option_PnL': option_pnl,
                    'Hedge_PnL': hedge_pnl,
                    'Financing_PnL': financing_pnl,
                    'Commission': total_commission,
                    'Net_PnL': option_pnl + hedge_pnl + financing_pnl - total_commission
                })

                # Check for new signal
                new_signal = row.get('Signal_A', None)

                # Determine which OPEX to use
                if days_to_expiry <= MIN_DAYS_TO_EXPIRY:
                    # Need to roll to next OPEX
                    next_opex_idx = opex_cycles.index(opex) + 1 if opex in opex_cycles else -1
                    if next_opex_idx >= len(opex_cycles):
                        # No more OPEX, close out
                        current_position = None
                        current_strike = None
                        hedge_shares = 0
                        continue
                    target_opex = opex_cycles[next_opex_idx]
                else:
                    target_opex = opex

                # Get new ATM
                new_atm = get_atm_option(current_date, stock_price, target_opex, df_options,
                                         'Call' if option_type == 'CALL' else 'Put')

                if new_atm is None or new_signal is None:
                    # Can't roll, close out for this OPEX
                    current_position = None
                    current_strike = None
                    hedge_shares = 0
                    prev_date = current_date
                    prev_stock = stock_price
                    continue

                # Enter new position with possibly new direction
                current_position = 'LONG' if new_signal == 'UNDERPRICED' else 'SHORT'
                current_strike = new_atm['strike']
                current_entry_price = new_atm['price']
                current_entry_date = current_date
                current_iv = new_atm['iv'] if new_atm['iv'] and not pd.isna(new_atm['iv']) else avg_iv

                # Cash flow from new option
                if current_position == 'LONG':
                    cash_position -= current_entry_price * LOTE_SIZE
                else:
                    cash_position += current_entry_price * LOTE_SIZE

                # New hedge
                new_days = (get_opex_end_date(target_opex, df_options) - current_date).days
                new_T = max(new_days, 1) / 365

                if option_type == 'CALL':
                    new_delta = bs_delta_call(stock_price, current_strike, new_T, risk_free, current_iv)
                else:
                    new_delta = bs_delta_call(stock_price, current_strike, new_T, risk_free, current_iv) - 1

                if current_position == 'SHORT':
                    new_delta = -new_delta

                target_hedge = round(-new_delta * LOTE_SIZE)
                hedge_shares = target_hedge
                cash_position -= hedge_shares * stock_price

                total_commission += current_entry_price * LOTE_SIZE * COMMISSION_OPTIONS
                total_commission += abs(hedge_shares) * stock_price * COMMISSION_STOCK  # New hedge commission after roll

                hedge_log.append({
                    'Date': current_date,
                    'OPEX': target_opex,
                    'Action': 'ROLL_OPEN',
                    'Position': current_position,
                    'Strike': current_strike,
                    'Option_Price': current_entry_price,
                    'Stock': stock_price,
                    'Delta': new_delta,
                    'Gamma': bs_gamma(stock_price, current_strike, new_T, risk_free, current_iv),
                    'ATM_Gamma': atm_gamma,
                    'Hedge': hedge_shares,
                    'Cash': cash_position,
                    'Option_PnL': option_pnl,
                    'Hedge_PnL': hedge_pnl,
                    'Financing_PnL': financing_pnl,
                    'Commission': total_commission,
                    'Net_PnL': option_pnl + hedge_pnl + financing_pnl - total_commission
                })

                trade_count += 1

            else:
                # ================================================================
                # CASE 4: Normal day - just rebalance hedge
                # ================================================================
                target_hedge = round(-position_delta * LOTE_SIZE)
                adjustment = target_hedge - hedge_shares

                if adjustment != 0:
                    cash_position -= adjustment * stock_price
                    total_commission += abs(adjustment) * stock_price * COMMISSION_STOCK

                hedge_shares = target_hedge

                hedge_log.append({
                    'Date': current_date,
                    'OPEX': opex,
                    'Action': 'REBALANCE',
                    'Position': current_position,
                    'Strike': current_strike,
                    'Option_Price': opt_price,
                    'Stock': stock_price,
                    'Delta': delta,
                    'Gamma': current_gamma,
                    'ATM_Gamma': atm_gamma,
                    'Hedge': hedge_shares,
                    'Cash': cash_position,
                    'Option_PnL': option_pnl,
                    'Hedge_PnL': hedge_pnl,
                    'Financing_PnL': financing_pnl,
                    'Commission': total_commission,
                    'Net_PnL': option_pnl + hedge_pnl + financing_pnl - total_commission
                })

            prev_date = current_date
            prev_stock = stock_price

        # ================================================================
        # End of OPEX - close any remaining position
        # ================================================================
        if current_position is not None:
            final_date = opex_df.iloc[-1]['Date']
            final_stock = opex_df.iloc[-1]['Last_Price']

            # Get final option price
            final_opt_price, _ = get_option_price(final_date, current_strike, opex, df_options,
                                                   'Call' if option_type == 'CALL' else 'Put')

            # If no option price found, calculate intrinsic value
            if final_opt_price is None:
                if option_type == 'CALL':
                    final_opt_price = max(0, final_stock - current_strike)
                else:
                    final_opt_price = max(0, current_strike - final_stock)

            if current_position == 'LONG':
                option_pnl += (final_opt_price - current_entry_price) * LOTE_SIZE
            else:
                option_pnl += (current_entry_price - final_opt_price) * LOTE_SIZE

            total_commission += final_opt_price * LOTE_SIZE * COMMISSION_OPTIONS

            # Close hedge
            cash_position += hedge_shares * final_stock
            total_commission += abs(hedge_shares) * final_stock * COMMISSION_STOCK

            hedge_log.append({
                'Date': final_date,
                'OPEX': opex,
                'Action': 'CLOSE_OPEX_END',
                'Position': current_position,
                'Strike': current_strike,
                'Option_Price': final_opt_price,
                'Stock': final_stock,
                'Delta': 0,
                'Gamma': 0,
                'ATM_Gamma': 0,
                'Hedge': 0,
                'Cash': cash_position,
                'Option_PnL': option_pnl,
                'Hedge_PnL': hedge_pnl,
                'Financing_PnL': financing_pnl,
                'Commission': total_commission,
                'Net_PnL': option_pnl + hedge_pnl + financing_pnl - total_commission
            })

        # Record OPEX summary
        total_pnl = option_pnl + hedge_pnl + financing_pnl - total_commission

        # Calculate investment for return
        if opex_df.iloc[0]['Last_Price'] > 0:
            investment = opex_df.iloc[0]['Last_Price'] * LOTE_SIZE * MARGIN_RATE  # 100% margin
            net_return = (total_pnl / investment) * 100 if investment > 0 else 0
        else:
            net_return = 0

        all_trades.append({
            'OPEX': opex,
            'Trades': trade_count,
            'Option_PnL': option_pnl,
            'Hedge_PnL': hedge_pnl,
            'Financing_PnL': financing_pnl,
            'Commission': total_commission,
            'Total_PnL': total_pnl,
            'Net_Return': net_return
        })

        print(f"    {opex}: {trade_count} trades, Option={option_pnl:+,.0f}, Hedge={hedge_pnl:+,.0f}, "
              f"Finance={financing_pnl:+,.0f}, Total={total_pnl:+,.0f} ({net_return:+.1f}%)")

    return pd.DataFrame(all_trades), pd.DataFrame(hedge_log)

# ============================================================================
# STRADDLE GAMMA SCALPING STRATEGY
# ============================================================================

# Straddle-specific thresholds
STRADDLE_NET_DELTA_THRESHOLD = 0.50  # Roll when |net delta| > 0.50 (too directional)

def bs_delta_put(S, K, T, r, sigma):
    """Put delta = call delta - 1"""
    return bs_delta_call(S, K, T, r, sigma) - 1

def run_gamma_scalping_straddle(df, df_options):
    """
    Run gamma scalping strategy for STRADDLE (ATM Call + ATM Put, same strike).

    Rolling triggers:
    1. Net delta exceeds ±0.50 (position too directional)
    2. Combined gamma < 50% of ATM straddle gamma
    3. Near expiry (< 7 days)
    """

    opex_cycles = sorted(df_options['OPEX'].unique())
    opex_cycles = [o for o in opex_cycles if o >= '2025-04']

    all_trades = []
    hedge_log = []

    for opex in opex_cycles:
        print(f"\n  Processing OPEX {opex}...")

        opex_options = df_options[df_options['OPEX'] == opex]
        opex_dates = sorted(opex_options['Date'].unique())

        if len(opex_dates) == 0:
            continue

        opex_start = opex_dates[0]
        opex_end = opex_dates[-1]
        opex_df = df[(df['Date'] >= opex_start) & (df['Date'] <= opex_end)].copy()

        if len(opex_df) == 0:
            continue

        # State variables
        current_position = None  # 'LONG' or 'SHORT'
        current_strike = None
        call_entry_price = None
        put_entry_price = None
        current_entry_date = None
        current_iv = None
        cash_position = 0
        hedge_shares = 0
        financing_pnl = 0
        hedge_pnl = 0
        option_pnl = 0
        total_commission = 0
        prev_date = None
        prev_stock = None
        trade_count = 0

        for idx, (_, row) in enumerate(opex_df.iterrows()):
            current_date = row['Date']
            stock_price = row['Last_Price']
            risk_free = row.get('Risk_Free_Rate', 0.4)
            signal = row.get('Signal_A', None)

            days_to_expiry = (opex_end - current_date).days
            T = max(days_to_expiry, 1) / 365
            avg_iv = row.get('IV_Avg', 0.5)

            # Calculate ATM straddle gamma for reference (2x single option gamma)
            atm_gamma_single = calculate_atm_gamma(stock_price, risk_free, avg_iv, days_to_expiry)
            atm_straddle_gamma = 2 * atm_gamma_single

            # ================================================================
            # CASE 1: No position - check if we should enter
            # ================================================================
            if current_position is None:
                if signal is None:
                    continue

                # Don't enter new positions if too close to expiry
                if days_to_expiry < MIN_DAYS_TO_EXPIRY:
                    continue

                # Get ATM options (same strike for call and put)
                atm_call = get_atm_option(current_date, stock_price, opex, df_options, 'Call')
                if atm_call is None:
                    continue

                # Get put at same strike
                put_price, put_iv = get_option_price(current_date, atm_call['strike'], opex, df_options, 'Put')
                if put_price is None:
                    continue

                current_position = 'LONG' if signal == 'UNDERPRICED' else 'SHORT'
                current_strike = atm_call['strike']
                call_entry_price = atm_call['price']
                put_entry_price = put_price
                current_entry_date = current_date
                current_iv = atm_call['iv'] if atm_call['iv'] and not pd.isna(atm_call['iv']) else avg_iv

                # Cash flow from options (straddle = call + put)
                straddle_premium = (call_entry_price + put_entry_price) * LOTE_SIZE
                if current_position == 'LONG':
                    cash_position -= straddle_premium  # Pay premium
                else:
                    cash_position += straddle_premium  # Receive premium

                # Calculate initial delta and hedge
                call_delta = bs_delta_call(stock_price, current_strike, T, risk_free, current_iv)
                put_delta = bs_delta_put(stock_price, current_strike, T, risk_free, current_iv)
                net_delta = call_delta + put_delta  # For straddle

                if current_position == 'SHORT':
                    net_delta = -net_delta

                target_hedge = round(-net_delta * LOTE_SIZE)
                hedge_shares = target_hedge
                cash_position -= hedge_shares * stock_price

                total_commission += straddle_premium * COMMISSION_OPTIONS
                total_commission += abs(hedge_shares) * stock_price * COMMISSION_STOCK  # Initial hedge commission

                hedge_log.append({
                    'Date': current_date,
                    'OPEX': opex,
                    'Action': 'ENTER',
                    'Position': current_position,
                    'Strike': current_strike,
                    'Call_Price': call_entry_price,
                    'Put_Price': put_entry_price,
                    'Stock': stock_price,
                    'Net_Delta': net_delta,
                    'Combined_Gamma': 2 * bs_gamma(stock_price, current_strike, T, risk_free, current_iv),
                    'ATM_Straddle_Gamma': atm_straddle_gamma,
                    'Hedge': hedge_shares,
                    'Cash': cash_position,
                    'Option_PnL': option_pnl,
                    'Hedge_PnL': hedge_pnl,
                    'Financing_PnL': financing_pnl,
                    'Commission': total_commission,
                    'Net_PnL': option_pnl + hedge_pnl + financing_pnl - total_commission
                })

                prev_date = current_date
                prev_stock = stock_price
                trade_count += 1
                continue

            # ================================================================
            # CASE 2: Have position - calculate financing and hedge P&L
            # ================================================================
            if prev_date is not None:
                calendar_days = (current_date - prev_date).days
                daily_rate = risk_free / 365
                period_interest = cash_position * daily_rate * calendar_days
                financing_pnl += period_interest
                cash_position += period_interest

            if prev_stock is not None and hedge_shares != 0:
                hedge_pnl += hedge_shares * (stock_price - prev_stock)

            # Get current option prices
            call_price, call_iv = get_option_price(current_date, current_strike, opex, df_options, 'Call')
            put_price, put_iv = get_option_price(current_date, current_strike, opex, df_options, 'Put')

            if call_price is None or put_price is None:
                prev_date = current_date
                prev_stock = stock_price
                continue

            opt_iv = call_iv if call_iv and not pd.isna(call_iv) else current_iv

            # Calculate current greeks
            call_delta = bs_delta_call(stock_price, current_strike, T, risk_free, opt_iv)
            put_delta = bs_delta_put(stock_price, current_strike, T, risk_free, opt_iv)
            net_delta = call_delta + put_delta
            combined_gamma = 2 * bs_gamma(stock_price, current_strike, T, risk_free, opt_iv)

            if current_position == 'SHORT':
                position_delta = -net_delta
            else:
                position_delta = net_delta

            # ================================================================
            # CASE 3: Check if we should ROLL
            # ================================================================
            should_roll = False
            roll_reason = None

            # Check net delta (position too directional)
            if abs(net_delta) > STRADDLE_NET_DELTA_THRESHOLD:
                should_roll = True
                roll_reason = f"Net delta too high ({abs(net_delta):.2f} > {STRADDLE_NET_DELTA_THRESHOLD})"

            # Check combined gamma decay
            if atm_straddle_gamma > 0 and combined_gamma < GAMMA_ROLL_THRESHOLD * atm_straddle_gamma:
                should_roll = True
                roll_reason = f"Gamma decay ({combined_gamma:.6f} < {GAMMA_ROLL_THRESHOLD * atm_straddle_gamma:.6f})"

            # Check if near expiry
            if days_to_expiry <= MIN_DAYS_TO_EXPIRY:
                should_roll = True
                roll_reason = f"Near expiry ({days_to_expiry} days left)"

            if should_roll:
                # Close current position
                if current_position == 'LONG':
                    option_pnl += ((call_price - call_entry_price) + (put_price - put_entry_price)) * LOTE_SIZE
                    cash_position += (call_price + put_price) * LOTE_SIZE
                else:
                    option_pnl += ((call_entry_price - call_price) + (put_entry_price - put_price)) * LOTE_SIZE
                    cash_position -= (call_price + put_price) * LOTE_SIZE

                total_commission += (call_price + put_price) * LOTE_SIZE * COMMISSION_OPTIONS

                # Close hedge
                cash_position += hedge_shares * stock_price
                total_commission += abs(hedge_shares) * stock_price * COMMISSION_STOCK

                hedge_log.append({
                    'Date': current_date,
                    'OPEX': opex,
                    'Action': f'ROLL_CLOSE ({roll_reason})',
                    'Position': current_position,
                    'Strike': current_strike,
                    'Call_Price': call_price,
                    'Put_Price': put_price,
                    'Stock': stock_price,
                    'Net_Delta': net_delta,
                    'Combined_Gamma': combined_gamma,
                    'ATM_Straddle_Gamma': atm_straddle_gamma,
                    'Hedge': 0,
                    'Cash': cash_position,
                    'Option_PnL': option_pnl,
                    'Hedge_PnL': hedge_pnl,
                    'Financing_PnL': financing_pnl,
                    'Commission': total_commission,
                    'Net_PnL': option_pnl + hedge_pnl + financing_pnl - total_commission
                })

                # Check for new signal and roll
                new_signal = row.get('Signal_A', None)

                if days_to_expiry <= MIN_DAYS_TO_EXPIRY:
                    next_opex_idx = opex_cycles.index(opex) + 1 if opex in opex_cycles else -1
                    if next_opex_idx >= len(opex_cycles):
                        current_position = None
                        current_strike = None
                        hedge_shares = 0
                        continue
                    target_opex = opex_cycles[next_opex_idx]
                else:
                    target_opex = opex

                # Get new ATM options
                new_atm_call = get_atm_option(current_date, stock_price, target_opex, df_options, 'Call')
                if new_atm_call is None or new_signal is None:
                    current_position = None
                    current_strike = None
                    hedge_shares = 0
                    prev_date = current_date
                    prev_stock = stock_price
                    continue

                new_put_price, _ = get_option_price(current_date, new_atm_call['strike'], target_opex, df_options, 'Put')
                if new_put_price is None:
                    current_position = None
                    current_strike = None
                    hedge_shares = 0
                    prev_date = current_date
                    prev_stock = stock_price
                    continue

                # Enter new position
                current_position = 'LONG' if new_signal == 'UNDERPRICED' else 'SHORT'
                current_strike = new_atm_call['strike']
                call_entry_price = new_atm_call['price']
                put_entry_price = new_put_price
                current_entry_date = current_date
                current_iv = new_atm_call['iv'] if new_atm_call['iv'] and not pd.isna(new_atm_call['iv']) else avg_iv

                straddle_premium = (call_entry_price + put_entry_price) * LOTE_SIZE
                if current_position == 'LONG':
                    cash_position -= straddle_premium
                else:
                    cash_position += straddle_premium

                # New hedge
                new_opex_end = get_opex_end_date(target_opex, df_options)
                new_days = (new_opex_end - current_date).days if new_opex_end else days_to_expiry
                new_T = max(new_days, 1) / 365

                new_call_delta = bs_delta_call(stock_price, current_strike, new_T, risk_free, current_iv)
                new_put_delta = bs_delta_put(stock_price, current_strike, new_T, risk_free, current_iv)
                new_net_delta = new_call_delta + new_put_delta

                if current_position == 'SHORT':
                    new_net_delta = -new_net_delta

                target_hedge = round(-new_net_delta * LOTE_SIZE)
                hedge_shares = target_hedge
                cash_position -= hedge_shares * stock_price

                total_commission += straddle_premium * COMMISSION_OPTIONS
                total_commission += abs(hedge_shares) * stock_price * COMMISSION_STOCK  # New hedge commission after roll

                hedge_log.append({
                    'Date': current_date,
                    'OPEX': target_opex,
                    'Action': 'ROLL_OPEN',
                    'Position': current_position,
                    'Strike': current_strike,
                    'Call_Price': call_entry_price,
                    'Put_Price': put_entry_price,
                    'Stock': stock_price,
                    'Net_Delta': new_net_delta,
                    'Combined_Gamma': 2 * bs_gamma(stock_price, current_strike, new_T, risk_free, current_iv),
                    'ATM_Straddle_Gamma': atm_straddle_gamma,
                    'Hedge': hedge_shares,
                    'Cash': cash_position,
                    'Option_PnL': option_pnl,
                    'Hedge_PnL': hedge_pnl,
                    'Financing_PnL': financing_pnl,
                    'Commission': total_commission,
                    'Net_PnL': option_pnl + hedge_pnl + financing_pnl - total_commission
                })

                trade_count += 1
            else:
                # Normal rebalancing (no hedge_log entry needed for daily rebalances)
                target_hedge = round(-position_delta * LOTE_SIZE)
                adjustment = target_hedge - hedge_shares

                if adjustment != 0:
                    cash_position -= adjustment * stock_price
                    total_commission += abs(adjustment) * stock_price * COMMISSION_STOCK

                hedge_shares = target_hedge

            prev_date = current_date
            prev_stock = stock_price

        # End of OPEX - close position
        if current_position is not None:
            final_date = opex_df.iloc[-1]['Date']
            final_stock = opex_df.iloc[-1]['Last_Price']

            final_call_price, _ = get_option_price(final_date, current_strike, opex, df_options, 'Call')
            final_put_price, _ = get_option_price(final_date, current_strike, opex, df_options, 'Put')

            # Use intrinsic if no price
            if final_call_price is None:
                final_call_price = max(0, final_stock - current_strike)
            if final_put_price is None:
                final_put_price = max(0, current_strike - final_stock)

            if current_position == 'LONG':
                option_pnl += ((final_call_price - call_entry_price) + (final_put_price - put_entry_price)) * LOTE_SIZE
            else:
                option_pnl += ((call_entry_price - final_call_price) + (put_entry_price - final_put_price)) * LOTE_SIZE

            total_commission += (final_call_price + final_put_price) * LOTE_SIZE * COMMISSION_OPTIONS
            cash_position += hedge_shares * final_stock
            total_commission += abs(hedge_shares) * final_stock * COMMISSION_STOCK

            hedge_log.append({
                'Date': final_date,
                'OPEX': opex,
                'Action': 'CLOSE_OPEX_END',
                'Position': current_position,
                'Strike': current_strike,
                'Call_Price': final_call_price,
                'Put_Price': final_put_price,
                'Stock': final_stock,
                'Net_Delta': 0,
                'Combined_Gamma': 0,
                'ATM_Straddle_Gamma': 0,
                'Hedge': 0,
                'Cash': cash_position,
                'Option_PnL': option_pnl,
                'Hedge_PnL': hedge_pnl,
                'Financing_PnL': financing_pnl,
                'Commission': total_commission,
                'Net_PnL': option_pnl + hedge_pnl + financing_pnl - total_commission
            })

        # Record OPEX summary
        total_pnl = option_pnl + hedge_pnl + financing_pnl - total_commission

        if opex_df.iloc[0]['Last_Price'] > 0:
            investment = opex_df.iloc[0]['Last_Price'] * LOTE_SIZE * MARGIN_RATE  # 100% margin
            net_return = (total_pnl / investment) * 100 if investment > 0 else 0
        else:
            net_return = 0

        all_trades.append({
            'OPEX': opex,
            'Trades': trade_count,
            'Option_PnL': option_pnl,
            'Hedge_PnL': hedge_pnl,
            'Financing_PnL': financing_pnl,
            'Commission': total_commission,
            'Total_PnL': total_pnl,
            'Net_Return': net_return
        })

        print(f"    {opex}: {trade_count} trades, Option={option_pnl:+,.0f}, Hedge={hedge_pnl:+,.0f}, "
              f"Finance={financing_pnl:+,.0f}, Total={total_pnl:+,.0f} ({net_return:+.1f}%)")

    return pd.DataFrame(all_trades), pd.DataFrame(hedge_log)


# ============================================================================
# STRANGLE GAMMA SCALPING STRATEGY
# ============================================================================

# Strangle-specific: OTM call and OTM put
STRANGLE_OTM_PERCENT = 0.05  # 5% OTM for each leg

def run_gamma_scalping_strangle(df, df_options):
    """
    Run gamma scalping strategy for STRANGLE (OTM Call + OTM Put, different strikes).

    Rolling triggers:
    1. Either leg goes ITM (stock crosses a strike)
    2. Combined gamma < 50% of fresh OTM strangle gamma
    3. Near expiry (< 7 days)
    """

    opex_cycles = sorted(df_options['OPEX'].unique())
    opex_cycles = [o for o in opex_cycles if o >= '2025-04']

    all_trades = []
    hedge_log = []

    for opex in opex_cycles:
        print(f"\n  Processing OPEX {opex}...")

        opex_options = df_options[df_options['OPEX'] == opex]
        opex_dates = sorted(opex_options['Date'].unique())

        if len(opex_dates) == 0:
            continue

        opex_start = opex_dates[0]
        opex_end = opex_dates[-1]
        opex_df = df[(df['Date'] >= opex_start) & (df['Date'] <= opex_end)].copy()

        if len(opex_df) == 0:
            continue

        # State variables
        current_position = None
        call_strike = None
        put_strike = None
        call_entry_price = None
        put_entry_price = None
        current_iv = None
        cash_position = 0
        hedge_shares = 0
        financing_pnl = 0
        hedge_pnl = 0
        option_pnl = 0
        total_commission = 0
        prev_date = None
        prev_stock = None
        trade_count = 0

        def get_otm_strangle_options(date, stock, target_opex):
            """Get OTM call (strike > stock) and OTM put (strike < stock)."""
            call_target = stock * (1 + STRANGLE_OTM_PERCENT)
            put_target = stock * (1 - STRANGLE_OTM_PERCENT)

            # Find OTM call
            calls = df_options[
                (df_options['Date'] == date) &
                (df_options['OPEX'] == target_opex) &
                (df_options['Type'] == 'Call') &
                (df_options['Strike'] > stock)
            ].copy()

            if len(calls) == 0:
                return None, None

            calls['distance'] = abs(calls['Strike'] - call_target)
            otm_call = calls.loc[calls['distance'].idxmin()]

            # Find OTM put
            puts = df_options[
                (df_options['Date'] == date) &
                (df_options['OPEX'] == target_opex) &
                (df_options['Type'] == 'Put') &
                (df_options['Strike'] < stock)
            ].copy()

            if len(puts) == 0:
                return None, None

            puts['distance'] = abs(puts['Strike'] - put_target)
            otm_put = puts.loc[puts['distance'].idxmin()]

            return (
                {'strike': otm_call['Strike'], 'price': otm_call['Last_Price'],
                 'iv': otm_call['Implied_Volatility']},
                {'strike': otm_put['Strike'], 'price': otm_put['Last_Price'],
                 'iv': otm_put['Implied_Volatility']}
            )

        for idx, (_, row) in enumerate(opex_df.iterrows()):
            current_date = row['Date']
            stock_price = row['Last_Price']
            risk_free = row.get('Risk_Free_Rate', 0.4)
            signal = row.get('Signal_A', None)

            days_to_expiry = (opex_end - current_date).days
            T = max(days_to_expiry, 1) / 365
            avg_iv = row.get('IV_Avg', 0.5)

            # ================================================================
            # CASE 1: No position - check if we should enter
            # ================================================================
            if current_position is None:
                if signal is None:
                    continue

                # Don't enter new positions if too close to expiry
                if days_to_expiry < MIN_DAYS_TO_EXPIRY:
                    continue

                otm_call, otm_put = get_otm_strangle_options(current_date, stock_price, opex)
                if otm_call is None or otm_put is None:
                    continue

                current_position = 'LONG' if signal == 'UNDERPRICED' else 'SHORT'
                call_strike = otm_call['strike']
                put_strike = otm_put['strike']
                call_entry_price = otm_call['price']
                put_entry_price = otm_put['price']
                current_iv = otm_call['iv'] if otm_call['iv'] and not pd.isna(otm_call['iv']) else avg_iv

                strangle_premium = (call_entry_price + put_entry_price) * LOTE_SIZE
                if current_position == 'LONG':
                    cash_position -= strangle_premium
                else:
                    cash_position += strangle_premium

                # Calculate initial delta and hedge
                call_delta = bs_delta_call(stock_price, call_strike, T, risk_free, current_iv)
                put_delta = bs_delta_put(stock_price, put_strike, T, risk_free, current_iv)
                net_delta = call_delta + put_delta

                if current_position == 'SHORT':
                    net_delta = -net_delta

                target_hedge = round(-net_delta * LOTE_SIZE)
                hedge_shares = target_hedge
                cash_position -= hedge_shares * stock_price

                total_commission += strangle_premium * COMMISSION_OPTIONS
                total_commission += abs(hedge_shares) * stock_price * COMMISSION_STOCK  # Initial hedge commission

                combined_gamma = (bs_gamma(stock_price, call_strike, T, risk_free, current_iv) +
                                  bs_gamma(stock_price, put_strike, T, risk_free, current_iv))

                hedge_log.append({
                    'Date': current_date,
                    'OPEX': opex,
                    'Action': 'ENTER',
                    'Position': current_position,
                    'Call_Strike': call_strike,
                    'Put_Strike': put_strike,
                    'Call_Price': call_entry_price,
                    'Put_Price': put_entry_price,
                    'Stock': stock_price,
                    'Net_Delta': net_delta,
                    'Combined_Gamma': combined_gamma,
                    'Hedge': hedge_shares,
                    'Cash': cash_position,
                    'Option_PnL': option_pnl,
                    'Hedge_PnL': hedge_pnl,
                    'Financing_PnL': financing_pnl,
                    'Commission': total_commission,
                    'Net_PnL': option_pnl + hedge_pnl + financing_pnl - total_commission
                })

                prev_date = current_date
                prev_stock = stock_price
                trade_count += 1
                continue

            # ================================================================
            # CASE 2: Have position - financing and hedge P&L
            # ================================================================
            if prev_date is not None:
                calendar_days = (current_date - prev_date).days
                daily_rate = risk_free / 365
                period_interest = cash_position * daily_rate * calendar_days
                financing_pnl += period_interest
                cash_position += period_interest

            if prev_stock is not None and hedge_shares != 0:
                hedge_pnl += hedge_shares * (stock_price - prev_stock)

            # Get current option prices
            call_price, call_iv = get_option_price(current_date, call_strike, opex, df_options, 'Call')
            put_price, put_iv = get_option_price(current_date, put_strike, opex, df_options, 'Put')

            if call_price is None or put_price is None:
                prev_date = current_date
                prev_stock = stock_price
                continue

            opt_iv = call_iv if call_iv and not pd.isna(call_iv) else current_iv

            # Calculate current greeks
            call_delta = bs_delta_call(stock_price, call_strike, T, risk_free, opt_iv)
            put_delta = bs_delta_put(stock_price, put_strike, T, risk_free, opt_iv)
            net_delta = call_delta + put_delta
            combined_gamma = (bs_gamma(stock_price, call_strike, T, risk_free, opt_iv) +
                              bs_gamma(stock_price, put_strike, T, risk_free, opt_iv))

            if current_position == 'SHORT':
                position_delta = -net_delta
            else:
                position_delta = net_delta

            # ================================================================
            # CASE 3: Check if we should ROLL
            # ================================================================
            should_roll = False
            roll_reason = None

            # Check if either leg went ITM
            call_itm = stock_price > call_strike
            put_itm = stock_price < put_strike

            if call_itm:
                should_roll = True
                roll_reason = f"Call went ITM (stock {stock_price:.0f} > call strike {call_strike:.0f})"
            elif put_itm:
                should_roll = True
                roll_reason = f"Put went ITM (stock {stock_price:.0f} < put strike {put_strike:.0f})"

            # Check gamma decay (compare to what a fresh OTM strangle would have)
            fresh_call_strike = stock_price * (1 + STRANGLE_OTM_PERCENT)
            fresh_put_strike = stock_price * (1 - STRANGLE_OTM_PERCENT)
            fresh_gamma = (bs_gamma(stock_price, fresh_call_strike, T, risk_free, avg_iv) +
                           bs_gamma(stock_price, fresh_put_strike, T, risk_free, avg_iv))

            if fresh_gamma > 0 and combined_gamma < GAMMA_ROLL_THRESHOLD * fresh_gamma:
                should_roll = True
                roll_reason = f"Gamma decay ({combined_gamma:.6f} < {GAMMA_ROLL_THRESHOLD * fresh_gamma:.6f})"

            # Check if near expiry
            if days_to_expiry <= MIN_DAYS_TO_EXPIRY:
                should_roll = True
                roll_reason = f"Near expiry ({days_to_expiry} days left)"

            if should_roll:
                # Close current position
                if current_position == 'LONG':
                    option_pnl += ((call_price - call_entry_price) + (put_price - put_entry_price)) * LOTE_SIZE
                    cash_position += (call_price + put_price) * LOTE_SIZE
                else:
                    option_pnl += ((call_entry_price - call_price) + (put_entry_price - put_price)) * LOTE_SIZE
                    cash_position -= (call_price + put_price) * LOTE_SIZE

                total_commission += (call_price + put_price) * LOTE_SIZE * COMMISSION_OPTIONS
                cash_position += hedge_shares * stock_price
                total_commission += abs(hedge_shares) * stock_price * COMMISSION_STOCK

                hedge_log.append({
                    'Date': current_date,
                    'OPEX': opex,
                    'Action': f'ROLL_CLOSE ({roll_reason})',
                    'Position': current_position,
                    'Call_Strike': call_strike,
                    'Put_Strike': put_strike,
                    'Call_Price': call_price,
                    'Put_Price': put_price,
                    'Stock': stock_price,
                    'Net_Delta': net_delta,
                    'Combined_Gamma': combined_gamma,
                    'Hedge': 0,
                    'Cash': cash_position,
                    'Option_PnL': option_pnl,
                    'Hedge_PnL': hedge_pnl,
                    'Financing_PnL': financing_pnl,
                    'Commission': total_commission,
                    'Net_PnL': option_pnl + hedge_pnl + financing_pnl - total_commission
                })

                # Check for new signal and roll
                new_signal = row.get('Signal_A', None)

                if days_to_expiry <= MIN_DAYS_TO_EXPIRY:
                    next_opex_idx = opex_cycles.index(opex) + 1 if opex in opex_cycles else -1
                    if next_opex_idx >= len(opex_cycles):
                        current_position = None
                        call_strike = None
                        put_strike = None
                        hedge_shares = 0
                        continue
                    target_opex = opex_cycles[next_opex_idx]
                else:
                    target_opex = opex

                # Get new OTM strangle
                new_call, new_put = get_otm_strangle_options(current_date, stock_price, target_opex)

                if new_call is None or new_put is None or new_signal is None:
                    current_position = None
                    call_strike = None
                    put_strike = None
                    hedge_shares = 0
                    prev_date = current_date
                    prev_stock = stock_price
                    continue

                # Enter new position
                current_position = 'LONG' if new_signal == 'UNDERPRICED' else 'SHORT'
                call_strike = new_call['strike']
                put_strike = new_put['strike']
                call_entry_price = new_call['price']
                put_entry_price = new_put['price']
                current_iv = new_call['iv'] if new_call['iv'] and not pd.isna(new_call['iv']) else avg_iv

                strangle_premium = (call_entry_price + put_entry_price) * LOTE_SIZE
                if current_position == 'LONG':
                    cash_position -= strangle_premium
                else:
                    cash_position += strangle_premium

                # New hedge
                new_opex_end = get_opex_end_date(target_opex, df_options)
                new_days = (new_opex_end - current_date).days if new_opex_end else days_to_expiry
                new_T = max(new_days, 1) / 365

                new_call_delta = bs_delta_call(stock_price, call_strike, new_T, risk_free, current_iv)
                new_put_delta = bs_delta_put(stock_price, put_strike, new_T, risk_free, current_iv)
                new_net_delta = new_call_delta + new_put_delta

                if current_position == 'SHORT':
                    new_net_delta = -new_net_delta

                target_hedge = round(-new_net_delta * LOTE_SIZE)
                hedge_shares = target_hedge
                cash_position -= hedge_shares * stock_price

                total_commission += strangle_premium * COMMISSION_OPTIONS
                total_commission += abs(hedge_shares) * stock_price * COMMISSION_STOCK  # New hedge commission after roll

                hedge_log.append({
                    'Date': current_date,
                    'OPEX': target_opex,
                    'Action': 'ROLL_OPEN',
                    'Position': current_position,
                    'Call_Strike': call_strike,
                    'Put_Strike': put_strike,
                    'Call_Price': call_entry_price,
                    'Put_Price': put_entry_price,
                    'Stock': stock_price,
                    'Net_Delta': new_net_delta,
                    'Combined_Gamma': (bs_gamma(stock_price, call_strike, new_T, risk_free, current_iv) +
                                       bs_gamma(stock_price, put_strike, new_T, risk_free, current_iv)),
                    'Hedge': hedge_shares,
                    'Cash': cash_position,
                    'Option_PnL': option_pnl,
                    'Hedge_PnL': hedge_pnl,
                    'Financing_PnL': financing_pnl,
                    'Commission': total_commission,
                    'Net_PnL': option_pnl + hedge_pnl + financing_pnl - total_commission
                })

                trade_count += 1
            else:
                # Normal rebalancing (no hedge_log entry needed for daily rebalances)
                target_hedge = round(-position_delta * LOTE_SIZE)
                adjustment = target_hedge - hedge_shares

                if adjustment != 0:
                    cash_position -= adjustment * stock_price
                    total_commission += abs(adjustment) * stock_price * COMMISSION_STOCK

                hedge_shares = target_hedge

            prev_date = current_date
            prev_stock = stock_price

        # End of OPEX - close position
        if current_position is not None:
            final_date = opex_df.iloc[-1]['Date']
            final_stock = opex_df.iloc[-1]['Last_Price']

            final_call_price, _ = get_option_price(final_date, call_strike, opex, df_options, 'Call')
            final_put_price, _ = get_option_price(final_date, put_strike, opex, df_options, 'Put')

            if final_call_price is None:
                final_call_price = max(0, final_stock - call_strike)
            if final_put_price is None:
                final_put_price = max(0, put_strike - final_stock)

            if current_position == 'LONG':
                option_pnl += ((final_call_price - call_entry_price) + (final_put_price - put_entry_price)) * LOTE_SIZE
            else:
                option_pnl += ((call_entry_price - final_call_price) + (put_entry_price - final_put_price)) * LOTE_SIZE

            total_commission += (final_call_price + final_put_price) * LOTE_SIZE * COMMISSION_OPTIONS
            cash_position += hedge_shares * final_stock
            total_commission += abs(hedge_shares) * final_stock * COMMISSION_STOCK

            hedge_log.append({
                'Date': final_date,
                'OPEX': opex,
                'Action': 'CLOSE_OPEX_END',
                'Position': current_position,
                'Call_Strike': call_strike,
                'Put_Strike': put_strike,
                'Call_Price': final_call_price,
                'Put_Price': final_put_price,
                'Stock': final_stock,
                'Net_Delta': 0,
                'Combined_Gamma': 0,
                'Hedge': 0,
                'Cash': cash_position,
                'Option_PnL': option_pnl,
                'Hedge_PnL': hedge_pnl,
                'Financing_PnL': financing_pnl,
                'Commission': total_commission,
                'Net_PnL': option_pnl + hedge_pnl + financing_pnl - total_commission
            })

        # Record OPEX summary
        total_pnl = option_pnl + hedge_pnl + financing_pnl - total_commission

        if opex_df.iloc[0]['Last_Price'] > 0:
            investment = opex_df.iloc[0]['Last_Price'] * LOTE_SIZE * MARGIN_RATE  # 100% margin
            net_return = (total_pnl / investment) * 100 if investment > 0 else 0
        else:
            net_return = 0

        all_trades.append({
            'OPEX': opex,
            'Trades': trade_count,
            'Option_PnL': option_pnl,
            'Hedge_PnL': hedge_pnl,
            'Financing_PnL': financing_pnl,
            'Commission': total_commission,
            'Total_PnL': total_pnl,
            'Net_Return': net_return
        })

        print(f"    {opex}: {trade_count} trades, Option={option_pnl:+,.0f}, Hedge={hedge_pnl:+,.0f}, "
              f"Finance={financing_pnl:+,.0f}, Total={total_pnl:+,.0f} ({net_return:+.1f}%)")

    return pd.DataFrame(all_trades), pd.DataFrame(hedge_log)


# ============================================================================
# STOCK BASELINE STRATEGY
# ============================================================================

def run_stock_strategy(df, position_type='LONG', start_date=None, end_date=None):
    """
    Run a simple buy-and-hold stock strategy as baseline.

    Parameters:
    - df: DataFrame with stock data
    - position_type: 'LONG' (buy and hold) or 'SHORT' (sell and hold)
    - start_date: Optional start date (uses first available if None)
    - end_date: Optional end date (uses last available if None)

    Returns:
    - DataFrame with single trade result
    """
    # Filter to date range if specified
    df_filtered = df.copy()
    if start_date is not None:
        df_filtered = df_filtered[df_filtered['Date'] >= start_date]
    if end_date is not None:
        df_filtered = df_filtered[df_filtered['Date'] <= end_date]

    # Get first and last trading dates
    entry_date = df_filtered['Date'].min()
    exit_date = df_filtered['Date'].max()

    entry_row = df_filtered[df_filtered['Date'] == entry_date].iloc[0]
    exit_row = df_filtered[df_filtered['Date'] == exit_date].iloc[0]

    entry_price = entry_row['Last_Price']
    exit_price = exit_row['Last_Price']

    # Calculate gross P&L
    if position_type == 'LONG':
        gross_pnl = (exit_price - entry_price) * LOTE_SIZE
    else:  # SHORT
        gross_pnl = (entry_price - exit_price) * LOTE_SIZE

    # Calculate financing costs (daily compounding)
    cash_position = 0
    financing_pnl = 0

    if position_type == 'LONG':
        # Need to borrow money to buy stock
        cash_position = -entry_price * LOTE_SIZE
    else:
        # SHORT sale: receive proceeds which are used as margin collateral
        # With 100% margin, proceeds = margin requirement
        # The margin collateral (= proceeds) earns the risk-free rate
        cash_position = entry_price * LOTE_SIZE

    # Calculate financing day by day
    trading_dates = sorted(df_filtered['Date'].unique())
    prev_date = entry_date

    for current_date in trading_dates:
        if current_date <= entry_date:
            continue

        day_row = df[df['Date'] == current_date].iloc[0]
        risk_free = day_row.get('Risk_Free_Rate', 0.4)

        calendar_days = (current_date - prev_date).days
        daily_rate = risk_free / 365
        period_interest = cash_position * daily_rate * calendar_days
        financing_pnl += period_interest
        cash_position += period_interest  # Compound

        prev_date = current_date

    # Commission (entry + exit)
    commission = entry_price * LOTE_SIZE * COMMISSION_STOCK * 2

    # Total P&L
    total_pnl = gross_pnl + financing_pnl - commission

    # Investment calculation differs by position type:
    # LONG: Need 100% capital (no margin for buying)
    # SHORT: Need 100% margin collateral (conservative assumption)
    if position_type == 'LONG':
        investment = entry_price * LOTE_SIZE  # 100% capital required
    else:
        investment = entry_price * LOTE_SIZE * MARGIN_RATE  # 100% margin for short
    net_return = (total_pnl / investment) * 100

    return pd.DataFrame([{
        'Strategy': f'{position_type}_STOCK',
        'Entry_Date': entry_date,
        'Exit_Date': exit_date,
        'Entry_Price': entry_price,
        'Exit_Price': exit_price,
        'Gross_PnL': gross_pnl,
        'Financing_PnL': financing_pnl,
        'Commission': commission,
        'Total_PnL': total_pnl,
        'Net_Return': net_return
    }])


# ============================================================================
# RUN ALL STRATEGIES
# ============================================================================

print("\n[STEP 5] Running gamma scalping strategies...")

# 1. CALL strategy
print("\n  Running CALL gamma scalping (Model A - RV vs IV)...")
call_trades, call_hedge_log = run_gamma_scalping_strategy(df, df_options, 'CALL')
call_trades.to_csv(f'{OUTPUT_DIR}/gamma_scalp_call_trades.csv', index=False)
call_hedge_log.to_csv(f'{OUTPUT_DIR}/gamma_scalp_call_hedge_log.csv', index=False)

# 2. PUT strategy
print("\n  Running PUT gamma scalping (Model A - RV vs IV)...")
put_trades, put_hedge_log = run_gamma_scalping_strategy(df, df_options, 'PUT')
put_trades.to_csv(f'{OUTPUT_DIR}/gamma_scalp_put_trades.csv', index=False)
put_hedge_log.to_csv(f'{OUTPUT_DIR}/gamma_scalp_put_hedge_log.csv', index=False)

# 3. STRADDLE strategy
print("\n  Running STRADDLE gamma scalping (Model A - RV vs IV)...")
straddle_trades, straddle_hedge_log = run_gamma_scalping_straddle(df, df_options)
straddle_trades.to_csv(f'{OUTPUT_DIR}/gamma_scalp_straddle_trades.csv', index=False)
straddle_hedge_log.to_csv(f'{OUTPUT_DIR}/gamma_scalp_straddle_hedge_log.csv', index=False)

# 4. STRANGLE strategy
print("\n  Running STRANGLE gamma scalping (Model A - RV vs IV)...")
strangle_trades, strangle_hedge_log = run_gamma_scalping_strangle(df, df_options)
strangle_trades.to_csv(f'{OUTPUT_DIR}/gamma_scalp_strangle_trades.csv', index=False)
strangle_hedge_log.to_csv(f'{OUTPUT_DIR}/gamma_scalp_strangle_hedge_log.csv', index=False)

# 5. BASELINE strategies (LONG and SHORT stock)
# Use same test period as gamma scalping (starting from first OPEX-2 months = ~2025-02-21)
BASELINE_START = pd.Timestamp('2025-02-21')  # Same as first gamma scalping entry
print("\n  Running BASELINE strategies (LONG/SHORT stock)...")
long_stock = run_stock_strategy(df, 'LONG', start_date=BASELINE_START)
short_stock = run_stock_strategy(df, 'SHORT', start_date=BASELINE_START)
baseline_df = pd.concat([long_stock, short_stock], ignore_index=True)
baseline_df.to_csv(f'{OUTPUT_DIR}/gamma_scalp_baseline.csv', index=False)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 100)
print("GAMMA SCALPING RESULTS SUMMARY")
print("=" * 100)

# Baseline strategies first
print("\n" + "-" * 50)
print("BASELINE STRATEGIES")
print("-" * 50)
for _, row in baseline_df.iterrows():
    print(f"\n{row['Strategy']}:")
    print(f"  Entry: {row['Entry_Date'].strftime('%Y-%m-%d')} at {row['Entry_Price']:.0f}")
    print(f"  Exit:  {row['Exit_Date'].strftime('%Y-%m-%d')} at {row['Exit_Price']:.0f}")
    print(f"  Gross P&L:     {row['Gross_PnL']:+,.0f} ARS")
    print(f"  Financing P&L: {row['Financing_PnL']:+,.0f} ARS")
    print(f"  Commission:    {row['Commission']:,.0f} ARS")
    print(f"  Total P&L:     {row['Total_PnL']:+,.0f} ARS")
    margin_text = "100% capital" if row['Strategy'] == 'LONG_STOCK' else "100% margin"
    print(f"  Net Return:    {row['Net_Return']:+.1f}% ({margin_text})")

# Gamma scalping strategies
print("\n" + "-" * 50)
print("GAMMA SCALPING STRATEGIES")
print("-" * 50)

summary_data = []

for name, trades_df in [('CALL', call_trades), ('PUT', put_trades),
                        ('STRADDLE', straddle_trades), ('STRANGLE', strangle_trades)]:
    total_return = trades_df['Net_Return'].sum()
    total_trades = trades_df['Trades'].sum()
    total_pnl = trades_df['Total_PnL'].sum()
    summary_data.append({
        'Strategy': name,
        'Trades': total_trades,
        'Total_PnL': total_pnl,
        'Total_Return': total_return
    })
    print(f"\n{name} Strategy:")
    print(trades_df.to_string(index=False))
    print(f"  Total: {total_trades} trades, P&L={total_pnl:+,.0f}, Return={total_return:+.1f}%")

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f'{OUTPUT_DIR}/gamma_scalp_summary.csv', index=False)

print("\n" + "-" * 50)
print("ALL STRATEGIES COMPARISON")
print("-" * 50)

# Combine baseline and gamma scalping for comparison
all_strategies = []
for _, row in baseline_df.iterrows():
    all_strategies.append({
        'Strategy': row['Strategy'],
        'Trades': 1,
        'Total_PnL': row['Total_PnL'],
        'Total_Return': row['Net_Return']
    })
all_strategies.extend(summary_data)

all_strategies_df = pd.DataFrame(all_strategies)
all_strategies_df = all_strategies_df.sort_values('Total_Return', ascending=False)
print(all_strategies_df.to_string(index=False))

# Save complete comparison
all_strategies_df.to_csv(f'{OUTPUT_DIR}/gamma_scalp_all_comparison.csv', index=False)

for rank, (_, row) in enumerate(all_strategies_df.iterrows(), 1):
    print(f"  {rank}. {row['Strategy']:12s}: {int(row['Trades']):3d} trades, {row['Total_Return']:+.1f}% return, P&L={row['Total_PnL']:+,.0f}")

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
