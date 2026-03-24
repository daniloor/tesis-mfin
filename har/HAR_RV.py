"""
HAR-RV KUMAR (2010) MODELS - GGAL IMPLEMENTATION
=================================================

This script implements HAR model variants from Kumar (2010) plus IV-augmented extensions:
"Heterogeneous Autoregressive Model for Indian Stock Market Volatility"

Models tested (following Kumar 2010, Table 3 & 4):
-------------------------------------------------
KUMAR (2010) ORIGINAL MODELS:
1. HAR-RV      - Base model with RV (Equation 7)
2. HAR-RBV     - Using Realized Bipower Variance (Equation 19 with V=RBV)
3. HAR-TRBV    - Using Threshold Bipower Variance (Equation 19 with V=TRBV)
4. HAR-CJ-RBV  - Continuous and Jump components using RBV (Equation 20)
5. HAR-TCJ-RBV - Continuous and Jump components using TRBV (Equation 21)

IV-AUGMENTED MODELS (EXTENSION):
6. HAR-RV-IV       - HAR-RV + daily Implied Volatility
7. HAR-RBV-IV      - HAR-RBV + daily Implied Volatility
8. HAR-RV-IV-Full  - HAR-RV + IV with d/w/m temporal structure
9. HAR-RBV-IV-Full - HAR-RBV + IV with d/w/m temporal structure
10. HAR-CJ-RBV-IV  - HAR-CJ-RBV + daily Implied Volatility

Data:
-----
- GGAL intraday data at 10-minute and 30-minute frequencies
- Daily IV data from data.dat (IV_Call_Avg, IV_Put_Avg)
- Following Kumar: 80% train / 20% test split
- Metrics: MSE, MAE, MAPE (in-sample and out-of-sample)

Key Equations from Kumar (2010):
---------------------------------
RBV (Equation 8):
  RBV_t = μ₁⁻² * Σ|r_{t,j}| * |r_{t,j-1}|
  where μ₁ = √(2/π) ≈ 0.7979

Threshold z-statistic (Equation 14):
  z_{t,j} = |r_{t,j}| / √(RBV_t)

Threshold function (Equation 15):
  I_{t,j}(c) = 1 if z_{t,j} ≤ c, else 0
  (Kumar uses c = 2.5 following Corsi et al. 2009)

TRBV (Equation 16):
  TRBV_t = μ₁⁻² * Σ I_{t,j}(c) * |r_{t,j}| * |r_{t,j-1}|

Jump component (Equation 12):
  J_t = max(RV_t - RBV_t, 0)

Continuous component (Equation 13):
  C_t = RBV_t

HAR-CJ-RBV Model (Equation 20):
  RV_{t+1} = c + β_C*C_t + β_C_w*C_t^(w) + β_C_m*C_t^(m)
           + β_J*J_t + β_J_w*J_t^(w) + ε_{t+1}

LOG-HAR-CJ-RBV Model (Equation 24):
  log(RV_{t+1}) = c + β_C*log(C_t) + β_C_w*log(C_t^(w)) + β_C_m*log(C_t^(m))
                + β_J*log(1+J_t) + β_J_w*log(1+J_t^(w)) + ε_{t+1}

Author: GGAL Volatility Forecasting Thesis
Date: February 2026
Reference: Kumar, M. (2010). "Heterogeneous Autoregressive Model for Indian
           Stock Market Volatility". Decision, 37(2), 203-219.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTRADAY_DIR = os.path.join(SCRIPT_DIR, '..', 'intraday')
PROCESS_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'process_data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("HAR-RV KUMAR (2010) MODELS - GGAL IMPLEMENTATION")
print("=" * 80)
print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Create output directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("[1/10] Loading intraday data...")

# Load 10-minute data
df_10min = pd.read_csv(os.path.join(INTRADAY_DIR, 'BCBA_DLY_GGAL, 10 (1).csv'))
df_10min['time'] = pd.to_datetime(df_10min['time'])
df_10min = df_10min.sort_values('time').reset_index(drop=True)

# Load 30-minute data
df_30min = pd.read_csv(os.path.join(INTRADAY_DIR, 'BCBA_DLY_GGAL, 30 (4).csv'))
df_30min['time'] = pd.to_datetime(df_30min['time'])
df_30min = df_30min.sort_values('time').reset_index(drop=True)

print(f"  ✓ 10-minute data: {len(df_10min):,} bars")
print(f"    Period: {df_10min['time'].min().date()} to {df_10min['time'].max().date()}")
print(f"  ✓ 30-minute data: {len(df_30min):,} bars")
print(f"    Period: {df_30min['time'].min().date()} to {df_30min['time'].max().date()}")

# Load daily data with IV (data.dat)
print("\n  Loading daily data with Implied Volatility (data.dat)...")
df_daily_iv = pd.read_csv(os.path.join(PROCESS_DATA_DIR, 'data.dat'))
df_daily_iv['Date'] = pd.to_datetime(df_daily_iv['Date'])
df_daily_iv = df_daily_iv.sort_values('Date').reset_index(drop=True)

# Extract IV columns
df_iv = df_daily_iv[['Date', 'IV_Call_Avg', 'IV_Put_Avg']].copy()
df_iv.columns = ['date', 'IV_Call', 'IV_Put']
df_iv['IV_Avg'] = (df_iv['IV_Call'] + df_iv['IV_Put']) / 2
df_iv['IV_Spread'] = df_iv['IV_Put'] - df_iv['IV_Call']  # Put-Call spread

print(f"  ✓ Daily IV data: {len(df_iv):,} days")
print(f"    Period: {df_iv['date'].min().date()} to {df_iv['date'].max().date()}")
print(f"    Mean IV_Call: {df_iv['IV_Call'].mean():.4f}")
print(f"    Mean IV_Put:  {df_iv['IV_Put'].mean():.4f}")
print(f"    Mean IV_Avg:  {df_iv['IV_Avg'].mean():.4f}")

# =============================================================================
# 2. CALCULATE VOLATILITY MEASURES (Kumar 2010 methodology)
# =============================================================================
print("\n[2/10] Calculating Kumar (2010) volatility measures...")

def calc_kumar_volatility(df, min_bars=10, threshold_c=2.5):
    """
    Calculate all volatility measures from Kumar (2010)

    Returns daily DataFrame with:
    - RV: Realized Volatility (Equation 6)
    - RBV: Realized Bipower Variance (Equation 8)
    - TRBV: Threshold Realized Bipower Variance (Equation 16)
    - J_RBV: Jump component using RBV (Equation 12)
    - C_RBV: Continuous component using RBV (Equation 13)
    - J_TRBV: Jump component using TRBV (Equation 17)
    - C_TRBV: Continuous component using TRBV (Equation 18)
    """
    df = df.copy()
    df['date'] = df['time'].dt.date

    # Calculate log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Reset returns at day boundaries
    df['prev_date'] = df['date'].shift(1)
    df.loc[df['date'] != df['prev_date'], 'log_return'] = np.nan

    # Also get lagged return for bipower variance
    df['log_return_lag1'] = df['log_return'].shift(1)
    # Reset lagged returns at day boundaries
    df.loc[df['date'] != df['prev_date'], 'log_return_lag1'] = np.nan

    # Calculate daily measures
    daily_list = []

    for date, group in df.groupby('date'):
        # Filter valid returns
        returns = group['log_return'].dropna()

        if len(returns) < min_bars:
            continue

        # 1. Realized Variance (Equation 6)
        RV = (returns ** 2).sum()

        # 2. Realized Bipower Variance (Equation 8)
        # RBV = μ₁⁻² * Σ|r_{t,j}| * |r_{t,j-1}|
        # where μ₁ = √(2/π)
        mu_1 = np.sqrt(2 / np.pi)

        # Get absolute returns and lagged absolute returns
        abs_returns = group['log_return'].abs()
        abs_returns_lag1 = group['log_return_lag1'].abs()

        # Bipower product (needs both current and lagged return to be valid)
        bipower_products = abs_returns * abs_returns_lag1
        bipower_products = bipower_products.dropna()

        if len(bipower_products) > 0:
            RBV = (mu_1 ** -2) * bipower_products.sum()
        else:
            RBV = RV  # Fallback to RV if no valid bipower pairs

        # 3. Threshold Realized Bipower Variance (Equation 14-16)
        # First, calculate z-statistics: z_{t,j} = |r_{t,j}| / √RBV
        if RBV > 0:
            z_stats = abs_returns / np.sqrt(RBV)

            # Threshold indicator: I(c) = 1 if z ≤ c, else 0
            threshold_indicator = (z_stats <= threshold_c).astype(float)

            # Shift to align with lagged returns
            threshold_indicator_for_bipower = threshold_indicator.shift(0)

            # TRBV: only include bipower products where threshold is met
            trbv_products = bipower_products * threshold_indicator_for_bipower.loc[bipower_products.index]
            trbv_products = trbv_products.dropna()

            if len(trbv_products) > 0:
                TRBV = (mu_1 ** -2) * trbv_products.sum()
            else:
                TRBV = RBV  # Fallback
        else:
            TRBV = RBV

        # 4. Jump and Continuous Components
        # Using RBV (Equations 12-13)
        J_RBV = max(RV - RBV, 0)
        C_RBV = RBV

        # Using TRBV (Equations 17-18)
        J_TRBV = max(RV - TRBV, 0)
        C_TRBV = TRBV

        # Store results
        # RV is stored as VOLATILITY (standard deviation), NOT variance, NOT percentage
        # This follows Kumar (2010) where RV represents realized volatility
        # MSE will be computed on this RV level (after exp() for log models)
        daily_list.append({
            'date': pd.to_datetime(date),
            'RV': np.sqrt(RV),       # Volatility (std dev of returns), NOT percentage
            'RBV': np.sqrt(RBV),     # Realized Bipower Volatility
            'TRBV': np.sqrt(TRBV),   # Threshold Realized Bipower Volatility
            'J_RBV': np.sqrt(J_RBV), # Jump component volatility
            'C_RBV': np.sqrt(C_RBV), # Continuous component volatility
            'J_TRBV': np.sqrt(J_TRBV),
            'C_TRBV': np.sqrt(C_TRBV),
            'n_bars': len(returns)
        })

    return pd.DataFrame(daily_list)

# Calculate for 10-minute data
df_vol_10min = calc_kumar_volatility(df_10min, min_bars=10)
print(f"\n  ✓ 10-minute volatility measures: {len(df_vol_10min):,} days")
print(f"    Period: {df_vol_10min['date'].min().date()} to {df_vol_10min['date'].max().date()}")
print(f"    Mean RV:   {df_vol_10min['RV'].mean():.6f} (volatility, not %)")
print(f"    Mean RBV:  {df_vol_10min['RBV'].mean():.6f}")
print(f"    Mean TRBV: {df_vol_10min['TRBV'].mean():.6f}")
print(f"    Mean RV (as %): {df_vol_10min['RV'].mean()*100:.4f}%")

# Calculate for 30-minute data
df_vol_30min = calc_kumar_volatility(df_30min, min_bars=5)
print(f"\n  ✓ 30-minute volatility measures: {len(df_vol_30min):,} days")
print(f"    Period: {df_vol_30min['date'].min().date()} to {df_vol_30min['date'].max().date()}")
print(f"    Mean RV:   {df_vol_30min['RV'].mean():.6f} (volatility, not %)")
print(f"    Mean RBV:  {df_vol_30min['RBV'].mean():.6f}")
print(f"    Mean TRBV: {df_vol_30min['TRBV'].mean():.6f}")
print(f"    Mean RV (as %): {df_vol_30min['RV'].mean()*100:.4f}%")

# =============================================================================
# 2.5 MERGE IV DATA WITH VOLATILITY MEASURES
# =============================================================================
print("\n[2.5/10] Merging IV data with volatility measures...")

# Merge IV with 10-minute volatility data
df_vol_10min = df_vol_10min.merge(df_iv, on='date', how='inner')
print(f"  ✓ 10-minute + IV merged: {len(df_vol_10min):,} days")
print(f"    Period: {df_vol_10min['date'].min().date()} to {df_vol_10min['date'].max().date()}")

# Merge IV with 30-minute volatility data
df_vol_30min = df_vol_30min.merge(df_iv, on='date', how='inner')
print(f"  ✓ 30-minute + IV merged: {len(df_vol_30min):,} days")
print(f"    Period: {df_vol_30min['date'].min().date()} to {df_vol_30min['date'].max().date()}")

# =============================================================================
# 3. CREATE HAR FEATURES (Kumar 2010 specifications)
# =============================================================================
print("\n[3/10] Creating HAR features (Kumar 2010 specifications)...")

def create_kumar_har_features(df):
    """
    Create all HAR features for Kumar (2010) models

    Returns DataFrame with features for all 10 models
    """
    df = df.copy()

    # Target variables for multiple horizons
    df['log_RV'] = np.log(df['RV'])

    # 1-day horizon: next day's log RV
    df['y_1d'] = df['log_RV'].shift(-1)

    # 5-day horizon: average of next 5 days' log RV
    df['y_5d'] = df['log_RV'].rolling(5).mean().shift(-5)

    # 22-day horizon: average of next 22 days' log RV
    df['y_22d'] = df['log_RV'].rolling(22).mean().shift(-22)

    # Keep 'y' as alias for 1-day for backward compatibility
    df['y'] = df['y_1d']

    # Log transforms of all volatility measures
    df['log_RBV'] = np.log(df['RBV'])
    df['log_TRBV'] = np.log(df['TRBV'])
    df['log_C_RBV'] = np.log(df['C_RBV'])
    df['log_C_TRBV'] = np.log(df['C_TRBV'])

    # For jumps, use log(1+J) to avoid log(0)
    df['log1p_J_RBV'] = np.log(1 + df['J_RBV'])
    df['log1p_J_TRBV'] = np.log(1 + df['J_TRBV'])

    # --- HAR-RV Features (Model 1) ---
    df['RV_d'] = df['log_RV'].shift(1)
    df['RV_w'] = df['log_RV'].rolling(5).mean().shift(1)
    df['RV_m'] = df['log_RV'].rolling(22).mean().shift(1)

    # --- HAR-RBV Features (Model 2) ---
    df['RBV_d'] = df['log_RBV'].shift(1)
    df['RBV_w'] = df['log_RBV'].rolling(5).mean().shift(1)
    df['RBV_m'] = df['log_RBV'].rolling(22).mean().shift(1)

    # --- HAR-TRBV Features (Model 3) ---
    df['TRBV_d'] = df['log_TRBV'].shift(1)
    df['TRBV_w'] = df['log_TRBV'].rolling(5).mean().shift(1)
    df['TRBV_m'] = df['log_TRBV'].rolling(22).mean().shift(1)

    # --- HAR-CJ-RBV Features (Model 4) ---
    # Continuous component
    df['C_RBV_d'] = df['log_C_RBV'].shift(1)
    df['C_RBV_w'] = df['log_C_RBV'].rolling(5).mean().shift(1)
    df['C_RBV_m'] = df['log_C_RBV'].rolling(22).mean().shift(1)
    # Jump component
    df['J_RBV_d'] = df['log1p_J_RBV'].shift(1)
    df['J_RBV_w'] = df['log1p_J_RBV'].rolling(5).mean().shift(1)

    # --- HAR-TCJ-RBV Features (Model 5) ---
    # Continuous component
    df['C_TRBV_d'] = df['log_C_TRBV'].shift(1)
    df['C_TRBV_w'] = df['log_C_TRBV'].rolling(5).mean().shift(1)
    df['C_TRBV_m'] = df['log_C_TRBV'].rolling(22).mean().shift(1)
    # Jump component
    df['J_TRBV_d'] = df['log1p_J_TRBV'].shift(1)
    df['J_TRBV_w'] = df['log1p_J_TRBV'].rolling(5).mean().shift(1)

    # --- IV Features (for IV-augmented models) ---
    # Check if IV columns exist in the dataframe
    if 'IV_Avg' in df.columns:
        # Log transform IV (IV is already in decimal form, e.g., 0.42 = 42%)
        df['log_IV_Avg'] = np.log(df['IV_Avg'])
        df['log_IV_Call'] = np.log(df['IV_Call'])
        df['log_IV_Put'] = np.log(df['IV_Put'])

        # Daily IV features (lagged by 1 day to avoid look-ahead bias)
        df['IV_d'] = df['log_IV_Avg'].shift(1)
        df['IV_Call_d'] = df['log_IV_Call'].shift(1)
        df['IV_Put_d'] = df['log_IV_Put'].shift(1)

        # Weekly IV features (5-day rolling average, lagged)
        df['IV_w'] = df['log_IV_Avg'].rolling(5).mean().shift(1)

        # Monthly IV features (22-day rolling average, lagged)
        df['IV_m'] = df['log_IV_Avg'].rolling(22).mean().shift(1)

        # IV Spread (Put-Call, can capture skew/sentiment)
        df['IV_Spread_d'] = df['IV_Spread'].shift(1)

    return df

# Create features for 10-minute data
df_10min_features = create_kumar_har_features(df_vol_10min)
print(f"  ✓ 10-minute HAR features created: {len(df_10min_features):,} days")

# Create features for 30-minute data
df_30min_features = create_kumar_har_features(df_vol_30min)
print(f"  ✓ 30-minute HAR features created: {len(df_30min_features):,} days")

# =============================================================================
# 4. DEFINE KUMAR (2010) MODEL SPECIFICATIONS
# =============================================================================
print("\n[4/10] Defining Kumar (2010) model specifications...")

KUMAR_MODELS = {
    # Standard HAR models (Models 1-5)
    'HAR-RV': {
        'features': ['RV_d', 'RV_w', 'RV_m'],
        'description': 'Base HAR-RV model (Kumar Eq. 7)',
        'equation': 'log(RV_{t+1}) = c + β_d*log(RV_t) + β_w*log(RV_t^w) + β_m*log(RV_t^m) + ε'
    },
    'HAR-RBV': {
        'features': ['RBV_d', 'RBV_w', 'RBV_m'],
        'description': 'HAR with Realized Bipower Variance (Kumar Eq. 19, V=RBV)',
        'equation': 'log(RV_{t+1}) = c + β_d*log(RBV_t) + β_w*log(RBV_t^w) + β_m*log(RBV_t^m) + ε'
    },
    'HAR-TRBV': {
        'features': ['TRBV_d', 'TRBV_w', 'TRBV_m'],
        'description': 'HAR with Threshold Bipower Variance (Kumar Eq. 19, V=TRBV)',
        'equation': 'log(RV_{t+1}) = c + β_d*log(TRBV_t) + β_w*log(TRBV_t^w) + β_m*log(TRBV_t^m) + ε'
    },
    'HAR-CJ-RBV': {
        'features': ['C_RBV_d', 'C_RBV_w', 'C_RBV_m', 'J_RBV_d', 'J_RBV_w'],
        'description': 'HAR with Continuous/Jump decomposition using RBV (Kumar Eq. 20)',
        'equation': 'log(RV_{t+1}) = c + β_C*log(C_t) + β_C_w*log(C_t^w) + β_C_m*log(C_t^m) + β_J*log(1+J_t) + β_J_w*log(1+J_t^w) + ε'
    },
    'HAR-TCJ-RBV': {
        'features': ['C_TRBV_d', 'C_TRBV_w', 'C_TRBV_m', 'J_TRBV_d', 'J_TRBV_w'],
        'description': 'HAR with Continuous/Jump decomposition using TRBV (Kumar Eq. 21)',
        'equation': 'log(RV_{t+1}) = c + β_C*log(C_t) + β_C_w*log(C_t^w) + β_C_m*log(C_t^m) + β_J*log(1+J_t) + β_J_w*log(1+J_t^w) + ε'
    },

    # ==========================================================================
    # IV-AUGMENTED HAR MODELS (Extension of Kumar 2010 with Implied Volatility)
    # ==========================================================================
    # These models add forward-looking information from options market

    'HAR-RV-IV': {
        'features': ['RV_d', 'RV_w', 'RV_m', 'IV_d'],
        'description': 'HAR-RV augmented with daily Implied Volatility',
        'equation': 'log(RV_{t+1}) = c + β_d*log(RV_t) + β_w*log(RV_t^w) + β_m*log(RV_t^m) + β_IV*log(IV_t) + ε'
    },
    'HAR-RBV-IV': {
        'features': ['RBV_d', 'RBV_w', 'RBV_m', 'IV_d'],
        'description': 'HAR-RBV augmented with daily Implied Volatility',
        'equation': 'log(RV_{t+1}) = c + β_d*log(RBV_t) + β_w*log(RBV_t^w) + β_m*log(RBV_t^m) + β_IV*log(IV_t) + ε'
    },
    'HAR-RV-IV-Full': {
        'features': ['RV_d', 'RV_w', 'RV_m', 'IV_d', 'IV_w', 'IV_m'],
        'description': 'HAR-RV with full IV temporal structure (d/w/m)',
        'equation': 'log(RV_{t+1}) = c + β_RV*log(RV_t^d/w/m) + β_IV*log(IV_t^d/w/m) + ε'
    },
    'HAR-RBV-IV-Full': {
        'features': ['RBV_d', 'RBV_w', 'RBV_m', 'IV_d', 'IV_w', 'IV_m'],
        'description': 'HAR-RBV with full IV temporal structure (d/w/m)',
        'equation': 'log(RV_{t+1}) = c + β_RBV*log(RBV_t^d/w/m) + β_IV*log(IV_t^d/w/m) + ε'
    },
    'HAR-CJ-RBV-IV': {
        'features': ['C_RBV_d', 'C_RBV_w', 'C_RBV_m', 'J_RBV_d', 'J_RBV_w', 'IV_d'],
        'description': 'HAR-CJ-RBV augmented with daily Implied Volatility',
        'equation': 'log(RV_{t+1}) = c + β_C*log(C_t^d/w/m) + β_J*log(1+J_t^d/w) + β_IV*log(IV_t) + ε'
    }
}

print(f"  ✓ Defined {len(KUMAR_MODELS)} model specifications")
print("\n  Kumar (2010) Original Models:")
for name, spec in KUMAR_MODELS.items():
    if 'IV' not in name:
        print(f"    - {name}: {len(spec['features'])} features")
print("\n  IV-Augmented Models (Extension):")
for name, spec in KUMAR_MODELS.items():
    if 'IV' in name:
        print(f"    - {name}: {len(spec['features'])} features")

# =============================================================================
# 5. TRAIN-TEST SPLIT (Following Kumar 2010: 80/20)
# =============================================================================
print("\n[5/10] Creating train-test split (80% train / 20% test)...")

def split_data(df, train_ratio=0.8):
    """Split data into train/test following Kumar (2010)"""
    df_clean = df.dropna()

    split_idx = int(len(df_clean) * train_ratio)
    train = df_clean.iloc[:split_idx]
    test = df_clean.iloc[split_idx:]

    return train, test

# Split 10-minute data
train_10min, test_10min = split_data(df_10min_features)
print(f"  ✓ 10-minute split:")
print(f"    Training:   {len(train_10min):,} days ({train_10min['date'].min().date()} to {train_10min['date'].max().date()})")
print(f"    Test:       {len(test_10min):,} days ({test_10min['date'].min().date()} to {test_10min['date'].max().date()})")

# Split 30-minute data
train_30min, test_30min = split_data(df_30min_features)
print(f"  ✓ 30-minute split:")
print(f"    Training:   {len(train_30min):,} days ({train_30min['date'].min().date()} to {train_30min['date'].max().date()})")
print(f"    Test:       {len(test_30min):,} days ({test_30min['date'].min().date()} to {test_30min['date'].max().date()})")

# =============================================================================
# 6. ESTIMATION FUNCTION (Following Kumar 2010 methodology)
# =============================================================================

def estimate_kumar_model(train, test, features, model_name, target_col='y_1d'):
    """
    Estimate HAR model following Kumar (2010) methodology

    IMPORTANT: Following Kumar (2010), Section 3.2, Page 211:
    "after recovering the dependent variable to its original form,
    we have compared the in-sample and out-of-sample of different
    models using MSE, MAE and MAPE"

    MSE/MAE/MAPE are computed on RV-level (VOLATILITY = std dev of returns):
    - NOT on log-scale
    - NOT on variance (variance = RV²)
    - NOT as percentage (percentage = RV × 100)

    RV is stored as volatility (standard deviation), typically ~0.02-0.03
    for daily returns. This is consistent with Kumar's Table 1 where
    Mean RV = 0.00082 (which is variance; sqrt = 0.0286 volatility).

    Args:
        train: Training DataFrame
        test: Test DataFrame
        features: List of feature column names
        model_name: Name of the model
        target_col: Target column ('y_1d', 'y_5d', or 'y_22d')

    Returns:
        Dictionary with in-sample and out-of-sample metrics
    """
    # Filter rows with valid target values
    train_valid = train.dropna(subset=[target_col] + features)
    test_valid = test.dropna(subset=[target_col] + features)

    # Prepare training data
    X_train = train_valid[features].values
    y_train = train_valid[target_col].values  # log-scale target (log of RV)

    # Prepare test data
    X_test = test_valid[features].values
    y_test = test_valid[target_col].values  # log-scale target (log of RV)

    # Fit model (OLS regression, following Kumar 2010)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # In-sample predictions (log-scale)
    y_train_pred_log = model.predict(X_train)

    # Out-of-sample predictions (log-scale)
    y_test_pred_log = model.predict(X_test)

    # =========================================================================
    # Following Kumar (2010): Transform back to RV-level for error metrics
    # RV-level means VOLATILITY (std dev), not variance, not percentage
    # =========================================================================
    # Convert from log-scale to RV-level (original form = volatility)
    y_train_level = np.exp(y_train)        # RV = exp(log(RV))
    y_train_pred_level = np.exp(y_train_pred_log)
    y_test_level = np.exp(y_test)
    y_test_pred_level = np.exp(y_test_pred_log)

    # R² computed on log-scale (as in Kumar's adjusted R²)
    r2_train = r2_score(y_train, y_train_pred_log)
    r2_test = r2_score(y_test, y_test_pred_log)

    # MSE, RMSE, MAE, MAPE computed on RV-level (VOLATILITY, following Kumar 2010)
    # Units: volatility (std dev of returns), e.g., 0.025 = 2.5% daily volatility
    # In-sample metrics (RV-level = volatility)
    mse_train = mean_squared_error(y_train_level, y_train_pred_level)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train_level, y_train_pred_level)
    mape_train = np.mean(np.abs((y_train_level - y_train_pred_level) / y_train_level)) * 100

    # Out-of-sample metrics (RV-level = volatility)
    mse_test = mean_squared_error(y_test_level, y_test_pred_level)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test_level, y_test_pred_level)
    mape_test = np.mean(np.abs((y_test_level - y_test_pred_level) / y_test_level)) * 100
    corr_test = np.corrcoef(y_test_level, y_test_pred_level)[0, 1]

    # Direction accuracy (additional metric, on log-scale changes)
    y_train_change = np.diff(y_train)
    y_train_pred_change = np.diff(y_train_pred_log)
    dir_acc_train = np.mean((y_train_change * y_train_pred_change) > 0)

    y_test_change = np.diff(y_test)
    y_test_pred_change = np.diff(y_test_pred_log)
    dir_acc_test = np.mean((y_test_change * y_test_pred_change) > 0)

    return {
        'model': model_name,
        'features': features,
        'n_features': len(features),
        'coefficients': dict(zip(features, model.coef_)),
        'intercept': model.intercept_,

        # In-sample (MSE/MAE/MAPE on RV-level, R² on log-scale)
        'R2_InSample': r2_train,
        'MSE_InSample': mse_train,
        'RMSE_InSample': rmse_train,
        'MAE_InSample': mae_train,
        'MAPE_InSample': mape_train,
        'DirAcc_InSample': dir_acc_train,

        # Out-of-sample (MSE/MAE/MAPE on RV-level, R² on log-scale)
        'R2_OutSample': r2_test,
        'MSE_OutSample': mse_test,
        'RMSE_OutSample': rmse_test,
        'MAE_OutSample': mae_test,
        'MAPE_OutSample': mape_test,
        'Correlation_OutSample': corr_test,
        'DirAcc_OutSample': dir_acc_test,

        # Sample sizes
        'Train_Size': len(X_train),
        'Test_Size': len(X_test),

        # Dates
        'Train_Start': train_valid['date'].min(),
        'Train_End': train_valid['date'].max(),
        'Test_Start': test_valid['date'].min(),
        'Test_End': test_valid['date'].max(),
        'target_col': target_col
    }

# =============================================================================
# 7. ESTIMATE ALL MODELS - 10-MINUTE DATA (ALL HORIZONS)
# =============================================================================
print("\n[6/10] Estimating Kumar (2010) models on 10-minute data (all horizons)...")

HORIZONS = {
    '1d': 'y_1d',
    '5d': 'y_5d',
    '22d': 'y_22d'
}

results_10min = []

for horizon_name, target_col in HORIZONS.items():
    print(f"\n  === {horizon_name.upper()} HORIZON ===")

    for model_name, spec in KUMAR_MODELS.items():
        print(f"    Estimating {model_name}...")

        result = estimate_kumar_model(
            train_10min,
            test_10min,
            spec['features'],
            f"10min_{model_name}",
            target_col=target_col
        )
        result['horizon'] = horizon_name
        result['data_freq'] = '10min'
        result['model_short'] = model_name

        results_10min.append(result)

        print(f"      ✓ R² (out-sample): {result['R2_OutSample']:.4f}, MSE: {result['MSE_OutSample']:.2E}")

# =============================================================================
# 8. ESTIMATE ALL MODELS - 30-MINUTE DATA (ALL HORIZONS)
# =============================================================================
print("\n[7/10] Estimating Kumar (2010) models on 30-minute data (all horizons)...")

results_30min = []

for horizon_name, target_col in HORIZONS.items():
    print(f"\n  === {horizon_name.upper()} HORIZON ===")

    for model_name, spec in KUMAR_MODELS.items():
        print(f"    Estimating {model_name}...")

        result = estimate_kumar_model(
            train_30min,
            test_30min,
            spec['features'],
            f"30min_{model_name}",
            target_col=target_col
        )
        result['horizon'] = horizon_name
        result['data_freq'] = '30min'
        result['model_short'] = model_name

        results_30min.append(result)

        print(f"      ✓ R² (out-sample): {result['R2_OutSample']:.4f}, MSE: {result['MSE_OutSample']:.2E}")

# =============================================================================
# 9. COMPILE RESULTS
# =============================================================================
print("\n[8/10] Compiling results...")

# Combine all results
all_results = results_10min + results_30min

# Create comprehensive results DataFrame
df_results = pd.DataFrame(all_results)

# Reorder columns for clarity
column_order = [
    'model', 'model_short', 'data_freq', 'horizon', 'n_features', 'Train_Size', 'Test_Size',
    'Train_Start', 'Train_End', 'Test_Start', 'Test_End',
    # In-sample
    'R2_InSample', 'MSE_InSample', 'RMSE_InSample', 'MAE_InSample', 'MAPE_InSample', 'DirAcc_InSample',
    # Out-of-sample
    'R2_OutSample', 'MSE_OutSample', 'RMSE_OutSample', 'MAE_OutSample', 'MAPE_OutSample',
    'Correlation_OutSample', 'DirAcc_OutSample',
    # Model details
    'features', 'coefficients', 'intercept', 'target_col'
]

df_results = df_results[column_order]

# Save to CSV
df_results.to_csv(os.path.join(RESULTS_DIR, 'kumar_models_comprehensive.csv'), index=False)
print(f"  ✓ Saved comprehensive results to: {RESULTS_DIR}/kumar_models_comprehensive.csv")

# =============================================================================
# 10. PRINT SUMMARY TABLES (Following Kumar 2010, Tables 3 & 4 format)
# =============================================================================
print("\n[9/10] Generating summary tables...")

print("\n" + "=" * 120)
print("OUT-OF-SAMPLE PERFORMANCE BY HORIZON (Following Kumar 2010, Table 4)")
print("=" * 120)

for data_freq in ['10min', '30min']:
    print(f"\n{'=' * 60}")
    print(f"{data_freq.upper()} DATA")
    print(f"{'=' * 60}")

    for horizon in ['1d', '5d', '22d']:
        df_subset = df_results[(df_results['data_freq'] == data_freq) &
                               (df_results['horizon'] == horizon)].copy()
        df_subset = df_subset.sort_values('MSE_OutSample')

        print(f"\n  --- {horizon.upper()} HORIZON ---")
        print(f"  {'Model':<15} {'R²':>8} {'MSE':>12} {'RMSE':>10} {'MAE':>10} {'MAPE':>8} {'Corr':>8}")
        print("  " + "-" * 75)

        for rank, (_, row) in enumerate(df_subset.iterrows(), 1):
            marker = "🏆" if rank == 1 else "  "
            print(f"  {marker}{row['model_short']:<13} {row['R2_OutSample']:>8.4f} {row['MSE_OutSample']:>12.2E} "
                  f"{row['RMSE_OutSample']:>10.4f} {row['MAE_OutSample']:>10.4f} {row['MAPE_OutSample']:>8.2f} "
                  f"{row['Correlation_OutSample']:>8.4f}")

        # Best model for this horizon
        best = df_subset.iloc[0]
        print(f"\n  Best: {best['model_short']} (MSE={best['MSE_OutSample']:.2E})")

# =============================================================================
# 11. IDENTIFY BEST MODELS BY HORIZON
# =============================================================================
print("\n[10/10] Summary: Best Models by Horizon...")

print("\n" + "=" * 100)
print("SUMMARY: BEST MODELS BY DATA FREQUENCY AND HORIZON")
print("=" * 100)

summary_results = []

for data_freq in ['10min', '30min']:
    print(f"\n{data_freq.upper()} DATA:")
    print("-" * 60)

    for horizon in ['1d', '5d', '22d']:
        df_subset = df_results[(df_results['data_freq'] == data_freq) &
                               (df_results['horizon'] == horizon)].copy()
        df_subset = df_subset.sort_values('MSE_OutSample')
        best = df_subset.iloc[0]

        print(f"  {horizon:>3} horizon: 🏆 {best['model_short']:<15} "
              f"MSE={best['MSE_OutSample']:.2E}  R²={best['R2_OutSample']:.4f}  "
              f"Corr={best['Correlation_OutSample']:.4f}")

        summary_results.append({
            'data_freq': data_freq,
            'horizon': horizon,
            'best_model': best['model_short'],
            'MSE': best['MSE_OutSample'],
            'R2': best['R2_OutSample'],
            'Corr': best['Correlation_OutSample']
        })

# Save summary
df_summary = pd.DataFrame(summary_results)
df_summary.to_csv(os.path.join(RESULTS_DIR, 'kumar_best_by_horizon.csv'), index=False)

# Create comparison plot
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Kumar (2010) HAR Models: Out-of-Sample MSE by Horizon', fontsize=14, fontweight='bold')

for i, data_freq in enumerate(['10min', '30min']):
    for j, horizon in enumerate(['1d', '5d', '22d']):
        ax = axes[i, j]
        df_subset = df_results[(df_results['data_freq'] == data_freq) &
                               (df_results['horizon'] == horizon)].copy()
        df_subset = df_subset.sort_values('MSE_OutSample')

        colors = ['green' if x == df_subset['MSE_OutSample'].min() else 'steelblue'
                  for x in df_subset['MSE_OutSample']]

        bars = ax.barh(df_subset['model_short'], df_subset['MSE_OutSample'], color=colors)
        ax.set_xlabel('MSE (Out-of-Sample)')
        ax.set_title(f'{data_freq} - {horizon} Horizon', fontweight='bold')
        ax.invert_yaxis()

        # Add value labels
        for bar, val in zip(bars, df_subset['MSE_OutSample']):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2E}',
                    va='center', ha='left', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'kumar_mse_by_horizon.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  ✓ Saved plot: {RESULTS_DIR}/kumar_mse_by_horizon.png")

# Create R² comparison plot
# NOTE: R² is shown for diagnostic purposes only. Following Kumar (2010) Section 3.2,
# MSE is the primary metric for model comparison. R² cannot be used to compare models
# with different target transformations (e.g., log vs level).
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Kumar (2010) HAR Models: Out-of-Sample R² by Horizon\n(For diagnostics only - use MSE for model comparison)',
             fontsize=14, fontweight='bold')

for i, data_freq in enumerate(['10min', '30min']):
    for j, horizon in enumerate(['1d', '5d', '22d']):
        ax = axes[i, j]
        df_subset = df_results[(df_results['data_freq'] == data_freq) &
                               (df_results['horizon'] == horizon)].copy()
        df_subset = df_subset.sort_values('R2_OutSample', ascending=False)

        colors = ['green' if x == df_subset['R2_OutSample'].max() else 'steelblue'
                  for x in df_subset['R2_OutSample']]

        bars = ax.barh(df_subset['model_short'], df_subset['R2_OutSample'], color=colors)
        ax.set_xlabel('R² (Out-of-Sample)')
        ax.set_title(f'{data_freq} - {horizon} Horizon', fontweight='bold')
        ax.invert_yaxis()

        # Add value labels
        for bar, val in zip(bars, df_subset['R2_OutSample']):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                    va='center', ha='left', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'kumar_r2_by_horizon.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved plot: {RESULTS_DIR}/kumar_r2_by_horizon.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nResults saved to:")
print(f"  - {RESULTS_DIR}/kumar_models_comprehensive.csv (all results)")
print(f"  - {RESULTS_DIR}/kumar_best_by_horizon.csv (summary)")
print(f"  - {RESULTS_DIR}/kumar_mse_by_horizon.png (MSE comparison - PRIMARY METRIC)")
print(f"  - {RESULTS_DIR}/kumar_r2_by_horizon.png (R² - for diagnostics only)")
print("\nNote: Following Kumar (2010) Section 3.2, MSE is the primary metric for model")
print("comparison. R² is shown for diagnostics but cannot compare models across different")
print("target transformations (log vs level).")
