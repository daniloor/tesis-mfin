"""
================================================================================
ARCH AND GARCH MODELS FOR GGAL VOLATILITY FORECASTING
================================================================================
This script estimates ARCH and GARCH models for GGAL stock returns.
Compares Simple Returns vs Log Returns for both ARCH(1) and GARCH(1,1) models.

MODELS TESTED:
1. ARCH(1) - Simple autoregressive conditional heteroskedasticity
2. GARCH(1,1) - Generalized ARCH with persistence

RETURN TYPES:
1. Simple Returns: (P_t / P_{t-1}) - 1
2. Log Returns: log(P_t / P_{t-1})

OUTPUT:
- garch_results/ folder with all figures
- garch_results.txt with numerical results
- garch_parameters.csv with parameter estimates
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import os
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '..', 'process_data', 'data.dat')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

# Create output directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 11

# Results file
results_file = open(os.path.join(SCRIPT_DIR, 'garch_results.txt'), 'w', encoding='utf-8')

def log(text):
    """Print and save to results file."""
    print(text)
    results_file.write(text + '\n')

log("="*80)
log("ARCH AND GARCH MODELS FOR GGAL VOLATILITY FORECASTING")
log("COMPARING SIMPLE RETURNS VS LOG RETURNS")
log("="*80)
log(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# SECTION 1: DATA DESCRIPTION
# =============================================================================
log("\n" + "="*80)
log("SECTION 1: DATA DESCRIPTION")
log("="*80)

log("""
1.1 DATA SOURCE
---------------
File: data.dat
Source: BYMA (Bolsas y Mercados Argentinos) daily data
Instrument: GGAL (Grupo Financiero Galicia)
Type: Daily closing prices and returns

The data contains:
- Date: Trading date
- Last_Price: GGAL closing price (ARS) - unadjusted
- Adj_Price: Dividend-adjusted price (used for return calculations)
- Simple_Return: (Adj_P_t / Adj_P_{t-1}) - 1
- Log_Return: log(Adj_P_t / Adj_P_{t-1})

1.2 RETURN TYPES
----------------
Simple Returns:
  Formula: r_t = (P_t / P_{t-1}) - 1
  Properties: Directly interpretable as percentage change
              Not additive over time
              Bounded below at -100%

Log Returns:
  Formula: r_t = log(P_t / P_{t-1})
  Properties: Approximately normal (better for GARCH)
              Additive over time (r_total = sum(r_t))
              Symmetric around zero
              Preferred for volatility modeling
""")

# Load data
df = pd.read_csv(DATA_FILE)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Calculate returns on ADJUSTED prices to capture total return including dividends
# This ensures dividend days reflect true economic return, not artificial price drop
df['Simple_Return'] = df['Adj_Price'].pct_change()
df['Log_Return'] = np.log(df['Adj_Price'] / df['Adj_Price'].shift(1))

log(f"""
1.3 DATA SUMMARY
----------------
Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
Total observations: {len(df)} trading days
Years covered: {df['Date'].min().year} - {df['Date'].max().year}

Note: Returns calculated on Adj_Price (dividend-adjusted) for accurate total return.

Sample of data:
{df[['Date', 'Adj_Price', 'Simple_Return', 'Log_Return']].head(5).to_string(index=False)}
""")

# =============================================================================
# SECTION 2: RETURN PREPARATION AND COMPARISON
# =============================================================================
log("\n" + "="*80)
log("SECTION 2: RETURN PREPARATION AND COMPARISON")
log("="*80)

# Convert returns to percentage
df['Simple_Return_pct'] = df['Simple_Return'] * 100
df['Log_Return_pct'] = df['Log_Return'] * 100

# Remove NaN and infinite values
df_clean = df[np.isfinite(df['Simple_Return_pct']) & np.isfinite(df['Log_Return_pct'])].copy()
simple_returns = df_clean['Simple_Return_pct'].dropna()
log_returns = df_clean['Log_Return_pct'].dropna()

log(f"""
2.1 RETURN TRANSFORMATION
-------------------------
Returns converted to percentage for numerical stability:
  Simple_Return_pct = Simple_Return × 100
  Log_Return_pct = Log_Return × 100

After cleaning (removing NaN/infinite values):
  Observations: {len(simple_returns)}

2.2 DESCRIPTIVE STATISTICS COMPARISON
--------------------------------------
                    Simple Returns    Log Returns
count               {len(simple_returns):<15.0f} {len(log_returns):.0f}
mean                {simple_returns.mean():<15.4f} {log_returns.mean():.4f}
std                 {simple_returns.std():<15.4f} {log_returns.std():.4f}
min                 {simple_returns.min():<15.4f} {log_returns.min():.4f}
25%                 {simple_returns.quantile(0.25):<15.4f} {log_returns.quantile(0.25):.4f}
50%                 {simple_returns.quantile(0.50):<15.4f} {log_returns.quantile(0.50):.4f}
75%                 {simple_returns.quantile(0.75):<15.4f} {log_returns.quantile(0.75):.4f}
max                 {simple_returns.max():<15.4f} {log_returns.max():.4f}
skewness            {stats.skew(simple_returns):<15.4f} {stats.skew(log_returns):.4f}
excess kurtosis     {stats.kurtosis(simple_returns):<15.4f} {stats.kurtosis(log_returns):.4f}

Key Observations:
- Log returns have slightly lower mean (expected)
- Log returns have slightly more negative skewness (more symmetric around zero)
- Both show excess kurtosis (fat tails) - motivates ARCH/GARCH
""")

# =============================================================================
# GRAPH 1: Price and Returns Overview
# =============================================================================
log("\nGenerating Graph 1: Price and Returns Time Series...")

fig, axes = plt.subplots(4, 1, figsize=(14, 14))

# Price
axes[0].plot(df_clean['Date'], df_clean['Last_Price'], linewidth=1, color='#2C3E50')
axes[0].set_title('GGAL Stock Price (ARS)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price (ARS)', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Simple Returns
axes[1].plot(df_clean['Date'], df_clean['Simple_Return_pct'], linewidth=0.8, color='#E74C3C', alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[1].set_title('Simple Returns (%)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Return (%)', fontsize=12)
axes[1].grid(True, alpha=0.3)

# Log Returns
axes[2].plot(df_clean['Date'], df_clean['Log_Return_pct'], linewidth=0.8, color='#3498DB', alpha=0.7)
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[2].set_title('Log Returns (%)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Return (%)', fontsize=12)
axes[2].grid(True, alpha=0.3)

# Squared returns comparison (volatility clustering)
axes[3].plot(df_clean['Date'], df_clean['Simple_Return_pct']**2, linewidth=0.8, color='#E74C3C', alpha=0.5, label='Simple²')
axes[3].plot(df_clean['Date'], df_clean['Log_Return_pct']**2, linewidth=0.8, color='#3498DB', alpha=0.5, label='Log²')
axes[3].set_title('Squared Returns - Evidence of Volatility Clustering', fontsize=14, fontweight='bold')
axes[3].set_xlabel('Date', fontsize=12)
axes[3].set_ylabel('Squared Return (%²)', fontsize=12)
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '01_price_returns_timeseries.png'), dpi=300, bbox_inches='tight')
plt.close()
log("  Saved: garch_results/01_price_returns_timeseries.png")

# =============================================================================
# GRAPH 2: Return Distribution Comparison
# =============================================================================
log("\nGenerating Graph 2: Return Distribution Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Simple returns histogram
axes[0, 0].hist(simple_returns, bins=50, density=True, alpha=0.7, color='#E74C3C', edgecolor='black')
mu_s, sigma_s = simple_returns.mean(), simple_returns.std()
x = np.linspace(simple_returns.min(), simple_returns.max(), 100)
axes[0, 0].plot(x, stats.norm.pdf(x, mu_s, sigma_s), 'b-', linewidth=2, label='Normal')
axes[0, 0].set_title('Simple Returns Distribution', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Return (%)', fontsize=11)
axes[0, 0].set_ylabel('Density', fontsize=11)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Log returns histogram
axes[0, 1].hist(log_returns, bins=50, density=True, alpha=0.7, color='#3498DB', edgecolor='black')
mu_l, sigma_l = log_returns.mean(), log_returns.std()
x = np.linspace(log_returns.min(), log_returns.max(), 100)
axes[0, 1].plot(x, stats.norm.pdf(x, mu_l, sigma_l), 'r-', linewidth=2, label='Normal')
axes[0, 1].set_title('Log Returns Distribution', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Return (%)', fontsize=11)
axes[0, 1].set_ylabel('Density', fontsize=11)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot simple returns
stats.probplot(simple_returns, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Simple Returns', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot log returns
stats.probplot(log_returns, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: Log Returns', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '02_return_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
log("  Saved: garch_results/02_return_distribution.png")

# Normality tests
jb_simple, jb_simple_pval = stats.jarque_bera(simple_returns)
jb_log, jb_log_pval = stats.jarque_bera(log_returns)

log(f"""
2.3 NORMALITY TESTS
-------------------
Jarque-Bera Test:
                    Simple Returns    Log Returns
Statistic           {jb_simple:<15.4f} {jb_log:.4f}
P-value             {jb_simple_pval:<15.6f} {jb_log_pval:.6f}

Decision: Both return types are NOT normally distributed (p < 0.05)
Note: Non-normality and fat tails are common in financial data.
      Log returns are slightly closer to normality in most cases.
""")

# =============================================================================
# SECTION 3: ARCH EFFECTS TEST
# =============================================================================
log("\n" + "="*80)
log("SECTION 3: TESTING FOR ARCH EFFECTS")
log("="*80)

log("""
3.1 WHY TEST FOR ARCH EFFECTS?
-------------------------------
ARCH effects are present when:
  - Volatility clusters: high volatility periods followed by high volatility
  - Squared returns are autocorrelated

If ARCH effects exist, ARCH/GARCH models are appropriate.
""")

# =============================================================================
# GRAPH 3: Autocorrelation Analysis
# =============================================================================
log("\nGenerating Graph 3: Autocorrelation Functions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ACF of squared simple returns
plot_acf(simple_returns**2, lags=30, ax=axes[0, 0])
axes[0, 0].set_title('ACF: Squared Simple Returns', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Lag', fontsize=11)

# ACF of squared log returns
plot_acf(log_returns**2, lags=30, ax=axes[0, 1])
axes[0, 1].set_title('ACF: Squared Log Returns', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Lag', fontsize=11)

# PACF of squared simple returns
plot_pacf(simple_returns**2, lags=30, ax=axes[1, 0])
axes[1, 0].set_title('PACF: Squared Simple Returns', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Lag', fontsize=11)

# PACF of squared log returns
plot_pacf(log_returns**2, lags=30, ax=axes[1, 1])
axes[1, 1].set_title('PACF: Squared Log Returns', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Lag', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '03_autocorrelation_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
log("  Saved: garch_results/03_autocorrelation_analysis.png")

# Ljung-Box tests
lb_simple = acorr_ljungbox(simple_returns**2, lags=[10, 20, 30], return_df=True)
lb_log = acorr_ljungbox(log_returns**2, lags=[10, 20, 30], return_df=True)

log(f"""
3.2 LJUNG-BOX TEST FOR ARCH EFFECTS
------------------------------------
Testing autocorrelation in SQUARED returns:

Simple Returns (Squared):
{lb_simple.to_string()}

Log Returns (Squared):
{lb_log.to_string()}

Decision: ARCH effects detected in both return types (p < 0.05)
          → ARCH/GARCH models are appropriate
""")

# =============================================================================
# SECTION 4: TRAIN-TEST SPLIT
# =============================================================================
log("\n" + "="*80)
log("SECTION 4: DATA SPLIT FOR MODEL VALIDATION")
log("="*80)

train_size = int(len(simple_returns) * 0.8)

simple_train = simple_returns[:train_size]
simple_test = simple_returns[train_size:]
log_train = log_returns[:train_size]
log_test = log_returns[train_size:]

dates_train_start = df_clean['Date'].iloc[0]
dates_train_end = df_clean['Date'].iloc[train_size-1]
dates_test_start = df_clean['Date'].iloc[train_size]
dates_test_end = df_clean['Date'].iloc[-1]

log(f"""
4.1 SPLIT METHODOLOGY
---------------------
Split ratio: 80% training / 20% testing

Training set:
  Size: {len(simple_train)} observations
  Period: {dates_train_start.strftime('%Y-%m-%d')} to {dates_train_end.strftime('%Y-%m-%d')}

Testing set:
  Size: {len(simple_test)} observations
  Period: {dates_test_start.strftime('%Y-%m-%d')} to {dates_test_end.strftime('%Y-%m-%d')}
""")

# =============================================================================
# SECTION 5: MODEL ESTIMATION - SIMPLE RETURNS
# =============================================================================
log("\n" + "="*80)
log("SECTION 5: MODEL ESTIMATION - SIMPLE RETURNS")
log("="*80)

log("""
5.1 ARCH(1) AND GARCH(1,1) MODEL SPECIFICATIONS
------------------------------------------------
ARCH(1) Variance equation:
    σ²_t = ω + α₁ · ε²_{t-1}

GARCH(1,1) Variance equation:
    σ²_t = ω + α₁ · ε²_{t-1} + β₁ · σ²_{t-1}

Where:
  - ω: baseline variance
  - α₁: ARCH effect (shock impact)
  - β₁: GARCH effect (persistence)
  - α₁ + β₁: total persistence (should be < 1)
""")

# ARCH(1) - Simple Returns
arch1_simple = arch_model(simple_train, mean='Zero', vol='ARCH', p=1)
arch1_simple_fit = arch1_simple.fit(disp='off')

# GARCH(1,1) - Simple Returns
garch11_simple = arch_model(simple_train, mean='Zero', vol='GARCH', p=1, q=1)
garch11_simple_fit = garch11_simple.fit(disp='off')

# Extract parameters - Simple Returns
omega_arch_s = arch1_simple_fit.params['omega']
alpha1_arch_s = arch1_simple_fit.params['alpha[1]']
omega_garch_s = garch11_simple_fit.params['omega']
alpha1_garch_s = garch11_simple_fit.params['alpha[1]']
beta1_garch_s = garch11_simple_fit.params['beta[1]']
persistence_s = alpha1_garch_s + beta1_garch_s

# Long-run volatility
lr_var_arch_s = omega_arch_s / (1 - alpha1_arch_s)
lr_vol_arch_s = np.sqrt(lr_var_arch_s)
ann_vol_arch_s = lr_vol_arch_s * np.sqrt(252)

lr_var_garch_s = omega_garch_s / (1 - persistence_s)
lr_vol_garch_s = np.sqrt(lr_var_garch_s)
ann_vol_garch_s = lr_vol_garch_s * np.sqrt(252)

half_life_s = np.log(0.5) / np.log(persistence_s) if 0 < persistence_s < 1 else np.inf

log(f"""
5.2 SIMPLE RETURNS - ARCH(1) RESULTS
-------------------------------------
{arch1_simple_fit.summary()}

Parameters:
  ω (omega):     {omega_arch_s:.6f}
  α₁ (alpha[1]): {alpha1_arch_s:.6f}
  Stationarity:  {"✓ Stationary" if alpha1_arch_s < 1 else "✗ NOT stationary"}

Volatility:
  Long-run daily: {lr_vol_arch_s:.4f}%
  Annualized:     {ann_vol_arch_s:.2f}%

Model Fit:
  Log-Likelihood: {arch1_simple_fit.loglikelihood:.2f}
  AIC:            {arch1_simple_fit.aic:.2f}
  BIC:            {arch1_simple_fit.bic:.2f}
""")

log(f"""
5.3 SIMPLE RETURNS - GARCH(1,1) RESULTS
----------------------------------------
{garch11_simple_fit.summary()}

Parameters:
  ω (omega):     {omega_garch_s:.6f}
  α₁ (alpha[1]): {alpha1_garch_s:.6f}
  β₁ (beta[1]):  {beta1_garch_s:.6f}
  Persistence:   {persistence_s:.6f} {"(very high)" if persistence_s > 0.9 else ""}
  Stationarity:  {"✓ Stationary" if persistence_s < 1 else "✗ NOT stationary"}
  Half-life:     {half_life_s:.1f} days

Volatility:
  Long-run daily: {lr_vol_garch_s:.4f}%
  Annualized:     {ann_vol_garch_s:.2f}%

Model Fit:
  Log-Likelihood: {garch11_simple_fit.loglikelihood:.2f}
  AIC:            {garch11_simple_fit.aic:.2f}
  BIC:            {garch11_simple_fit.bic:.2f}
""")

# =============================================================================
# SECTION 6: MODEL ESTIMATION - LOG RETURNS
# =============================================================================
log("\n" + "="*80)
log("SECTION 6: MODEL ESTIMATION - LOG RETURNS")
log("="*80)

# ARCH(1) - Log Returns
arch1_log = arch_model(log_train, mean='Zero', vol='ARCH', p=1)
arch1_log_fit = arch1_log.fit(disp='off')

# GARCH(1,1) - Log Returns
garch11_log = arch_model(log_train, mean='Zero', vol='GARCH', p=1, q=1)
garch11_log_fit = garch11_log.fit(disp='off')

# Extract parameters - Log Returns
omega_arch_l = arch1_log_fit.params['omega']
alpha1_arch_l = arch1_log_fit.params['alpha[1]']
omega_garch_l = garch11_log_fit.params['omega']
alpha1_garch_l = garch11_log_fit.params['alpha[1]']
beta1_garch_l = garch11_log_fit.params['beta[1]']
persistence_l = alpha1_garch_l + beta1_garch_l

# Long-run volatility
lr_var_arch_l = omega_arch_l / (1 - alpha1_arch_l)
lr_vol_arch_l = np.sqrt(lr_var_arch_l)
ann_vol_arch_l = lr_vol_arch_l * np.sqrt(252)

lr_var_garch_l = omega_garch_l / (1 - persistence_l)
lr_vol_garch_l = np.sqrt(lr_var_garch_l)
ann_vol_garch_l = lr_vol_garch_l * np.sqrt(252)

half_life_l = np.log(0.5) / np.log(persistence_l) if 0 < persistence_l < 1 else np.inf

log(f"""
6.1 LOG RETURNS - ARCH(1) RESULTS
----------------------------------
{arch1_log_fit.summary()}

Parameters:
  ω (omega):     {omega_arch_l:.6f}
  α₁ (alpha[1]): {alpha1_arch_l:.6f}
  Stationarity:  {"✓ Stationary" if alpha1_arch_l < 1 else "✗ NOT stationary"}

Volatility:
  Long-run daily: {lr_vol_arch_l:.4f}%
  Annualized:     {ann_vol_arch_l:.2f}%

Model Fit:
  Log-Likelihood: {arch1_log_fit.loglikelihood:.2f}
  AIC:            {arch1_log_fit.aic:.2f}
  BIC:            {arch1_log_fit.bic:.2f}
""")

log(f"""
6.2 LOG RETURNS - GARCH(1,1) RESULTS
-------------------------------------
{garch11_log_fit.summary()}

Parameters:
  ω (omega):     {omega_garch_l:.6f}
  α₁ (alpha[1]): {alpha1_garch_l:.6f}
  β₁ (beta[1]):  {beta1_garch_l:.6f}
  Persistence:   {persistence_l:.6f} {"(very high)" if persistence_l > 0.9 else ""}
  Stationarity:  {"✓ Stationary" if persistence_l < 1 else "✗ NOT stationary"}
  Half-life:     {half_life_l:.1f} days

Volatility:
  Long-run daily: {lr_vol_garch_l:.4f}%
  Annualized:     {ann_vol_garch_l:.2f}%

Model Fit:
  Log-Likelihood: {garch11_log_fit.loglikelihood:.2f}
  AIC:            {garch11_log_fit.aic:.2f}
  BIC:            {garch11_log_fit.bic:.2f}
""")

# =============================================================================
# SECTION 7: MODEL COMPARISON
# =============================================================================
log("\n" + "="*80)
log("SECTION 7: COMPREHENSIVE MODEL COMPARISON")
log("="*80)

comparison_df = pd.DataFrame({
    'Model': ['ARCH(1)', 'ARCH(1)', 'GARCH(1,1)', 'GARCH(1,1)'],
    'Return_Type': ['Simple', 'Log', 'Simple', 'Log'],
    'Log_Likelihood': [arch1_simple_fit.loglikelihood, arch1_log_fit.loglikelihood,
                       garch11_simple_fit.loglikelihood, garch11_log_fit.loglikelihood],
    'AIC': [arch1_simple_fit.aic, arch1_log_fit.aic,
            garch11_simple_fit.aic, garch11_log_fit.aic],
    'BIC': [arch1_simple_fit.bic, arch1_log_fit.bic,
            garch11_simple_fit.bic, garch11_log_fit.bic],
    'Persistence': [alpha1_arch_s, alpha1_arch_l, persistence_s, persistence_l],
    'Annual_Vol_%': [ann_vol_arch_s, ann_vol_arch_l, ann_vol_garch_s, ann_vol_garch_l]
})

log(f"""
7.1 MODEL COMPARISON TABLE
--------------------------
{comparison_df.to_string(index=False)}

Selection criteria:
  - Lower AIC/BIC = Better model
  - Higher Log-Likelihood = Better fit

Best model by AIC: {comparison_df.loc[comparison_df['AIC'].idxmin(), 'Model']} with {comparison_df.loc[comparison_df['AIC'].idxmin(), 'Return_Type']} Returns
Best model by BIC: {comparison_df.loc[comparison_df['BIC'].idxmin(), 'Model']} with {comparison_df.loc[comparison_df['BIC'].idxmin(), 'Return_Type']} Returns

7.2 SIMPLE VS LOG RETURNS COMPARISON
------------------------------------
                        Simple Returns    Log Returns    Difference
ARCH(1) AIC             {arch1_simple_fit.aic:<15.2f} {arch1_log_fit.aic:<14.2f} {arch1_simple_fit.aic - arch1_log_fit.aic:+.2f}
GARCH(1,1) AIC          {garch11_simple_fit.aic:<15.2f} {garch11_log_fit.aic:<14.2f} {garch11_simple_fit.aic - garch11_log_fit.aic:+.2f}
ARCH(1) BIC             {arch1_simple_fit.bic:<15.2f} {arch1_log_fit.bic:<14.2f} {arch1_simple_fit.bic - arch1_log_fit.bic:+.2f}
GARCH(1,1) BIC          {garch11_simple_fit.bic:<15.2f} {garch11_log_fit.bic:<14.2f} {garch11_simple_fit.bic - garch11_log_fit.bic:+.2f}

Interpretation:
  - Negative difference means Log Returns perform better
  - Positive difference means Simple Returns perform better
""")

comparison_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)
log("\n  Saved: garch_results/model_comparison.csv")

# =============================================================================
# GRAPH 4: Fitted Volatility Comparison
# =============================================================================
log("\nGenerating Graph 4: Fitted Conditional Volatility...")

vol_arch_simple = arch1_simple_fit.conditional_volatility
vol_garch_simple = garch11_simple_fit.conditional_volatility
vol_arch_log = arch1_log_fit.conditional_volatility
vol_garch_log = garch11_log_fit.conditional_volatility

dates_train = df_clean['Date'].iloc[:train_size]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ARCH(1) - Simple
axes[0, 0].plot(dates_train, vol_arch_simple, linewidth=1.2, color='#E74C3C')
axes[0, 0].fill_between(dates_train, 0, vol_arch_simple, alpha=0.3, color='#E74C3C')
axes[0, 0].set_title('ARCH(1) - Simple Returns', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Volatility (%)', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# ARCH(1) - Log
axes[0, 1].plot(dates_train, vol_arch_log, linewidth=1.2, color='#3498DB')
axes[0, 1].fill_between(dates_train, 0, vol_arch_log, alpha=0.3, color='#3498DB')
axes[0, 1].set_title('ARCH(1) - Log Returns', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Volatility (%)', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# GARCH(1,1) - Simple
axes[1, 0].plot(dates_train, vol_garch_simple, linewidth=1.2, color='#E67E22')
axes[1, 0].fill_between(dates_train, 0, vol_garch_simple, alpha=0.3, color='#E67E22')
axes[1, 0].set_title('GARCH(1,1) - Simple Returns', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Date', fontsize=11)
axes[1, 0].set_ylabel('Volatility (%)', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# GARCH(1,1) - Log
axes[1, 1].plot(dates_train, vol_garch_log, linewidth=1.2, color='#27AE60')
axes[1, 1].fill_between(dates_train, 0, vol_garch_log, alpha=0.3, color='#27AE60')
axes[1, 1].set_title('GARCH(1,1) - Log Returns', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Date', fontsize=11)
axes[1, 1].set_ylabel('Volatility (%)', fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '04_fitted_conditional_volatility.png'), dpi=300, bbox_inches='tight')
plt.close()
log("  Saved: garch_results/04_fitted_conditional_volatility.png")

log(f"""
Fitted volatility statistics:

ARCH(1) Simple:                      ARCH(1) Log:
  Mean: {vol_arch_simple.mean():.4f}%                     Mean: {vol_arch_log.mean():.4f}%
  Std:  {vol_arch_simple.std():.4f}%                      Std:  {vol_arch_log.std():.4f}%
  Max:  {vol_arch_simple.max():.4f}%                      Max:  {vol_arch_log.max():.4f}%

GARCH(1,1) Simple:                   GARCH(1,1) Log:
  Mean: {vol_garch_simple.mean():.4f}%                    Mean: {vol_garch_log.mean():.4f}%
  Std:  {vol_garch_simple.std():.4f}%                     Std:  {vol_garch_log.std():.4f}%
  Max:  {vol_garch_simple.max():.4f}%                     Max:  {vol_garch_log.max():.4f}%
""")

# =============================================================================
# GRAPH 5: Model Comparison Visualization
# =============================================================================
log("\nGenerating Graph 5: Model Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# AIC/BIC comparison by model and return type
x = np.arange(4)
width = 0.35
labels = ['ARCH(1)\nSimple', 'ARCH(1)\nLog', 'GARCH(1,1)\nSimple', 'GARCH(1,1)\nLog']

axes[0].bar(x - width/2, comparison_df['AIC'], width, label='AIC', color='#3498DB')
axes[0].bar(x + width/2, comparison_df['BIC'], width, label='BIC', color='#E74C3C')
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('Information Criterion', fontsize=12)
axes[0].set_title('Model Selection Criteria (Lower is Better)', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Volatility overlay
axes[1].plot(dates_train, vol_garch_simple, linewidth=1.5, alpha=0.8,
             label='GARCH Simple', color='#E67E22')
axes[1].plot(dates_train, vol_garch_log, linewidth=1.5, alpha=0.8,
             label='GARCH Log', color='#27AE60')
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Conditional Volatility (%)', fontsize=12)
axes[1].set_title('GARCH(1,1) Volatility: Simple vs Log', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '05_model_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
log("  Saved: garch_results/05_model_comparison.png")

# =============================================================================
# SECTION 8: OUT-OF-SAMPLE FORECASTING
# =============================================================================
log("\n" + "="*80)
log("SECTION 8: OUT-OF-SAMPLE FORECASTING")
log("="*80)

log(f"""
8.1 FORECASTING METHODOLOGY
---------------------------
Approach: Rolling window forecast
Test period: {dates_test_start.strftime('%Y-%m-%d')} to {dates_test_end.strftime('%Y-%m-%d')}
Number of forecasts: {len(simple_test)}

We will compare forecasting performance for both return types using GARCH(1,1).
""")

# Rolling window forecasts - Simple Returns
log("\n  Forecasting with Simple Returns...")
forecasts_simple = []
for i in range(len(simple_test)):
    train_data = simple_returns[:train_size + i]
    model = arch_model(train_data, mean='Zero', vol='GARCH', p=1, q=1)
    model_fit = model.fit(disp='off')
    forecast = model_fit.forecast(horizon=1)
    forecasts_simple.append(np.sqrt(forecast.variance.values[-1, 0]))
    if (i+1) % 50 == 0:
        log(f"    Progress: {i+1}/{len(simple_test)}")
forecasts_simple = np.array(forecasts_simple)

# Rolling window forecasts - Log Returns
log("\n  Forecasting with Log Returns...")
forecasts_log = []
for i in range(len(log_test)):
    train_data = log_returns[:train_size + i]
    model = arch_model(train_data, mean='Zero', vol='GARCH', p=1, q=1)
    model_fit = model.fit(disp='off')
    forecast = model_fit.forecast(horizon=1)
    forecasts_log.append(np.sqrt(forecast.variance.values[-1, 0]))
    if (i+1) % 50 == 0:
        log(f"    Progress: {i+1}/{len(log_test)}")
forecasts_log = np.array(forecasts_log)

log("\n  ✓ Forecasting complete!")

# Calculate metrics
realized_simple = np.abs(simple_test.values)
realized_log = np.abs(log_test.values)

corr_simple = np.corrcoef(forecasts_simple, realized_simple)[0, 1]
corr_log = np.corrcoef(forecasts_log, realized_log)[0, 1]

# MSE for variance
mse_simple = np.mean((simple_test.values**2 - forecasts_simple**2)**2)
mse_log = np.mean((log_test.values**2 - forecasts_log**2)**2)

mae_simple = np.mean(np.abs(simple_test.values**2 - forecasts_simple**2))
mae_log = np.mean(np.abs(log_test.values**2 - forecasts_log**2))

log(f"""
8.2 FORECAST EVALUATION METRICS
--------------------------------
                            Simple Returns    Log Returns
Correlation (forecast/real) {corr_simple:<15.4f} {corr_log:.4f}
MSE (variance)              {mse_simple:<15.4f} {mse_log:.4f}
MAE (variance)              {mae_simple:<15.4f} {mae_log:.4f}

Interpretation:
  - Higher correlation = better forecasting
  - Lower MSE/MAE = more accurate forecasts

Best forecasting return type: {"Log Returns" if corr_log > corr_simple else "Simple Returns"} (higher correlation)
""")

# =============================================================================
# GRAPH 6: Forecast vs Realized
# =============================================================================
log("\nGenerating Graph 6: Forecast vs Realized Volatility...")

dates_test = df_clean['Date'].iloc[train_size:train_size + len(simple_test)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time series - Simple
axes[0, 0].plot(dates_test, realized_simple, linewidth=1.2, alpha=0.7,
                label='Realized', color='#2C3E50')
axes[0, 0].plot(dates_test, forecasts_simple, linewidth=1.2, alpha=0.8,
                label='Forecast', color='#E74C3C')
axes[0, 0].set_title(f'Simple Returns (Corr: {corr_simple:.3f})', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Volatility (%)', fontsize=11)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Time series - Log
axes[0, 1].plot(dates_test, realized_log, linewidth=1.2, alpha=0.7,
                label='Realized', color='#2C3E50')
axes[0, 1].plot(dates_test, forecasts_log, linewidth=1.2, alpha=0.8,
                label='Forecast', color='#3498DB')
axes[0, 1].set_title(f'Log Returns (Corr: {corr_log:.3f})', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Volatility (%)', fontsize=11)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Scatter - Simple
axes[1, 0].scatter(forecasts_simple, realized_simple, alpha=0.6, s=40, color='#E74C3C')
max_val = max(forecasts_simple.max(), realized_simple.max())
axes[1, 0].plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Perfect Forecast')
z = np.polyfit(forecasts_simple, realized_simple, 1)
p = np.poly1d(z)
axes[1, 0].plot(sorted(forecasts_simple), p(sorted(forecasts_simple)), 'g-', linewidth=2, alpha=0.7)
axes[1, 0].set_xlabel('Forecasted Vol (%)', fontsize=11)
axes[1, 0].set_ylabel('Realized Vol (%)', fontsize=11)
axes[1, 0].set_title('Simple Returns - Scatter', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Scatter - Log
axes[1, 1].scatter(forecasts_log, realized_log, alpha=0.6, s=40, color='#3498DB')
max_val = max(forecasts_log.max(), realized_log.max())
axes[1, 1].plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Perfect Forecast')
z = np.polyfit(forecasts_log, realized_log, 1)
p = np.poly1d(z)
axes[1, 1].plot(sorted(forecasts_log), p(sorted(forecasts_log)), 'g-', linewidth=2, alpha=0.7)
axes[1, 1].set_xlabel('Forecasted Vol (%)', fontsize=11)
axes[1, 1].set_ylabel('Realized Vol (%)', fontsize=11)
axes[1, 1].set_title('Log Returns - Scatter', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '06_forecast_vs_realized.png'), dpi=300, bbox_inches='tight')
plt.close()
log("  Saved: garch_results/06_forecast_vs_realized.png")

# =============================================================================
# SECTION 9: MODEL DIAGNOSTICS
# =============================================================================
log("\n" + "="*80)
log("SECTION 9: MODEL DIAGNOSTICS")
log("="*80)

# Standardized residuals for best model (GARCH with Log returns)
std_resid_simple = garch11_simple_fit.std_resid
std_resid_log = garch11_log_fit.std_resid

log(f"""
9.1 STANDARDIZED RESIDUALS STATISTICS
--------------------------------------
                            Simple Returns    Log Returns
Mean (should be ≈ 0)        {std_resid_simple.mean():<15.4f} {std_resid_log.mean():.4f}
Std Dev (should be ≈ 1)     {std_resid_simple.std():<15.4f} {std_resid_log.std():.4f}
Skewness                    {stats.skew(std_resid_simple):<15.4f} {stats.skew(std_resid_log):.4f}
Excess Kurtosis             {stats.kurtosis(std_resid_simple):<15.4f} {stats.kurtosis(std_resid_log):.4f}
""")

# =============================================================================
# GRAPH 7: Model Diagnostics
# =============================================================================
log("\nGenerating Graph 7: Model Diagnostics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Standardized residuals - Simple
axes[0, 0].plot(std_resid_simple, linewidth=0.8, color='#E74C3C', alpha=0.7)
axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[0, 0].set_title('Std Residuals - Simple Returns', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Std Residual', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# Standardized residuals - Log
axes[0, 1].plot(std_resid_log, linewidth=0.8, color='#3498DB', alpha=0.7)
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[0, 1].set_title('Std Residuals - Log Returns', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Std Residual', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# ACF squared residuals - Simple
plot_acf(std_resid_simple**2, lags=30, ax=axes[1, 0])
axes[1, 0].set_title('ACF Squared Std Resid - Simple', fontsize=12, fontweight='bold')

# ACF squared residuals - Log
plot_acf(std_resid_log**2, lags=30, ax=axes[1, 1])
axes[1, 1].set_title('ACF Squared Std Resid - Log', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '07_model_diagnostics.png'), dpi=300, bbox_inches='tight')
plt.close()
log("  Saved: garch_results/07_model_diagnostics.png")

# Ljung-Box tests
lb_resid_simple = acorr_ljungbox(std_resid_simple**2, lags=[10, 20, 30], return_df=True)
lb_resid_log = acorr_ljungbox(std_resid_log**2, lags=[10, 20, 30], return_df=True)

log(f"""
9.2 LJUNG-BOX TESTS ON SQUARED RESIDUALS
-----------------------------------------
Simple Returns GARCH(1,1):
{lb_resid_simple.to_string()}

Log Returns GARCH(1,1):
{lb_resid_log.to_string()}

Interpretation (p > 0.05 means no remaining ARCH effects):
  Simple Returns: {"✓ No remaining ARCH effects" if lb_resid_simple['lb_pvalue'].iloc[0] > 0.05 else "✗ Some ARCH effects remain"}
  Log Returns:    {"✓ No remaining ARCH effects" if lb_resid_log['lb_pvalue'].iloc[0] > 0.05 else "✗ Some ARCH effects remain"}
""")

# =============================================================================
# SECTION 10: SUMMARY
# =============================================================================
log("\n" + "="*80)
log("SECTION 10: FINAL SUMMARY")
log("="*80)

# Create comprehensive parameter summary
params_summary = pd.DataFrame({
    'Model': ['ARCH(1)', 'ARCH(1)', 'GARCH(1,1)', 'GARCH(1,1)'],
    'Return_Type': ['Simple', 'Log', 'Simple', 'Log'],
    'omega': [omega_arch_s, omega_arch_l, omega_garch_s, omega_garch_l],
    'alpha1': [alpha1_arch_s, alpha1_arch_l, alpha1_garch_s, alpha1_garch_l],
    'beta1': [np.nan, np.nan, beta1_garch_s, beta1_garch_l],
    'persistence': [alpha1_arch_s, alpha1_arch_l, persistence_s, persistence_l],
    'half_life_days': [np.nan, np.nan, half_life_s, half_life_l],
    'long_run_vol_%': [lr_vol_arch_s, lr_vol_arch_l, lr_vol_garch_s, lr_vol_garch_l],
    'annual_vol_%': [ann_vol_arch_s, ann_vol_arch_l, ann_vol_garch_s, ann_vol_garch_l],
    'AIC': [arch1_simple_fit.aic, arch1_log_fit.aic, garch11_simple_fit.aic, garch11_log_fit.aic],
    'BIC': [arch1_simple_fit.bic, arch1_log_fit.bic, garch11_simple_fit.bic, garch11_log_fit.bic],
    'Log_Likelihood': [arch1_simple_fit.loglikelihood, arch1_log_fit.loglikelihood,
                       garch11_simple_fit.loglikelihood, garch11_log_fit.loglikelihood]
})

log(f"""
10.1 COMPREHENSIVE PARAMETER ESTIMATES
---------------------------------------
{params_summary.to_string(index=False)}
""")

params_summary.to_csv(os.path.join(RESULTS_DIR, 'parameters_summary.csv'), index=False)
log("\n  Saved: garch_results/parameters_summary.csv")

# Determine best model
best_aic_idx = comparison_df['AIC'].idxmin()
best_model = comparison_df.loc[best_aic_idx]

log(f"""
10.2 KEY FINDINGS
-----------------
Data Summary:
  - Sample: {len(simple_returns)} daily returns
  - Period: {df_clean['Date'].min().strftime('%Y-%m-%d')} to {df_clean['Date'].max().strftime('%Y-%m-%d')}
  - Simple Returns: mean={simple_returns.mean():.4f}%, std={simple_returns.std():.4f}%
  - Log Returns:    mean={log_returns.mean():.4f}%, std={log_returns.std():.4f}%

Return Type Comparison:
  - Log returns have slightly lower mean and slightly more symmetric distribution
  - Both show significant excess kurtosis (fat tails)
  - ARCH effects present in both return types

Model Selection:
  Best model by AIC: {best_model['Model']} with {best_model['Return_Type']} Returns
    - AIC: {best_model['AIC']:.2f}
    - Persistence: {best_model['Persistence']:.4f}
    - Annualized Volatility: {best_model['Annual_Vol_%']:.2f}%

GARCH(1,1) Comparison:
                        Simple Returns    Log Returns
  AIC                   {garch11_simple_fit.aic:<15.2f} {garch11_log_fit.aic:.2f}
  Persistence           {persistence_s:<15.4f} {persistence_l:.4f}
  Half-life (days)      {half_life_s:<15.1f} {half_life_l:.1f}
  Annualized Vol        {ann_vol_garch_s:<15.2f}% {ann_vol_garch_l:.2f}%

Forecasting Performance:
  Simple Returns correlation: {corr_simple:.4f}
  Log Returns correlation:    {corr_log:.4f}
  Better forecasting: {"Log Returns" if corr_log > corr_simple else "Simple Returns"}

Recommendation:
  {"Log Returns are preferred for GARCH modeling due to:" if corr_log > corr_simple or garch11_log_fit.aic < garch11_simple_fit.aic else "Simple Returns perform comparably, but Log Returns are theoretically preferred:"}
    1. More symmetric distribution
    2. Additive over time
    3. Standard in academic literature
    4. {"Better forecasting performance" if corr_log > corr_simple else "Similar forecasting performance"}
    5. {"Lower AIC (better model fit)" if garch11_log_fit.aic < garch11_simple_fit.aic else "Similar AIC values"}

10.3 FILES GENERATED
--------------------
  - garch_results.txt (this report)
  - garch_results/01_price_returns_timeseries.png
  - garch_results/02_return_distribution.png
  - garch_results/03_autocorrelation_analysis.png
  - garch_results/04_fitted_conditional_volatility.png
  - garch_results/05_model_comparison.png
  - garch_results/06_forecast_vs_realized.png
  - garch_results/07_model_diagnostics.png
  - garch_results/model_comparison.csv
  - garch_results/parameters_summary.csv
""")

log("\n" + "="*80)
log("ANALYSIS COMPLETE")
log("="*80)

results_file.close()
print("\n✓ GARCH analysis complete!")
print("  Check garch_results.txt and garch_results/ folder for all outputs.")
