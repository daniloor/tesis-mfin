"""
================================================================================
RV VS IV COMPREHENSIVE ANALYSIS
================================================================================
This script analyzes the relationship between Realized Volatility (RV) and
Implied Volatility (IV) for GGAL options, comparing:
1. Full data period (all available data with IV)
2. Backtesting test period (2025-02-21 to 2025-12-18)

Outputs:
- results/ folder with visualizations
- Descriptive statistics printed and saved
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTRADAY_DIR = os.path.join(SCRIPT_DIR, '..', 'intraday')
PROCESS_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'process_data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

# Create output directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

print("="*80)
print("RV VS IV COMPREHENSIVE ANALYSIS")
print("="*80)

# =============================================================================
# SECTION 1: LOAD AND PREPARE DATA
# =============================================================================
print("\n[1] Loading data...")

# Load 10-minute intraday data for RV calculation
df_10min = pd.read_csv(os.path.join(INTRADAY_DIR, 'BCBA_DLY_GGAL, 10 (1).csv'))
df_10min['time'] = pd.to_datetime(df_10min['time'])
df_10min = df_10min.sort_values('time').reset_index(drop=True)

# Load daily data with IV
df_daily = pd.read_csv(os.path.join(PROCESS_DATA_DIR, 'data.dat'))
df_daily['Date'] = pd.to_datetime(df_daily['Date'])
df_daily = df_daily.sort_values('Date').reset_index(drop=True)

print(f"  10-min data: {df_10min['time'].min().strftime('%Y-%m-%d')} to {df_10min['time'].max().strftime('%Y-%m-%d')}")
print(f"  Daily data:  {df_daily['Date'].min().strftime('%Y-%m-%d')} to {df_daily['Date'].max().strftime('%Y-%m-%d')}")

# =============================================================================
# SECTION 2: CALCULATE REALIZED VOLATILITY FROM INTRADAY DATA
# =============================================================================
print("\n[2] Calculating Realized Volatility...")

def calc_rv(df, min_bars=30):
    """Calculate daily realized volatility from intraday data."""
    df = df.copy()
    df['date'] = df['time'].dt.date
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['prev_date'] = df['date'].shift(1)
    df.loc[df['date'] != df['prev_date'], 'log_return'] = np.nan

    daily = df.groupby('date').agg(
        RV=('log_return', lambda x: np.sqrt((x**2).sum())),
        n_bars=('log_return', 'count'),
        close=('close', 'last'),
    ).reset_index()

    daily['date'] = pd.to_datetime(daily['date'])
    daily['RV_pct'] = daily['RV'] * 100  # Convert to percentage
    daily = daily[daily['n_bars'] >= min_bars].reset_index(drop=True)
    return daily

df_rv = calc_rv(df_10min, min_bars=30)
print(f"  RV calculated for {len(df_rv)} trading days")

# =============================================================================
# SECTION 3: MERGE RV WITH IV DATA
# =============================================================================
print("\n[3] Merging RV with IV data...")

# Prepare IV data
df_iv = df_daily[['Date', 'IV_Call_Avg', 'IV_Put_Avg']].copy()
df_iv.columns = ['date', 'IV_Call', 'IV_Put']

# Merge
df = df_rv.merge(df_iv, on='date', how='inner')

# Calculate IV_daily (daily volatility from annualized IV)
# IV is typically annualized, so divide by sqrt(252) to get daily
df['IV_daily'] = ((df['IV_Call'] + df['IV_Put']) / 2) * 100 / np.sqrt(252)

# Calculate VRP (Variance Risk Premium)
df['VRP'] = df['IV_daily'] - df['RV_pct']

# Calculate RV annualized for comparison
df['RV_annual'] = df['RV_pct'] * np.sqrt(252)
df['IV_annual'] = ((df['IV_Call'] + df['IV_Put']) / 2) * 100

print(f"  Merged dataset: {len(df)} trading days with both RV and IV")
print(f"  Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

# =============================================================================
# SECTION 4: DEFINE PERIODS
# =============================================================================
print("\n[4] Defining analysis periods...")

# Test period for backtesting (from GAMMA_SCALPING_STRATEGY.py)
TEST_START = pd.Timestamp('2025-02-21')
TEST_END = pd.Timestamp('2025-12-18')

# Full period
df_full = df.copy()

# Test period only
df_test = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy()

print(f"  Full period: {df_full['date'].min().strftime('%Y-%m-%d')} to {df_full['date'].max().strftime('%Y-%m-%d')} ({len(df_full)} days)")
print(f"  Test period: {df_test['date'].min().strftime('%Y-%m-%d')} to {df_test['date'].max().strftime('%Y-%m-%d')} ({len(df_test)} days)")

# =============================================================================
# SECTION 5: DESCRIPTIVE STATISTICS FUNCTION
# =============================================================================

def calc_descriptive_stats(data, name):
    """Calculate comprehensive descriptive statistics."""
    stats_dict = {
        'Period': name,
        'N_Days': len(data),
        'Start_Date': data['date'].min().strftime('%Y-%m-%d'),
        'End_Date': data['date'].max().strftime('%Y-%m-%d'),
        # RV Statistics (Daily %)
        'RV_Mean': data['RV_pct'].mean(),
        'RV_Std': data['RV_pct'].std(),
        'RV_Min': data['RV_pct'].min(),
        'RV_Max': data['RV_pct'].max(),
        'RV_Median': data['RV_pct'].median(),
        'RV_Skew': data['RV_pct'].skew(),
        'RV_Kurt': data['RV_pct'].kurtosis(),
        # IV Statistics (Daily %)
        'IV_Mean': data['IV_daily'].mean(),
        'IV_Std': data['IV_daily'].std(),
        'IV_Min': data['IV_daily'].min(),
        'IV_Max': data['IV_daily'].max(),
        'IV_Median': data['IV_daily'].median(),
        'IV_Skew': data['IV_daily'].skew(),
        'IV_Kurt': data['IV_daily'].kurtosis(),
        # VRP Statistics
        'VRP_Mean': data['VRP'].mean(),
        'VRP_Std': data['VRP'].std(),
        'VRP_Min': data['VRP'].min(),
        'VRP_Max': data['VRP'].max(),
        'VRP_Median': data['VRP'].median(),
        # Relationship metrics
        'Correlation_RV_IV': data['RV_pct'].corr(data['IV_daily']),
        'IV_Over_RV_Mean': (data['IV_daily'] / data['RV_pct']).replace([np.inf, -np.inf], np.nan).mean(),
        'Days_IV_GT_RV': (data['IV_daily'] > data['RV_pct']).sum(),
        'Days_IV_LT_RV': (data['IV_daily'] < data['RV_pct']).sum(),
        'Pct_IV_GT_RV': (data['IV_daily'] > data['RV_pct']).mean() * 100,
        # Annualized versions
        'RV_Annual_Mean': data['RV_annual'].mean(),
        'IV_Annual_Mean': data['IV_annual'].mean(),
    }
    return stats_dict

# =============================================================================
# SECTION 6: CALCULATE AND DISPLAY STATISTICS
# =============================================================================
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

stats_full = calc_descriptive_stats(df_full, 'Full Period')
stats_test = calc_descriptive_stats(df_test, 'Test Period (Backtest)')

def print_stats_comparison(s1, s2):
    """Print side-by-side comparison of statistics."""
    print("\n" + "-"*80)
    print(f"{'Metric':<35} {'Full Period':>20} {'Test Period':>20}")
    print("-"*80)

    # Period info
    print(f"{'Number of Trading Days':<35} {s1['N_Days']:>20,} {s2['N_Days']:>20,}")
    print(f"{'Start Date':<35} {s1['Start_Date']:>20} {s2['Start_Date']:>20}")
    print(f"{'End Date':<35} {s1['End_Date']:>20} {s2['End_Date']:>20}")

    print("\n--- REALIZED VOLATILITY (Daily %) ---")
    print(f"{'RV Mean':<35} {s1['RV_Mean']:>20.4f} {s2['RV_Mean']:>20.4f}")
    print(f"{'RV Std Dev':<35} {s1['RV_Std']:>20.4f} {s2['RV_Std']:>20.4f}")
    print(f"{'RV Min':<35} {s1['RV_Min']:>20.4f} {s2['RV_Min']:>20.4f}")
    print(f"{'RV Max':<35} {s1['RV_Max']:>20.4f} {s2['RV_Max']:>20.4f}")
    print(f"{'RV Median':<35} {s1['RV_Median']:>20.4f} {s2['RV_Median']:>20.4f}")
    print(f"{'RV Skewness':<35} {s1['RV_Skew']:>20.4f} {s2['RV_Skew']:>20.4f}")
    print(f"{'RV Kurtosis':<35} {s1['RV_Kurt']:>20.4f} {s2['RV_Kurt']:>20.4f}")
    print(f"{'RV Annualized Mean':<35} {s1['RV_Annual_Mean']:>19.2f}% {s2['RV_Annual_Mean']:>19.2f}%")

    print("\n--- IMPLIED VOLATILITY (Daily %) ---")
    print(f"{'IV Mean':<35} {s1['IV_Mean']:>20.4f} {s2['IV_Mean']:>20.4f}")
    print(f"{'IV Std Dev':<35} {s1['IV_Std']:>20.4f} {s2['IV_Std']:>20.4f}")
    print(f"{'IV Min':<35} {s1['IV_Min']:>20.4f} {s2['IV_Min']:>20.4f}")
    print(f"{'IV Max':<35} {s1['IV_Max']:>20.4f} {s2['IV_Max']:>20.4f}")
    print(f"{'IV Median':<35} {s1['IV_Median']:>20.4f} {s2['IV_Median']:>20.4f}")
    print(f"{'IV Skewness':<35} {s1['IV_Skew']:>20.4f} {s2['IV_Skew']:>20.4f}")
    print(f"{'IV Kurtosis':<35} {s1['IV_Kurt']:>20.4f} {s2['IV_Kurt']:>20.4f}")
    print(f"{'IV Annualized Mean':<35} {s1['IV_Annual_Mean']:>19.2f}% {s2['IV_Annual_Mean']:>19.2f}%")

    print("\n--- VARIANCE RISK PREMIUM (IV - RV, Daily %) ---")
    print(f"{'VRP Mean':<35} {s1['VRP_Mean']:>20.4f} {s2['VRP_Mean']:>20.4f}")
    print(f"{'VRP Std Dev':<35} {s1['VRP_Std']:>20.4f} {s2['VRP_Std']:>20.4f}")
    print(f"{'VRP Min':<35} {s1['VRP_Min']:>20.4f} {s2['VRP_Min']:>20.4f}")
    print(f"{'VRP Max':<35} {s1['VRP_Max']:>20.4f} {s2['VRP_Max']:>20.4f}")
    print(f"{'VRP Median':<35} {s1['VRP_Median']:>20.4f} {s2['VRP_Median']:>20.4f}")

    print("\n--- RV vs IV RELATIONSHIP ---")
    print(f"{'Correlation (RV, IV)':<35} {s1['Correlation_RV_IV']:>20.4f} {s2['Correlation_RV_IV']:>20.4f}")
    print(f"{'Avg IV/RV Ratio':<35} {s1['IV_Over_RV_Mean']:>20.4f} {s2['IV_Over_RV_Mean']:>20.4f}")
    print(f"{'Days IV > RV':<35} {s1['Days_IV_GT_RV']:>20} {s2['Days_IV_GT_RV']:>20}")
    print(f"{'Days IV < RV':<35} {s1['Days_IV_LT_RV']:>20} {s2['Days_IV_LT_RV']:>20}")
    print(f"{'% Days IV > RV':<35} {s1['Pct_IV_GT_RV']:>19.1f}% {s2['Pct_IV_GT_RV']:>19.1f}%")
    print("-"*80)

print_stats_comparison(stats_full, stats_test)

# =============================================================================
# SECTION 7: GENERATE VISUALIZATIONS
# =============================================================================
print("\n[7] Generating visualizations...")

# -----------------------------------------------------------------------------
# FIGURE 1: RV vs IV Time Series Comparison (Full vs Test Period)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# Panel 1: Full Period - RV vs IV
ax1 = axes[0]
ax1.plot(df_full['date'], df_full['RV_pct'], linewidth=0.8, label='Volatilidad Realizada (RV)',
         color='#2E86AB', alpha=0.8)
ax1.plot(df_full['date'], df_full['IV_daily'], linewidth=0.8, label='Volatilidad Implicita (IV)',
         color='#F18F01', alpha=0.8)
ax1.axvline(x=TEST_START, color='green', linestyle='--', linewidth=2, label='Inicio Periodo Test')
ax1.axvline(x=TEST_END, color='red', linestyle='--', linewidth=2, label='Fin Periodo Test')
ax1.axhline(y=df_full['RV_pct'].mean(), color='#2E86AB', linestyle=':', alpha=0.5)
ax1.axhline(y=df_full['IV_daily'].mean(), color='#F18F01', linestyle=':', alpha=0.5)
ax1.set_title('PERIODO COMPLETO: Volatilidad Realizada vs Volatilidad Implicita (Diaria %)', fontweight='bold', fontsize=14)
ax1.set_ylabel('Volatilidad (%)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

# Panel 2: Test Period Only - RV vs IV
ax2 = axes[1]
ax2.plot(df_test['date'], df_test['RV_pct'], linewidth=1.2, label='Volatilidad Realizada (RV)',
         color='#2E86AB', alpha=0.9)
ax2.plot(df_test['date'], df_test['IV_daily'], linewidth=1.2, label='Volatilidad Implicita (IV)',
         color='#F18F01', alpha=0.9)
ax2.axhline(y=df_test['RV_pct'].mean(), color='#2E86AB', linestyle='--',
            label=f'RV Mean: {df_test["RV_pct"].mean():.3f}%')
ax2.axhline(y=df_test['IV_daily'].mean(), color='#F18F01', linestyle='--',
            label=f'IV Mean: {df_test["IV_daily"].mean():.3f}%')
ax2.set_title(f'TEST PERIOD ({TEST_START.strftime("%Y-%m-%d")} to {TEST_END.strftime("%Y-%m-%d")}): RV vs IV',
              fontweight='bold', fontsize=14)
ax2.set_ylabel('Volatilidad (%)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

# Panel 3: VRP Comparison
ax3 = axes[2]
ax3.plot(df_full['date'], df_full['VRP'], linewidth=0.8, color='#27ae60', alpha=0.7)
ax3.fill_between(df_full['date'], 0, df_full['VRP'],
                 where=df_full['VRP'] > 0, alpha=0.3, color='green', label='VRP > 0 (IV sobrepriceada)')
ax3.fill_between(df_full['date'], 0, df_full['VRP'],
                 where=df_full['VRP'] < 0, alpha=0.3, color='red', label='VRP < 0 (IV subpriceada)')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.axvline(x=TEST_START, color='green', linestyle='--', linewidth=2)
ax3.axvline(x=TEST_END, color='red', linestyle='--', linewidth=2)
ax3.axhline(y=df_full['VRP'].mean(), color='purple', linestyle='--',
            label=f'Mean VRP: {df_full["VRP"].mean():.3f}%')
ax3.set_title('Prima de Riesgo de Varianza (VRP = IV - RV)', fontweight='bold', fontsize=14)
ax3.set_xlabel('Fecha')
ax3.set_ylabel('VRP (%)')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '01_rv_iv_timeseries.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: results/01_rv_iv_timeseries.png")

# -----------------------------------------------------------------------------
# FIGURE 2: Distribution Comparison (Full vs Test)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# RV Distribution
ax1 = axes[0, 0]
ax1.hist(df_full['RV_pct'], bins=50, alpha=0.6, color='#2E86AB', label='Periodo Completo', density=True)
ax1.hist(df_test['RV_pct'], bins=30, alpha=0.6, color='#e74c3c', label='Periodo Test', density=True)
ax1.axvline(df_full['RV_pct'].mean(), color='#2E86AB', linestyle='--', linewidth=2)
ax1.axvline(df_test['RV_pct'].mean(), color='#e74c3c', linestyle='--', linewidth=2)
ax1.set_title('Comparacion Distribucion RV', fontweight='bold')
ax1.set_xlabel('RV (%)')
ax1.set_ylabel('Densidad')
ax1.legend()

# IV Distribution
ax2 = axes[0, 1]
ax2.hist(df_full['IV_daily'], bins=50, alpha=0.6, color='#F18F01', label='Periodo Completo', density=True)
ax2.hist(df_test['IV_daily'], bins=30, alpha=0.6, color='#9b59b6', label='Periodo Test', density=True)
ax2.axvline(df_full['IV_daily'].mean(), color='#F18F01', linestyle='--', linewidth=2)
ax2.axvline(df_test['IV_daily'].mean(), color='#9b59b6', linestyle='--', linewidth=2)
ax2.set_title('Comparacion Distribucion IV', fontweight='bold')
ax2.set_xlabel('IV (%)')
ax2.set_ylabel('Densidad')
ax2.legend()

# VRP Distribution
ax3 = axes[0, 2]
ax3.hist(df_full['VRP'], bins=50, alpha=0.6, color='#27ae60', label='Periodo Completo', density=True)
ax3.hist(df_test['VRP'], bins=30, alpha=0.6, color='#e67e22', label='Periodo Test', density=True)
ax3.axvline(0, color='black', linestyle='-', linewidth=1)
ax3.axvline(df_full['VRP'].mean(), color='#27ae60', linestyle='--', linewidth=2)
ax3.axvline(df_test['VRP'].mean(), color='#e67e22', linestyle='--', linewidth=2)
ax3.set_title('Comparacion Distribucion VRP', fontweight='bold')
ax3.set_xlabel('VRP (%)')
ax3.set_ylabel('Densidad')
ax3.legend()

# RV vs IV Scatter - Full
ax4 = axes[1, 0]
ax4.scatter(df_full['RV_pct'], df_full['IV_daily'], alpha=0.3, s=20, c='#2E86AB')
ax4.plot([0, df_full['RV_pct'].max()], [0, df_full['RV_pct'].max()], 'r--',
         label='Precio Justo (RV=IV)', linewidth=2)
# Regression line
slope, intercept, r, p, se = stats.linregress(df_full['RV_pct'], df_full['IV_daily'])
x_line = np.linspace(df_full['RV_pct'].min(), df_full['RV_pct'].max(), 100)
ax4.plot(x_line, slope * x_line + intercept, 'g-',
         label=f'Regression (R={r:.3f})', linewidth=2)
ax4.set_title(f'RV vs IV Dispersion - Periodo Completo (R={r:.3f})', fontweight='bold')
ax4.set_xlabel('RV (%)')
ax4.set_ylabel('IV (%)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# RV vs IV Scatter - Test
ax5 = axes[1, 1]
ax5.scatter(df_test['RV_pct'], df_test['IV_daily'], alpha=0.5, s=30, c='#e74c3c')
ax5.plot([0, df_test['RV_pct'].max()], [0, df_test['RV_pct'].max()], 'r--',
         label='Precio Justo (RV=IV)', linewidth=2)
slope_t, intercept_t, r_t, p_t, se_t = stats.linregress(df_test['RV_pct'], df_test['IV_daily'])
x_line_t = np.linspace(df_test['RV_pct'].min(), df_test['RV_pct'].max(), 100)
ax5.plot(x_line_t, slope_t * x_line_t + intercept_t, 'g-',
         label=f'Regression (R={r_t:.3f})', linewidth=2)
ax5.set_title(f'RV vs IV Dispersion - Periodo Test (R={r_t:.3f})', fontweight='bold')
ax5.set_xlabel('RV (%)')
ax5.set_ylabel('IV (%)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Box plots comparison
ax6 = axes[1, 2]
data_box = [df_full['RV_pct'], df_test['RV_pct'], df_full['IV_daily'], df_test['IV_daily'],
            df_full['VRP'], df_test['VRP']]
labels = ['RV\nFull', 'RV\nTest', 'IV\nFull', 'IV\nTest', 'VRP\nFull', 'VRP\nTest']
colors = ['#2E86AB', '#e74c3c', '#F18F01', '#9b59b6', '#27ae60', '#e67e22']
bp = ax6.boxplot(data_box, labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax6.set_title('Comparacion de Distribuciones (Box Plots)', fontweight='bold')
ax6.set_ylabel('Valor (%)')
ax6.axhline(0, color='black', linestyle='--', alpha=0.5)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '02_distribution_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: results/02_distribution_comparison.png")

# -----------------------------------------------------------------------------
# FIGURE 3: Rolling Statistics Comparison
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Calculate rolling statistics (21-day = ~1 month)
window = 21
df_full['RV_roll'] = df_full['RV_pct'].rolling(window).mean()
df_full['IV_roll'] = df_full['IV_daily'].rolling(window).mean()
df_full['VRP_roll'] = df_full['VRP'].rolling(window).mean()
df_full['Corr_roll'] = df_full['RV_pct'].rolling(window).corr(df_full['IV_daily'])

# Panel 1: Rolling RV vs IV
ax1 = axes[0]
ax1.plot(df_full['date'], df_full['RV_roll'], linewidth=1.5, label='RV (21-day avg)', color='#2E86AB')
ax1.plot(df_full['date'], df_full['IV_roll'], linewidth=1.5, label='IV (21-day avg)', color='#F18F01')
ax1.axvline(x=TEST_START, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax1.axvline(x=TEST_END, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.set_title('Promedio Movil 21 Dias: RV vs IV', fontweight='bold', fontsize=14)
ax1.set_ylabel('Volatilidad (%)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Panel 2: Rolling VRP
ax2 = axes[1]
ax2.plot(df_full['date'], df_full['VRP_roll'], linewidth=1.5, color='#27ae60')
ax2.fill_between(df_full['date'], 0, df_full['VRP_roll'],
                 where=df_full['VRP_roll'] > 0, alpha=0.3, color='green')
ax2.fill_between(df_full['date'], 0, df_full['VRP_roll'],
                 where=df_full['VRP_roll'] < 0, alpha=0.3, color='red')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axvline(x=TEST_START, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax2.axvline(x=TEST_END, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.set_title('Promedio Movil 21 Dias: VRP', fontweight='bold', fontsize=14)
ax2.set_ylabel('VRP (%)')
ax2.grid(True, alpha=0.3)

# Panel 3: Rolling Correlation
ax3 = axes[2]
ax3.plot(df_full['date'], df_full['Corr_roll'], linewidth=1.5, color='#9b59b6')
ax3.axhline(y=df_full['Corr_roll'].mean(), color='purple', linestyle='--',
            label=f'Mean Corr: {df_full["Corr_roll"].mean():.3f}')
ax3.axvline(x=TEST_START, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax3.axvline(x=TEST_END, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax3.set_title('Correlacion Movil 21 Dias (RV, IV)', fontweight='bold', fontsize=14)
ax3.set_xlabel('Fecha')
ax3.set_ylabel('Correlacion')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '03_rolling_statistics.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: results/03_rolling_statistics.png")

# -----------------------------------------------------------------------------
# FIGURE 4: Summary Dashboard
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 12))

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main title
fig.suptitle('RV vs IV Analysis Summary: Full Period vs Backtesting Test Period',
             fontsize=16, fontweight='bold', y=0.98)

# Panel 1: Bar chart of means
ax1 = fig.add_subplot(gs[0, 0])
metrics = ['RV', 'IV', 'VRP']
full_means = [stats_full['RV_Mean'], stats_full['IV_Mean'], stats_full['VRP_Mean']]
test_means = [stats_test['RV_Mean'], stats_test['IV_Mean'], stats_test['VRP_Mean']]
x = np.arange(len(metrics))
width = 0.35
bars1 = ax1.bar(x - width/2, full_means, width, label='Periodo Completo', color='#2E86AB', alpha=0.7)
bars2 = ax1.bar(x + width/2, test_means, width, label='Periodo Test', color='#e74c3c', alpha=0.7)
ax1.set_xlabel('Metric')
ax1.set_ylabel('Mean Value (%)')
ax1.set_title('Mean Values Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)

# Panel 2: Correlation comparison
ax2 = fig.add_subplot(gs[0, 1])
corrs = [stats_full['Correlation_RV_IV'], stats_test['Correlation_RV_IV']]
colors = ['#2E86AB', '#e74c3c']
bars = ax2.bar(['Full Period', 'Test Period'], corrs, color=colors, alpha=0.7)
ax2.set_ylabel('Correlacion')
ax2.set_title('RV-IV Correlation Comparison', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 1)
for bar, corr in zip(bars, corrs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{corr:.3f}', ha='center', fontsize=12, fontweight='bold')

# Panel 3: IV/RV Ratio
ax3 = fig.add_subplot(gs[0, 2])
ratios = [stats_full['IV_Over_RV_Mean'], stats_test['IV_Over_RV_Mean']]
bars = ax3.bar(['Full Period', 'Test Period'], ratios, color=['#F18F01', '#9b59b6'], alpha=0.7)
ax3.axhline(1, color='red', linestyle='--', label='Fair Pricing (IV=RV)', linewidth=2)
ax3.set_ylabel('Ratio')
ax3.set_title('Average IV/RV Ratio', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
for bar, ratio in zip(bars, ratios):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{ratio:.3f}', ha='center', fontsize=12, fontweight='bold')

# Panel 4: % Days IV > RV
ax4 = fig.add_subplot(gs[1, 0])
pcts = [stats_full['Pct_IV_GT_RV'], stats_test['Pct_IV_GT_RV']]
bars = ax4.bar(['Full Period', 'Test Period'], pcts, color=['#27ae60', '#e67e22'], alpha=0.7)
ax4.axhline(50, color='red', linestyle='--', label='50% threshold', linewidth=2)
ax4.set_ylabel('Percentage')
ax4.set_title('% Days IV > RV (Options Overpriced)', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, 100)
for bar, pct in zip(bars, pcts):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{pct:.1f}%', ha='center', fontsize=12, fontweight='bold')

# Panel 5: Annualized volatilities
ax5 = fig.add_subplot(gs[1, 1])
metrics_ann = ['RV Annual', 'IV Annual']
full_ann = [stats_full['RV_Annual_Mean'], stats_full['IV_Annual_Mean']]
test_ann = [stats_test['RV_Annual_Mean'], stats_test['IV_Annual_Mean']]
x = np.arange(len(metrics_ann))
bars1 = ax5.bar(x - width/2, full_ann, width, label='Periodo Completo', color='#2E86AB', alpha=0.7)
bars2 = ax5.bar(x + width/2, test_ann, width, label='Periodo Test', color='#e74c3c', alpha=0.7)
ax5.set_xlabel('Metric')
ax5.set_ylabel('Annualized Volatility (%)')
ax5.set_title('Annualized Volatility Comparison', fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_ann)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Panel 6: Key statistics table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
table_data = [
    ['Metric', 'Full', 'Test'],
    ['N Days', f"{stats_full['N_Days']}", f"{stats_test['N_Days']}"],
    ['RV Mean', f"{stats_full['RV_Mean']:.4f}%", f"{stats_test['RV_Mean']:.4f}%"],
    ['IV Mean', f"{stats_full['IV_Mean']:.4f}%", f"{stats_test['IV_Mean']:.4f}%"],
    ['VRP Mean', f"{stats_full['VRP_Mean']:.4f}%", f"{stats_test['VRP_Mean']:.4f}%"],
    ['Corr(RV,IV)', f"{stats_full['Correlation_RV_IV']:.4f}", f"{stats_test['Correlation_RV_IV']:.4f}"],
    ['IV/RV Ratio', f"{stats_full['IV_Over_RV_Mean']:.4f}", f"{stats_test['IV_Over_RV_Mean']:.4f}"],
    ['%IV>RV', f"{stats_full['Pct_IV_GT_RV']:.1f}%", f"{stats_test['Pct_IV_GT_RV']:.1f}%"],
]
table = ax6.table(cellText=table_data, loc='center', cellLoc='center',
                  colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)
for i in range(len(table_data)):
    for j in range(3):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#3498db')
            cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#ecf0f1')
ax6.set_title('Key Statistics Summary', fontweight='bold', pad=20)

# Panel 7-9: Time series mini-charts for test period
ax7 = fig.add_subplot(gs[2, :])
ax7.plot(df_test['date'], df_test['RV_pct'], linewidth=1, label='RV', color='#2E86AB', alpha=0.8)
ax7.plot(df_test['date'], df_test['IV_daily'], linewidth=1, label='IV', color='#F18F01', alpha=0.8)
ax7.fill_between(df_test['date'], df_test['RV_pct'], df_test['IV_daily'],
                 where=df_test['IV_daily'] > df_test['RV_pct'],
                 alpha=0.3, color='green', label='VRP > 0')
ax7.fill_between(df_test['date'], df_test['RV_pct'], df_test['IV_daily'],
                 where=df_test['IV_daily'] < df_test['RV_pct'],
                 alpha=0.3, color='red', label='VRP < 0')
ax7.set_title(f'Test Period Detail: {TEST_START.strftime("%Y-%m-%d")} to {TEST_END.strftime("%Y-%m-%d")}',
              fontweight='bold', fontsize=12)
ax7.set_xlabel('Fecha')
ax7.set_ylabel('Volatilidad (%)')
ax7.legend(loc='upper right')
ax7.grid(True, alpha=0.3)
ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax7.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

plt.savefig(os.path.join(RESULTS_DIR, '04_summary_dashboard.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: results/04_summary_dashboard.png")

# =============================================================================
# SECTION 8: SAVE STATISTICS TO CSV
# =============================================================================
print("\n[8] Saving statistics to CSV...")

# Create DataFrame with all statistics
stats_df = pd.DataFrame([stats_full, stats_test])
stats_df.to_csv('results/statistics_comparison.csv', index=False)
print("  Saved: results/statistics_comparison.csv")

# Save daily data for reference
df_full[['date', 'RV_pct', 'IV_daily', 'VRP', 'RV_annual', 'IV_annual']].to_csv(
    'results/daily_rv_iv_data.csv', index=False)
print("  Saved: results/daily_rv_iv_data.csv")

# =============================================================================
# SECTION 9: CONCLUSIONS
# =============================================================================
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print(f"""
1. PERIOD COVERAGE
   - Full Period: {stats_full['N_Days']} trading days ({stats_full['Start_Date']} to {stats_full['End_Date']})
   - Test Period: {stats_test['N_Days']} trading days ({stats_test['Start_Date']} to {stats_test['End_Date']})

2. VOLATILITY LEVELS
   - Full Period: RV mean = {stats_full['RV_Mean']:.4f}%, IV mean = {stats_full['IV_Mean']:.4f}%
   - Test Period: RV mean = {stats_test['RV_Mean']:.4f}%, IV mean = {stats_test['IV_Mean']:.4f}%

3. VARIANCE RISK PREMIUM (VRP = IV - RV)
   - Full Period: VRP mean = {stats_full['VRP_Mean']:.4f}% (IV {"overpriced" if stats_full['VRP_Mean'] > 0 else "underpriced"})
   - Test Period: VRP mean = {stats_test['VRP_Mean']:.4f}% (IV {"overpriced" if stats_test['VRP_Mean'] > 0 else "underpriced"})

4. IV PRICING ACCURACY
   - Full Period: IV > RV on {stats_full['Pct_IV_GT_RV']:.1f}% of days
   - Test Period: IV > RV on {stats_test['Pct_IV_GT_RV']:.1f}% of days

5. CORRELATION
   - Full Period: Correlation(RV, IV) = {stats_full['Correlation_RV_IV']:.4f}
   - Test Period: Correlation(RV, IV) = {stats_test['Correlation_RV_IV']:.4f}

6. IV/RV RATIO (Mispricing Indicator)
   - Full Period: Average IV/RV = {stats_full['IV_Over_RV_Mean']:.4f} ({"IV overprices by " + str(round((stats_full['IV_Over_RV_Mean']-1)*100, 1)) + "%" if stats_full['IV_Over_RV_Mean'] > 1 else "IV underprices"})
   - Test Period: Average IV/RV = {stats_test['IV_Over_RV_Mean']:.4f} ({"IV overprices by " + str(round((stats_test['IV_Over_RV_Mean']-1)*100, 1)) + "%" if stats_test['IV_Over_RV_Mean'] > 1 else "IV underprices"})

IMPLICATIONS FOR TRADING:
- VRP > 0 suggests selling options (short volatility) could be profitable
- VRP < 0 suggests buying options (long volatility) could be profitable
- High correlation indicates IV tracks RV reasonably well
- IV/RV ratio > 1 means options are systematically overpriced
""")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("""
Output files:
- results/01_rv_iv_timeseries.png
- results/02_distribution_comparison.png
- results/03_rolling_statistics.png
- results/04_summary_dashboard.png
- results/statistics_comparison.csv
- results/daily_rv_iv_data.csv
""")
