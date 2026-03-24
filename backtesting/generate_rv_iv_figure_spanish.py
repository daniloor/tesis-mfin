"""
Generate RV vs IV figure for thesis in Spanish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
warnings.filterwarnings('ignore')

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTRADAY_DIR = os.path.join(SCRIPT_DIR, '..', 'intraday')
PROCESS_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'process_data')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

print("Generando figura RV vs IV en español...")

# Load data
df_10min = pd.read_csv(os.path.join(INTRADAY_DIR, 'BCBA_DLY_GGAL, 10 (1).csv'))
df_10min['time'] = pd.to_datetime(df_10min['time'])
df_10min = df_10min.sort_values('time').reset_index(drop=True)

df_daily = pd.read_csv(os.path.join(PROCESS_DATA_DIR, 'data.dat'))
df_daily['Date'] = pd.to_datetime(df_daily['Date'])
df_daily = df_daily.sort_values('Date').reset_index(drop=True)

# Calculate RV
def calc_rv(df, min_bars=30):
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
    daily['RV_pct'] = daily['RV'] * 100
    daily = daily[daily['n_bars'] >= min_bars].reset_index(drop=True)
    return daily

df_rv = calc_rv(df_10min, min_bars=30)

# Merge with IV
df_iv = df_daily[['Date', 'IV_Call_Avg', 'IV_Put_Avg']].copy()
df_iv.columns = ['date', 'IV_Call', 'IV_Put']
df = df_rv.merge(df_iv, on='date', how='inner')
df['IV_daily'] = ((df['IV_Call'] + df['IV_Put']) / 2) * 100 / np.sqrt(252)
df['VRP'] = df['IV_daily'] - df['RV_pct']

# Define test period
TEST_START = pd.Timestamp('2025-02-21')
TEST_END = pd.Timestamp('2025-12-18')

# Create figure with 3 panels
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Panel 1: Full Period - RV vs IV
ax1 = axes[0]
ax1.plot(df['date'], df['RV_pct'], linewidth=0.8, label='Volatilidad Realizada (RV)',
         color='#2E86AB', alpha=0.8)
ax1.plot(df['date'], df['IV_daily'], linewidth=0.8, label='Volatilidad Implícita (IV)',
         color='#F18F01', alpha=0.8)
ax1.axvline(x=TEST_START, color='green', linestyle='--', linewidth=1.5, label='Inicio período test')
ax1.axvline(x=TEST_END, color='red', linestyle='--', linewidth=1.5, label='Fin período test')
ax1.axhline(y=df['RV_pct'].mean(), color='#2E86AB', linestyle=':', alpha=0.5)
ax1.axhline(y=df['IV_daily'].mean(), color='#F18F01', linestyle=':', alpha=0.5)
ax1.set_title('Período Completo: Volatilidad Realizada vs Volatilidad Implícita (Diaria %)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Volatilidad (%)')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

# Panel 2: Test Period Only
df_test = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy()
ax2 = axes[1]
ax2.plot(df_test['date'], df_test['RV_pct'], linewidth=1.2, label='Volatilidad Realizada (RV)',
         color='#2E86AB', alpha=0.9)
ax2.plot(df_test['date'], df_test['IV_daily'], linewidth=1.2, label='Volatilidad Implícita (IV)',
         color='#F18F01', alpha=0.9)
ax2.axhline(y=df_test['RV_pct'].mean(), color='#2E86AB', linestyle='--',
            label=f'Media RV: {df_test["RV_pct"].mean():.2f}%')
ax2.axhline(y=df_test['IV_daily'].mean(), color='#F18F01', linestyle='--',
            label=f'Media IV: {df_test["IV_daily"].mean():.2f}%')
ax2.set_title(f'Período de Test ({TEST_START.strftime("%d-%m-%Y")} a {TEST_END.strftime("%d-%m-%Y")}): RV vs IV',
              fontweight='bold', fontsize=12)
ax2.set_ylabel('Volatilidad (%)')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

# Panel 3: VRP
ax3 = axes[2]
ax3.plot(df['date'], df['VRP'], linewidth=0.8, color='#27ae60', alpha=0.7)
ax3.fill_between(df['date'], 0, df['VRP'],
                 where=df['VRP'] > 0, alpha=0.3, color='green', label='VRP > 0 (IV sobrepriceada)')
ax3.fill_between(df['date'], 0, df['VRP'],
                 where=df['VRP'] < 0, alpha=0.3, color='red', label='VRP < 0 (IV subpriceada)')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.axvline(x=TEST_START, color='green', linestyle='--', linewidth=1.5)
ax3.axvline(x=TEST_END, color='red', linestyle='--', linewidth=1.5)
ax3.axhline(y=df['VRP'].mean(), color='purple', linestyle='--',
            label=f'VRP media: {df["VRP"].mean():.2f}%')
ax3.set_title('Prima de Riesgo de Varianza (VRP = IV - RV)', fontweight='bold', fontsize=12)
ax3.set_xlabel('Fecha')
ax3.set_ylabel('VRP (%)')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

plt.tight_layout()

# Save to thesis figures folder
output_path = '/home/dorradre/daniloor/tesis/tesis/figures/rv_vs_iv_timeseries.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figura guardada en: {output_path}")
print("¡Listo!")
