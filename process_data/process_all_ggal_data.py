"""
Process all GGAL Excel files to create:
1. data.dat - Consolidated stock data with dividend adjustments
2. options_data_YYYY_MM.dat - Individual options data files for each opex period

Uses the complete dividend list (dividends_complete.csv) for accurate price adjustments
and the dividend detection algorithm to identify original strikes in options data.
"""

import pandas as pd
import numpy as np
from pyxlsb import open_workbook
import glob
import os
import re
from datetime import datetime, timedelta
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
OUTPUT_STOCK_FILE = os.path.join(SCRIPT_DIR, 'data.dat')
OUTPUT_OPTIONS_PREFIX = os.path.join(SCRIPT_DIR, 'options_data')
DIVIDENDS_FILE = os.path.join(SCRIPT_DIR, 'dividends_complete.csv')

# Column mappings
STOCK_COLUMNS_MAPPING = {
    'FECHA': 'Date',
    'ÚLTIMO': 'Last_Price',
    'APE.': 'Open',
    'MAX.': 'High',
    'MIN.': 'Low',
    'C. ANT.': 'Prev_Close',
    'MONTO $': 'Amount_Money',
    'NOMINAL': 'Volume_Units',
    'CANT OP.': 'Num_Trades',
    'HORA': 'Time',
    'VOLUMEN CALLS': 'Volume_Calls',
    'VOLUMEN PUTS': 'Volume_Put',
    'VI % CALL Prom.': 'IV_Call_Avg',
    'VI % PUT Prom.': 'IV_Put_Avg',
    'TLR': 'Risk_Free_Rate',
    'DÍAS AL VTO.': 'Days_To_Expiry',
    'PLAZO (años)': 'Years_To_Expiry'
}

OPTIONS_COLUMNS_MAPPING = {
    'FECHA': 'Date',
    'ESPECIE': 'Ticker',
    'BASE': 'Strike',
    'TIPO': 'Type',
    'ÚLTIMO': 'Last_Price',
    'MONTO': 'Amount_Money',
    'APE': 'Open',
    'APE.': 'Open',
    'MAX': 'High',
    'MAX.': 'High',
    'MIN': 'Low',
    'MIN.': 'Low',
    'C. ANT': 'Prev_Close',
    'C. ANT.': 'Prev_Close',
    'NOMINAL': 'Volume_Units',
    'VI %': 'Implied_Volatility',
    'VE %': 'VE_Pct',
    'PARIDAD': 'Parity'
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_year_month_from_filename(filename):
    """Extract year and month from filename like '$GGAL - Info Opex 2021-04.xlsb'"""
    match = re.search(r'(\d{4})-(\d{2})', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def convert_excel_date(date_value):
    """Convert Excel date (number or string) to datetime"""
    if pd.isna(date_value):
        return pd.NaT

    if isinstance(date_value, (int, float)):
        # Excel date format (days since 1899-12-30)
        return pd.Timestamp('1899-12-30') + pd.Timedelta(days=date_value)
    else:
        # String format
        try:
            return pd.to_datetime(date_value, format='%d/%m/%Y', dayfirst=True)
        except:
            return pd.to_datetime(date_value, dayfirst=True)

def find_sheets(wb_sheets):
    """Find stock and options sheet names from workbook sheets list"""
    stock_sheet = None
    options_sheet = None

    for sheet_name in wb_sheets:
        sheet_lower = sheet_name.lower()
        if 'ggal' in sheet_lower and 'lote' not in sheet_lower:
            stock_sheet = sheet_name
        if 'lote' in sheet_lower:
            options_sheet = sheet_name

    return stock_sheet, options_sheet

def read_sheet_data(filepath, sheet_name):
    """Read data from a sheet in xlsb file"""
    with open_workbook(filepath) as wb:
        with wb.get_sheet(sheet_name) as sheet:
            data = []
            for row in sheet.rows():
                data.append([item.v for item in row])

    if len(data) < 2:
        return None

    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.dropna(how='all')
    return df

def detect_dividends_in_period(df_options_raw):
    """
    Detect dividend payments from option strike changes using the improved algorithm.
    Returns: list of dict with 'date' and 'dividend' keys
    """
    # Convert dates and strikes
    df = df_options_raw.copy()
    df['Date'] = df['FECHA'].apply(convert_excel_date)
    df['Strike'] = pd.to_numeric(df['BASE'], errors='coerce')
    df['Last_Price'] = pd.to_numeric(df['ÚLTIMO'], errors='coerce')

    # Remove invalid data
    df = df[df['Date'].notna() & df['Strike'].notna()].copy()
    df = df.sort_values(['Date', 'Strike']).reset_index(drop=True)

    detected_dividends = []
    dates = sorted(df['Date'].unique())

    for i in range(len(dates) - 1):
        today = dates[i]
        tomorrow = dates[i + 1]

        # Get ALL strikes for each day
        strikes_today = sorted(df[df['Date'] == today]['Strike'].round(2).unique())
        strikes_tomorrow = sorted(df[df['Date'] == tomorrow]['Strike'].round(2).unique())

        if len(strikes_today) < 3 or len(strikes_tomorrow) < 3:
            continue

        size_changed = len(strikes_today) != len(strikes_tomorrow)

        if size_changed:
            old_size = len(strikes_today)
            new_size = len(strikes_tomorrow)

            if new_size > old_size:
                # Size INCREASED: new strikes added
                strikes_tomorrow_first = strikes_tomorrow[:old_size]
                strikes_tomorrow_last = strikes_tomorrow[-old_size:]

                best_match = None
                best_differences = []

                for candidate_tomorrow in [strikes_tomorrow_first, strikes_tomorrow_last]:
                    differences = [st_today - st_tomorrow for st_today, st_tomorrow in zip(strikes_today, candidate_tomorrow)]

                    if len(differences) >= 3:
                        rounded_diffs = [round(d, 1) for d in differences if abs(d) > 0.01]

                        if len(rounded_diffs) >= 3:
                            counter = Counter(rounded_diffs)
                            most_common_value, most_common_count = counter.most_common(1)[0]

                            if most_common_count >= len(rounded_diffs) * 0.7 and most_common_value > 0.5:
                                close_diffs = [d for d in differences if abs(d - most_common_value) < 0.5]

                                if len(close_diffs) >= 3:
                                    avg_div = np.mean(close_diffs)
                                    std_div = np.std(close_diffs)

                                    if best_match is None or std_div < np.std(best_differences):
                                        best_match = candidate_tomorrow
                                        best_differences = close_diffs

                if best_match is not None:
                    avg_div = np.mean(best_differences)
                    std_div = np.std(best_differences)

                    detected_dividends.append({
                        'date': tomorrow,
                        'dividend': round(avg_div, 6)
                    })

            else:
                # Size DECREASED: strikes removed
                num_removed = old_size - new_size

                candidates_today = []
                candidates_today.append(strikes_today[num_removed:])
                candidates_today.append(strikes_today[:-num_removed] if num_removed > 0 else strikes_today)

                if old_size > num_removed + 1:
                    candidates_today.append([strikes_today[0]] + strikes_today[num_removed+1:])

                if old_size > num_removed + 1:
                    candidates_today.append(strikes_today[:-(num_removed+1)] + [strikes_today[-1]])

                best_match = None
                best_differences = []

                for candidate_today in candidates_today:
                    if len(candidate_today) != new_size:
                        continue

                    differences = [st_today - st_tomorrow for st_today, st_tomorrow in zip(candidate_today, strikes_tomorrow)]
                    non_zero_diffs = [d for d in differences if abs(d) > 0.01]

                    if len(non_zero_diffs) >= 3:
                        rounded_diffs = [round(d, 1) for d in non_zero_diffs]
                        counter = Counter(rounded_diffs)
                        most_common_value, most_common_count = counter.most_common(1)[0]

                        if most_common_count >= len(non_zero_diffs) * 0.7 and most_common_value > 0.5:
                            close_diffs = [d for d in non_zero_diffs if abs(d - most_common_value) < 0.5]

                            if len(close_diffs) >= 3:
                                avg_div = np.mean(close_diffs)
                                std_div = np.std(close_diffs)

                                if best_match is None or std_div < np.std(best_differences):
                                    best_match = candidate_today
                                    best_differences = close_diffs

                if best_match is not None:
                    avg_div = np.mean(best_differences)
                    std_div = np.std(best_differences)

                    detected_dividends.append({
                        'date': tomorrow,
                        'dividend': round(avg_div, 6)
                    })

        else:
            # Size same: check for dividend after removing invalid strikes
            strikes_today_set = set(strikes_today)
            strikes_tomorrow_set = set(strikes_tomorrow)
            common_strikes = strikes_today_set & strikes_tomorrow_set

            valid_strikes_today = [s for s in strikes_today if s not in common_strikes]
            valid_strikes_tomorrow = [s for s in strikes_tomorrow if s not in common_strikes]

            if len(valid_strikes_today) != len(valid_strikes_tomorrow):
                continue

            if len(valid_strikes_today) < 3:
                continue

            differences = [st_today - st_tomorrow for st_today, st_tomorrow in zip(valid_strikes_today, valid_strikes_tomorrow)]

            if len(differences) >= 3:
                avg_div = np.mean(differences)
                std_div = np.std(differences)

                if avg_div > 0.5 and std_div < (avg_div * 0.10):
                    detected_dividends.append({
                        'date': tomorrow,
                        'dividend': round(avg_div, 6)
                    })

    return detected_dividends

def load_complete_dividends():
    """Load the complete dividend list from CSV"""
    df_div = pd.read_csv(DIVIDENDS_FILE)
    df_div['Date'] = pd.to_datetime(df_div['Date'])
    return df_div

def apply_dividend_adjustments_yahoo_style(df_stock, df_dividends):
    """
    Apply dividend adjustments to stock prices using Yahoo Finance method.
    Adjusted close should have same returns as unadjusted close between dividend dates.
    """
    df = df_stock.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    # Initialize
    df['Dividend'] = 0.0
    df['Adj_Price'] = df['Last_Price'].copy()

    # Mark dividend dates
    for _, div_row in df_dividends.iterrows():
        div_date = div_row['Date']
        div_amount = div_row['Dividend']

        # Find matching date in stock data (within 3 days)
        nearby = df[(df['Date'] >= div_date - timedelta(days=3)) &
                    (df['Date'] <= div_date + timedelta(days=3))]

        if len(nearby) > 0:
            closest_date = nearby.iloc[0]['Date']
            df.loc[df['Date'] == closest_date, 'Dividend'] = div_amount

    # Calculate adjustment factors (Yahoo Finance style)
    # Work backward from most recent date
    df = df.sort_values('Date', ascending=False).reset_index(drop=True)

    cumulative_factor = 1.0

    for idx, row in df.iterrows():
        df.at[idx, 'Adj_Price'] = row['Last_Price'] * cumulative_factor

        if row['Dividend'] > 0:
            # Adjustment factor = (Price - Dividend) / Price
            adjustment_factor = (row['Last_Price'] - row['Dividend']) / row['Last_Price']
            cumulative_factor *= adjustment_factor

    # Sort back to ascending order
    df = df.sort_values('Date').reset_index(drop=True)

    return df

def adjust_options_original_strikes(df_options, detected_dividends, df_complete_dividends):
    """
    Calculate Original_Strike for options based on dividends paid DURING THIS OPEX PERIOD ONLY.
    Original_Strike = Strike + cumulative_dividends_paid_during_this_opex_after_option_date

    Only dividends that occurred within the date range of this options file are considered.
    This ensures that Original_Strike reflects adjustments made to THIS option series,
    not historical dividends from before the options were created.
    """
    df = df_options.copy()
    df['Original_Strike'] = df['Strike'].copy()

    # Get the date range of this OPEX period
    opex_start = df['Date'].min()
    opex_end = df['Date'].max()

    # Filter dividends to only those within this OPEX period
    opex_dividends = df_complete_dividends[
        (df_complete_dividends['Date'] >= opex_start) &
        (df_complete_dividends['Date'] <= opex_end)
    ]

    # Apply only dividends that occurred during this OPEX
    for _, div_row in opex_dividends.iterrows():
        div_date = div_row['Date']
        div_amount = div_row['Dividend']

        # Find options after this dividend date
        mask = df['Date'] > div_date
        df.loc[mask, 'Original_Strike'] += div_amount

    return df

def process_stock_data(df_raw):
    """Process raw stock data and return cleaned DataFrame"""
    if 'FECHA' not in df_raw.columns:
        return None

    df_raw = df_raw[df_raw['FECHA'].notna()]

    # Select and rename columns
    existing_cols = [col for col in STOCK_COLUMNS_MAPPING.keys() if col in df_raw.columns]
    df = df_raw[existing_cols].copy()
    df.rename(columns={k: STOCK_COLUMNS_MAPPING[k] for k in existing_cols}, inplace=True)

    # Convert dates
    df['Date'] = df['Date'].apply(convert_excel_date)

    # Convert numeric columns
    numeric_cols = ['Last_Price', 'Open', 'High', 'Low', 'Prev_Close', 'Amount_Money',
                    'Volume_Units', 'Num_Trades', 'Volume_Calls', 'Volume_Put',
                    'IV_Call_Avg', 'IV_Put_Avg', 'Risk_Free_Rate', 'Days_To_Expiry',
                    'Years_To_Expiry']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows where Last_Price is NaN (holidays/missing data)
    df = df[df['Last_Price'].notna()].copy()

    df = df.sort_values('Date').reset_index(drop=True)

    # Calculate returns using shifted Last_Price (not Prev_Close which has zero values)
    df['Simple_Return'] = df['Last_Price'].pct_change()
    df['Log_Return'] = np.log(df['Last_Price'] / df['Last_Price'].shift(1))

    return df

def process_options_data(df_raw):
    """Process raw options data and return cleaned DataFrame"""
    if 'FECHA' not in df_raw.columns:
        return None

    df_raw = df_raw[df_raw['FECHA'].notna()]

    # Select and rename columns
    existing_cols = [col for col in OPTIONS_COLUMNS_MAPPING.keys() if col in df_raw.columns]
    df = df_raw[existing_cols].copy()
    df.rename(columns={k: OPTIONS_COLUMNS_MAPPING[k] for k in existing_cols}, inplace=True)

    # Convert dates
    df['Date'] = df['Date'].apply(convert_excel_date)

    # Convert numeric columns
    numeric_cols = ['Strike', 'Last_Price', 'Amount_Money', 'Open', 'High', 'Low',
                    'Prev_Close', 'Volume_Units', 'Implied_Volatility', 'VE_Pct', 'Parity']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate returns
    df['Return'] = 0.0
    if 'Prev_Close' in df.columns:
        valid_mask = (df['Last_Price'] > 0) & (df['Prev_Close'] > 0)
        df.loc[valid_mask, 'Return'] = (df.loc[valid_mask, 'Last_Price'] / df.loc[valid_mask, 'Prev_Close']) - 1

    df = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)

    return df

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("="*80)
    print("PROCESSING ALL GGAL DATA FILES")
    print("="*80)

    # Load complete dividend list
    print("\nLoading complete dividend list...")
    df_complete_dividends = load_complete_dividends()
    print(f"Loaded {len(df_complete_dividends)} dividends/coupons")

    # Find all Excel files
    excel_files = sorted(glob.glob(f'{DATA_DIR}/$GGAL - Info Opex*.xlsb'))
    print(f"\nFound {len(excel_files)} Excel files")

    # Storage for all data
    all_stock_data = []
    all_detected_dividends = []

    # Process each file
    for i, filepath in enumerate(excel_files, 1):
        filename = os.path.basename(filepath)
        year, month = extract_year_month_from_filename(filename)

        if not year or not month:
            print(f"\n[{i}/{len(excel_files)}] ⚠️  {filename} - Could not extract year/month")
            continue

        print(f"\n[{i}/{len(excel_files)}] Processing {filename}")

        # Find sheet names
        with open_workbook(filepath) as wb:
            sheet_names = wb.sheets

        stock_sheet, options_sheet = find_sheets(sheet_names)

        if not stock_sheet or not options_sheet:
            print(f"  ⚠️  Could not find both sheets (stock: {stock_sheet}, options: {options_sheet})")
            continue

        # Read raw data
        df_stock_raw = read_sheet_data(filepath, stock_sheet)
        df_options_raw = read_sheet_data(filepath, options_sheet)

        if df_stock_raw is None or df_options_raw is None:
            print(f"  ⚠️  Could not read data")
            continue

        # Process stock data
        df_stock = process_stock_data(df_stock_raw)
        if df_stock is not None and len(df_stock) > 0:
            all_stock_data.append(df_stock)
            print(f"  ✓ Stock: {len(df_stock)} rows")

        # Detect dividends in this period
        detected_divs = detect_dividends_in_period(df_options_raw)
        all_detected_dividends.extend(detected_divs)

        if len(detected_divs) > 0:
            print(f"  ✓ Detected {len(detected_divs)} dividend(s) in options data")
            for div in detected_divs:
                print(f"     {div['date'].date()}: ${div['dividend']:.6f}")

        # Process options data
        df_options = process_options_data(df_options_raw)

        if df_options is not None and len(df_options) > 0:
            # Adjust Original_Strike based on complete dividend list
            df_options = adjust_options_original_strikes(df_options, detected_divs, df_complete_dividends)

            # Save individual options file
            output_options_file = f'{OUTPUT_OPTIONS_PREFIX}_{year}_{month}.dat'
            df_options.to_csv(output_options_file, index=False)
            print(f"  ✓ Options: {len(df_options)} rows → {output_options_file}")

    # Consolidate all stock data
    if len(all_stock_data) > 0:
        print("\n" + "="*80)
        print("CONSOLIDATING STOCK DATA")
        print("="*80)

        df_all_stock = pd.concat(all_stock_data, ignore_index=True)
        df_all_stock = df_all_stock.sort_values('Date').drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)

        print(f"\nTotal stock records before adjustment: {len(df_all_stock)}")
        print(f"Date range: {df_all_stock['Date'].min().date()} to {df_all_stock['Date'].max().date()}")

        # Apply dividend adjustments
        print("\nApplying dividend adjustments (Yahoo Finance style)...")
        df_all_stock = apply_dividend_adjustments_yahoo_style(df_all_stock, df_complete_dividends)

        # Save consolidated stock data
        df_all_stock.to_csv(OUTPUT_STOCK_FILE, index=False)
        print(f"\n✓ Saved consolidated stock data: {OUTPUT_STOCK_FILE}")
        print(f"  Records: {len(df_all_stock)}")
        print(f"  Columns: {', '.join(df_all_stock.columns)}")

        # Summary of adjustments
        num_divs_found = (df_all_stock['Dividend'] > 0).sum()
        total_divs = df_all_stock['Dividend'].sum()
        print(f"\n  Dividend adjustments: {num_divs_found} dates, total ${total_divs:.2f}")

        # Check for negative adjusted prices
        negative_adj = df_all_stock[df_all_stock['Adj_Price'] < 0]
        if len(negative_adj) > 0:
            print(f"  ⚠️  WARNING: {len(negative_adj)} records with negative Adj_Price")
        else:
            print(f"  ✓ No negative adjusted prices")

    # Summary of detected dividends vs complete list
    print("\n" + "="*80)
    print("DIVIDEND DETECTION SUMMARY")
    print("="*80)

    print(f"\nComplete dividend list: {len(df_complete_dividends)} entries")
    print(f"Detected from options: {len(all_detected_dividends)} entries")

    # Match detected with complete
    df_detected = pd.DataFrame(all_detected_dividends)
    if len(df_detected) > 0:
        print("\nMatching detected dividends with complete list:")
        for _, detected in df_detected.iterrows():
            det_date = detected['date']
            det_div = detected['dividend']

            # Find nearby in complete list
            nearby = df_complete_dividends[
                (df_complete_dividends['Date'] >= det_date - timedelta(days=3)) &
                (df_complete_dividends['Date'] <= det_date + timedelta(days=3))
            ]

            if len(nearby) > 0:
                actual = nearby.iloc[0]
                diff = abs(det_div - actual['Dividend'])
                print(f"  {det_date.date()} ${det_div:.3f} ↔ {actual['Date'].date()} ${actual['Dividend']:.3f} (diff: ${diff:.3f})")
            else:
                print(f"  {det_date.date()} ${det_div:.3f} ← NOT IN COMPLETE LIST")

    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
