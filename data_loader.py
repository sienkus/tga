# data_loader.py
# Module for loading, cleaning, and preparing TGA data.

import pandas as pd
import config # Import configurations from our config file

def load_and_preprocess_data():
    """
    Loads TGA data from the specified Excel file, cleans it, standardizes column names,
    and returns a single unified DataFrame.

    Returns:
        pandas.DataFrame: A unified DataFrame containing cleaned data from all sheets,
                          or an empty DataFrame if loading fails.
    """
    all_data_list = []
    print("--- Starting Data Loading and Preprocessing ---")

    for rate, sheet in config.SHEET_NAMES.items():
        print(f"\nProcessing Sheet: {sheet} (Heating Rate: {rate}°C/min)")
        try:
            header_row_index = 1 if rate in config.RATES_WITH_HEADER_ROW_2 else 0
            if header_row_index == 1:
                print(f"  Note: Reading sheet with header from row {header_row_index + 1}.")

            df_raw = pd.read_excel(config.EXCEL_FILE_PATH, sheet_name=sheet, header=header_row_index)
            print(f"  DEBUG: Original columns read: {df_raw.columns.tolist()}")

            # Clean and rename columns
            original_columns_stripped = [str(col).strip() for col in df_raw.columns]
            df_raw.columns = original_columns_stripped
            
            rename_map = config.RENAMING_MAPS.get(rate, {})
            df_raw.rename(columns=rename_map, inplace=True)
            print(f"  Standardized columns: {df_raw.columns.tolist()}")
            
            # Check for required columns
            missing_std = [col for col in config.REQUIRED_STD_COLS if col not in df_raw.columns]
            if missing_std:
                print(f"  ERROR: Missing required columns after renaming: {missing_std}. Skipping sheet.")
                continue

            # Add heating rate and select relevant columns
            df_raw['Heating_Rate'] = rate
            all_std_cols = list(config.RENAMING_MAPS[rate].values())
            cols_to_keep = [col for col in all_std_cols if col in df_raw.columns] + ['Heating_Rate']
            df_sheet = df_raw[cols_to_keep].copy()

            # Convert to numeric and drop invalid rows
            for col in config.REQUIRED_STD_COLS:
                df_sheet[col] = pd.to_numeric(df_sheet[col], errors='coerce')
            
            initial_rows = len(df_sheet)
            df_sheet.dropna(subset=config.REQUIRED_STD_COLS, inplace=True)
            if initial_rows > len(df_sheet):
                print(f"  Dropped {initial_rows - len(df_sheet)} rows with invalid numeric data.")

            if df_sheet.empty:
                print("  WARNING: No valid data remains for this sheet after cleaning.")
                continue
            
            print(f"  Successfully processed sheet. Shape: {df_sheet.shape}")
            all_data_list.append(df_sheet)

        except FileNotFoundError:
            print(f"FATAL ERROR: Excel file not found at '{config.EXCEL_FILE_PATH}'. Please check config.py.")
            return pd.DataFrame() # Return empty dataframe on critical error
        except Exception as e:
            print(f"  An unexpected error occurred processing sheet '{sheet}': {e}")
    
    if not all_data_list:
        print("\nFATAL ERROR: No data was successfully loaded from any sheet. Halting.")
        return pd.DataFrame()
        
    all_data = pd.concat(all_data_list, ignore_index=True)
    print("\n--- Data Loading and Preprocessing Complete ---")
    print(f"Total combined data points: {len(all_data)}")
    print(f"Heating rates loaded: {sorted(all_data['Heating_Rate'].unique())}")
    
    return all_data

if __name__ == '__main__':
    # This block allows you to run this file directly to test data loading
    print("Testing data_loader.py as a standalone script...")
    combined_data = load_and_preprocess_data()
    if not combined_data.empty:
        print("\nTest successful. Combined DataFrame head:")
        print(combined_data.head())
        print("\nCombined DataFrame info:")
        combined_data.info()
