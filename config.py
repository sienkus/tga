# config.py
# Central configuration file for the TGA analysis project.

import os

# --- Directory Setup ---
# Create an 'outputs' directory to save models, results, and plots if it doesn't exist
# This keeps the project tidy.
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- File Paths ---
# !! IMPORTANT: Update this path to where your Excel file is located !!
EXCEL_FILE_PATH = '/Users/sienkadounia/Downloads/TGA- empty fruit bunch - to Hassan (1).xlsx'

# Paths for saving outputs
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_random_forest_model.joblib")
BEST_HYPERPARAMS_PATH = os.path.join(OUTPUT_DIR, "best_hyperparameters.json")
EXTRAPOLATION_RESULTS_PATH = os.path.join(OUTPUT_DIR, "extrapolated_tga_predictions.csv")
DTG_RESULTS_PATH = os.path.join(OUTPUT_DIR, "derived_dtg_predictions.csv")

# --- Data Loading Configuration ---
SHEET_NAMES = {
    10: '10°Cmin',
    20: '20°Cmin',
    50: '50°Cmin'
}

# Specify which rates have the header in the second row (index 1)
# CORRECTED: All sheets appear to need the header skipped.
RATES_WITH_HEADER_ROW_2 = [10, 20, 50]

# --- Column Name Standardization ---
# Define the consistent, internal names we will use after loading.
STD_COL_TIME = 'Time_min'
STD_COL_TEMP = 'Temperature_C'
STD_COL_MASS = 'Mass_mg'
STD_COL_ATG = 'ATG_percent'
STD_COL_DTG_TIME = 'DTG_percent_per_min'
STD_COL_DTG_TEMP = 'DTG_percent_per_C'

# Define the original column names for each sheet to map FROM.
# These must be exact matches to the names in the Excel file after stripping whitespace.
RENAMING_MAPS = {
    10: {
        # The log showed ['Size', 6.249, 'mg',...]. The real headers should be read now.
        # Based on previous user input for 10C/min.
        't (min)': STD_COL_TIME,
        'T (°C)': STD_COL_TEMP,
        'mg': STD_COL_MASS, 
        'ATG (%)': STD_COL_ATG,
        'Deriv, Weight (%/min)': STD_COL_DTG_TIME,
        'Deriv, Weight (%/°C)': STD_COL_DTG_TEMP
    },
    20: {
        't': STD_COL_TIME,
        'T': STD_COL_TEMP,
        'mg': STD_COL_MASS,
        'ATG (%)': STD_COL_ATG,
        'DTG (%/min)': STD_COL_DTG_TIME,
        'DTG (%/°C)': STD_COL_DTG_TEMP
    },
    50: {
        'Time (min)': STD_COL_TIME,
        'Temperature (°C)': STD_COL_TEMP,
        'Weight (mg)': STD_COL_MASS,
        'ATG (%)': STD_COL_ATG,
        'DTG (%/min)': STD_COL_DTG_TIME,
        'DTG (%/°C)': STD_COL_DTG_TEMP
    }
}


# --- Modeling Configuration ---
# The columns we absolutely NEED for the model (using standard names)
REQUIRED_STD_COLS = [STD_COL_TEMP, STD_COL_ATG]
# Features we will use for modeling: X = f(T, Rate) -> ATG
FEATURE_COLS = [STD_COL_TEMP, 'Heating_Rate']
TARGET_COL = STD_COL_ATG

# Data splitting parameters
TEST_SIZE = 0.3  # 30% for validation + test
VALIDATION_SIZE = 0.5 # 50% of the 30% -> 15% of total for validation
RANDOM_STATE = 42

# --- Extrapolation Configuration ---
UNSEEN_RATES = [30, 40, 60]
INTERPOLATION_POINTS = 200 # Number of points for smooth extrapolated curves
BOOTSTRAP_ITERATIONS = 100 # Number of bootstrap samples for confidence intervals
