# predictor.py
# Module for loading a trained model and making predictions.

import pandas as pd
import numpy as np
import joblib
from scipy.signal import savgol_filter
import warnings
import config # Import configurations

def extrapolate_and_save(model, original_data):
    """
    Uses a trained model to extrapolate TGA curves for unseen heating rates,
    calculates DTG, and saves the results to CSV files.

    Args:
        model (sklearn.base.BaseEstimator): The trained machine learning model.
        original_data (pandas.DataFrame): The original data, used for defining temp range.
    """
    print("\n--- Starting Extrapolation for Unseen Heating Rates ---")
    
    # Define temperature range for prediction curves
    min_temp = original_data[config.STD_COL_TEMP].min()
    max_temp = original_data[config.STD_COL_TEMP].max()
    temp_range = np.linspace(min_temp, max_temp, num=config.INTERPOLATION_POINTS)
    
    # Create input DataFrame for all unseen rates
    X_unseen_list = []
    for rate in config.UNSEEN_RATES:
        df_rate_new = pd.DataFrame({
            config.STD_COL_TEMP: temp_range,
            'Heating_Rate': rate
        })
        X_unseen_list.append(df_rate_new)
    
    X_unseen = pd.concat(X_unseen_list, ignore_index=True)
    
    # Predict ATG using the loaded model
    print(f"Predicting ATG for rates {config.UNSEEN_RATES}...")
    predictions_atg = model.predict(X_unseen[config.FEATURE_COLS])
    X_unseen['Predicted_ATG_percent'] = predictions_atg
    
    # Save the ATG predictions
    X_unseen.to_csv(config.EXTRAPOLATION_RESULTS_PATH, index=False)
    print(f"Extrapolated TGA predictions saved to: {config.EXTRAPOLATION_RESULTS_PATH}")
    
    # --- Calculate and Save DTG ---
    print("\nCalculating DTG from predicted TGA curves...")
    dtg_results_list = []
    for rate in config.UNSEEN_RATES:
        rate_data = X_unseen[X_unseen['Heating_Rate'] == rate].copy()
        rate_data.sort_values(by=config.STD_COL_TEMP, inplace=True)
        
        atg_predicted = rate_data['Predicted_ATG_percent'].values
        temperatures = rate_data[config.STD_COL_TEMP].values
        
        # Smooth predicted ATG before differentiation for cleaner DTG
        if len(atg_predicted) > 5:
            atg_predicted_smooth = savgol_filter(atg_predicted, window_length=5, polyorder=2)
        else:
            atg_predicted_smooth = atg_predicted
            
        delta_atg = np.diff(atg_predicted_smooth)
        delta_temp = np.diff(temperatures)
        
        # Avoid division by zero
        dtg_values = np.divide(delta_atg, delta_temp, out=np.zeros_like(delta_atg), where=delta_temp!=0)
        dtg_values_positive = -dtg_values
        
        df_dtg_rate = pd.DataFrame({
            config.STD_COL_TEMP: temperatures[1:],
            'Heating_Rate': rate,
            'Predicted_DTG_percent_per_C': dtg_values_positive
        })
        dtg_results_list.append(df_dtg_rate)
    
    if dtg_results_list:
        all_dtg_predictions = pd.concat(dtg_results_list, ignore_index=True)
        all_dtg_predictions.to_csv(config.DTG_RESULTS_PATH, index=False)
        print(f"Derived DTG predictions saved to: {config.DTG_RESULTS_PATH}")
    else:
        print("Could not generate DTG predictions.")

if __name__ == '__main__':
    # This block allows you to test this file directly
    print("Testing predictor.py as a standalone script...")
    try:
        print("Loading pre-trained model...")
        loaded_model = joblib.load(config.BEST_MODEL_PATH)
        
        # Predictor needs original data to set temperature range, so we load it.
        from data_loader import load_and_preprocess_data
        data = load_and_preprocess_data()
        
        if loaded_model and not data.empty:
            extrapolate_and_save(loaded_model, data)
        else:
            print("Skipping prediction test because model or data could not be loaded.")

    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{config.BEST_MODEL_PATH}'.")
        print("Please run model_trainer.py first to train and save a model.")
    except Exception as e:
        print(f"An error occurred: {e}")
