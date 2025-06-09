# model_trainer.py
# Module for training models, performing hyperparameter tuning, and saving artifacts.

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import config # Import configurations

def train_and_evaluate_model(all_data):
    """
    Splits data, trains a RandomForest model with hyperparameter tuning,
    evaluates it, and saves the best model and its parameters.

    Args:
        all_data (pandas.DataFrame): The unified, preprocessed TGA data.

    Returns:
        sklearn.ensemble.RandomForestRegressor: The trained final model, or None if training fails.
    """
    print("\n--- Starting Model Training and Evaluation ---")
    try:
        X = all_data[config.FEATURE_COLS]
        y = all_data[config.TARGET_COL]

        # --- Data Splitting ---
        # Stratified split to ensure all heating rates are represented proportionally
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=X['Heating_Rate']
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=config.VALIDATION_SIZE, random_state=config.RANDOM_STATE, stratify=X_temp['Heating_Rate']
        )
        print(f"Data split complete: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

        # --- Hyperparameter Tuning with GridSearchCV ---
        print("\nPerforming Hyperparameter Tuning for RandomForestRegressor...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_leaf': [2, 5, 10]
        }
        
        rf = RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3,
                                   scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train, y_train)

        print(f"\nBest hyperparameters found: {grid_search.best_params_}")
        
        # Save the best hyperparameters to a file
        with open(config.BEST_HYPERPARAMS_PATH, 'w') as f:
            json.dump(grid_search.best_params_, f, indent=4)
        print(f"Best hyperparameters saved to: {config.BEST_HYPERPARAMS_PATH}")

        best_model = grid_search.best_estimator_

        # --- Training Final Model ---
        # Train on the combination of training and validation sets for final model
        print("\nTraining final model on combined train+validation data...")
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])
        final_model = clone(best_model)
        final_model.fit(X_train_val, y_train_val)

        # --- Evaluation on Test Set ---
        print("\nEvaluating final model on the unseen test set...")
        y_pred = final_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Test Set Performance:")
        print(f"  R-squared (R²): {r2:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")

        # --- Save the Final Trained Model ---
        joblib.dump(final_model, config.BEST_MODEL_PATH)
        print(f"\nFinal trained model saved to: {config.BEST_MODEL_PATH}")

        return final_model

    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return None

if __name__ == '__main__':
    # This block allows you to test this file directly
    print("Testing model_trainer.py as a standalone script...")
    # First, we need data. We'll call the data loader.
    from data_loader import load_and_preprocess_data
    data = load_and_preprocess_data()
    if not data.empty:
        train_and_evaluate_model(data)
    else:
        print("Skipping model training test because data loading failed.")
