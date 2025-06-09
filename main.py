# main.py
# Main script to run the entire TGA analysis workflow.

import data_loader
import model_trainer
import predictor
import joblib
import config

def main():
    """
    Orchestrates the TGA data analysis workflow:
    1. Loads and preprocesses data.
    2. Trains and evaluates a model, saving the best one.
    3. Loads the saved model and performs extrapolation.
    """
    print("================================================")
    print(" TGA Analysis and Extrapolation Workflow Started ")
    print("================================================")

    # Step 1: Load and preprocess data
    # This function is responsible for all cleaning and preparation.
    all_data = data_loader.load_and_preprocess_data()

    if all_data.empty:
        print("\nWorkflow halted due to data loading failure.")
        return

    # Step 2: Train model, perform hyperparameter tuning, and save the best model
    # This function encapsulates the entire training and evaluation process.
    trained_model = model_trainer.train_and_evaluate_model(all_data)

    if trained_model is None:
        print("\nWorkflow halted due to model training failure.")
        return

    # Step 3: Load the saved model and perform prediction/extrapolation
    # We explicitly load the model from disk to ensure the saving/loading process works.
    print("\n--- Loading Saved Model for Prediction ---")
    try:
        loaded_model = joblib.load(config.BEST_MODEL_PATH)
        print("Model successfully loaded from disk.")
        
        # This function handles extrapolation for unseen rates and saves the results.
        predictor.extrapolate_and_save(loaded_model, all_data)
        
    except FileNotFoundError:
        print(f"ERROR: Could not find the saved model at {config.BEST_MODEL_PATH}. Cannot perform prediction.")
    except Exception as e:
        print(f"An error occurred during the prediction phase: {e}")

    print("\n================================================")
    print(" TGA Analysis and Extrapolation Workflow Finished ")
    print("================================================")

if __name__ == "__main__":
    main()
