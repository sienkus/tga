TGA Data Analysis and Extrapolation Project

This project provides a complete workflow for analyzing Thermogravimetric Analysis (TGA) data using machine learning. It loads experimental data, trains a predictive model, and uses that model to extrapolate TGA and DTG curves for unseen heating rates.
Project Structure

The project is organized into modular Python scripts for clarity and maintainability:

    main.py: The main entry point to run the entire workflow.

    config.py: A centralized configuration file for file paths, model parameters, and other settings. You must edit this file first.

    data_loader.py: Handles loading the raw TGA data from the Excel file, cleaning it, and standardizing the format.

    model_trainer.py: Contains the logic for training the machine learning model (Random Forest), performing hyperparameter tuning with GridSearchCV, evaluating performance, and saving the best model and its parameters.

    predictor.py: Contains functions to load a saved model, make predictions for new (unseen) heating rates, derive DTG curves, and save the results.

    requirements.txt: Lists all the necessary Python packages for this project.

    outputs/: A directory that will be automatically created to store all generated files:

        best_random_forest_model.joblib: The saved, trained machine learning model.

        best_hyperparameters.json: The optimal hyperparameters found during tuning.

        extrapolated_tga_predictions.csv: The predicted TGA curves for the unseen heating rates.

        derived_dtg_predictions.csv: The calculated DTG curves from the predictions.

How to Run
1. Setup

a. Clone the repository:

git clone https://github.com/sienkus/tga.git
cd tga

b. Create a virtual environment (Recommended):

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

c. Install dependencies:
Make sure you have pip installed, then run:

pip install -r requirements.txt

2. Configuration

IMPORTANT: Before running the project, you must edit the config.py file.

Open config.py and update the EXCEL_FILE_PATH variable to the correct absolute or relative path of your TGA- empty fruit bunch - to Hassan (1).xlsx file.

# In config.py
EXCEL_FILE_PATH = '/path/to/your/TGA- empty fruit bunch - to Hassan (1).xlsx'

You can also adjust other settings in this file, such as the unseen heating rates or the number of bootstrap iterations.
3. Execute the Workflow

Once the setup and configuration are complete, run the main script from your terminal:

python main.py

This command will execute the entire pipeline:

    Load and preprocess the data.

    Train the Random Forest model with hyperparameter tuning.

    Save the best model and its parameters to the outputs/ directory.

    Load the saved model to perform extrapolation for the unseen heating rates.

    Save the final TGA and DTG predictions as CSV files in the outputs/ directory.

After the script finishes, you can find all the generated results in the outputs folder.