# UFC Fight Predictor

This project uses historical UFC fight and fighter statistics to train a machine learning model (XGBoost) that predicts the outcome of future matchups. A Streamlit web app provides an interface for comparing fighters and visualizing predictions.

---

## Features
- Cleans and preprocesses UFC fighter and fight data
- Balances training data so fighter position does not bias results
- Engineers comparative features such as reach difference, striking ratios, and takedown accuracy
- Trains an XGBoost classifier with hyperparameter tuning
- Evaluates the model and reports accuracy, precision, recall, and feature importance
- Provides a Streamlit app to input fighters and display win probability predictions

---

## Project Structure
ufc-stats/
├── data/ # Raw fighter and fight CSVs
├── models/ # Saved model, scaler, and feature columns
├── src/
│ ├── train_xgboost.py # Training script
│ └── ufc_app.py # Streamlit app
└── README.md

yaml
Copy code

---

## Installation

1. Clone the repository:
   ```bash
   git clone <your_repo_url>
   cd ufc-stats
Create and activate a Python environment (example with pyenv + venv):

bash
Copy code
pyenv install 3.12.6
pyenv local 3.12.6
python -m venv venv
source venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Training the Model
To train and save the model, scaler, and feature columns:

bash
Copy code
python src/train_xgboost.py
Artifacts will be saved to:

models/model_xgb.pkl

models/scaler.pkl

models/feature_cols.pkl

Running the App
To launch the Streamlit app:

bash
Copy code
streamlit run src/ufc_app.py
This will start a local server (default: http://localhost:8501) where you can compare fighters and view win probability predictions.

Data
The project expects the following CSV files in the data/ directory:

ufc_fighters_master.csv: fighter statistics

ufc_fights_all.csv: fight outcomes

The preprocessing functions will clean and normalize these files for use with the model.

Notes
The model currently uses striking, grappling, reach, and age statistics as features.

Performance can be improved by adding fight history, weight class, and opponent strength features.

Predictions are probabilistic and not guaranteed. This is a data science project, not a betting tool.
