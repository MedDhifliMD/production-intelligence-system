import pandas as pd
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
from preprocessing import clean_and_feature_engineer, aggregate_to_boards

def train_and_save_pipeline(csv_path, output_dir='models'):
    """
    Full pipeline to load data, preprocess it, train models, and save artifacts.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print("Preprocessing data...")
    df_clean = clean_and_feature_engineer(df)
    df_boards, scaler = aggregate_to_boards(df_clean)

    # Save scaler for use in the app
    joblib.dump(scaler, os.path.join(output_dir, 'coordinate_scaler.pkl'))

    # Prepare features
    features = [
        'Total_Components_Placed', 'Unique_Feeders_Used', 
        'Avg_Coordinate_X', 'Avg_Coordinate_Y', 
        'NPM_Hour', 'NPM_DayOfWeek', 'Delay_to_Verif_Mins'
    ]
    X = df_boards[features]
    y = df_boards['Board_Failed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, os.path.join(output_dir, 'model_xgboost.pkl'))

    # Train Neural Network
    print("Training Neural Network...")
    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    nn_model.fit(X_train, y_train)
    joblib.dump(nn_model, os.path.join(output_dir, 'baseline_neural_network.pkl'))

    # Evaluation
    for name, model in [("XGBoost", xgb_model), ("Neural Network", nn_model)]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        print(f"\n--- {name} Evaluation ---")
        print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
        if len(y_test.unique()) > 1:
            print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")

if __name__ == "__main__":
    # Adjust this path to your actual local dataset path
    DATA_PATH = r'C:\Users\dhifl\Desktop\B4CSolutions\Training\production-intelligence-system\final_dataset.csv' 
    print(f"Data file found at {DATA_PATH}")
    if os.path.exists(DATA_PATH):
        train_and_save_pipeline(DATA_PATH)
    else:
        print(f"Data file not found at {DATA_PATH}. Please place your CSV file in the project root.")
