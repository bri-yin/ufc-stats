import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
def load_data():
    """Load and initial preprocessing of UFC data"""
    # Load fighter master data
    fighters_df = pd.read_csv('data/ufc_fighters_master.csv')
    
    # Load fights data
    fights_df = pd.read_csv('data/ufc_fights_all.csv')
    
    return fighters_df, fights_df

def preprocess_fighter_data(fighters_df):
    """Clean and preprocess fighter statistics"""
    # Clean column names (remove any extra spaces)
    fighters_df.columns = fighters_df.columns.str.strip()
    
    # Handle missing values
    numeric_cols = ['height_in', 'reach_in', 'weight_lbs', 'slpm', 'str_acc', 
                   'sapm', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg']
    
    for col in numeric_cols:
        if col in fighters_df.columns:
            fighters_df[col] = pd.to_numeric(fighters_df[col], errors='coerce')
            fighters_df[col] = fighters_df[col].fillna(fighters_df[col].median())
    
    # Process date of birth to calculate age
    if 'dob' in fighters_df.columns:
        fighters_df['dob'] = pd.to_datetime(fighters_df['dob'], errors='coerce')
        current_date = datetime.now()
        fighters_df['age'] = (current_date - fighters_df['dob']).dt.days / 365.25
        fighters_df['age'] = fighters_df['age'].fillna(fighters_df['age'].median())
    
    # Encode stance
    if 'stance' in fighters_df.columns:
        le_stance = LabelEncoder()
        fighters_df['stance_encoded'] = le_stance.fit_transform(fighters_df['stance'].fillna('Orthodox'))
    
    # Create derived features
    if 'height_in' in fighters_df.columns and 'reach_in' in fighters_df.columns:
        fighters_df['reach_height_ratio'] = fighters_df['reach_in'] / fighters_df['height_in']
        fighters_df['reach_height_ratio'] = fighters_df['reach_height_ratio'].fillna(1.0)
    
    return fighters_df

def preprocess_fights_data(fights_df):
    """Clean and preprocess fights data"""
    print(f"Original fights data shape: {fights_df.shape}")
    print(f"Original columns: {list(fights_df.columns)}")
    
    # Assume column structure based on the sample data provided
    expected_cols = ['event', 'event_url', 'fighter_1', 'fighter_2', 'result', 
                    'method', 'round', 'time', 'fight_url']
    
    if len(fights_df.columns) >= len(expected_cols):
        fights_df.columns = expected_cols[:len(fights_df.columns)]
    
    print(f"After renaming columns: {list(fights_df.columns)}")
    
    # Clean result column
    fights_df['result'] = fights_df['result'].astype(str).str.strip().str.lower()
    print(f"Unique results: {fights_df['result'].value_counts()}")
    
    # Since red fighter (fighter_1) always wins in this dataset,
    # we'll create balanced data by randomly swapping fighter positions
    print("Creating balanced dataset by randomly swapping red/blue positions...")
    
    # Create a copy for manipulation
    balanced_fights = []
    
    for idx, row in fights_df.iterrows():
        # Original fight (red wins)
        original_fight = row.copy()
        original_fight['fighter_1_wins'] = 1
        balanced_fights.append(original_fight)
        
        # Swapped fight (blue wins) - swap fighter positions
        swapped_fight = row.copy()
        swapped_fight['fighter_1'] = row['fighter_2']  # Blue fighter now in red corner
        swapped_fight['fighter_2'] = row['fighter_1']  # Red fighter now in blue corner
        swapped_fight['fighter_1_wins'] = 0  # New fighter_1 (originally blue) loses
        balanced_fights.append(swapped_fight)
    
    # Create balanced dataframe
    fights_df = pd.DataFrame(balanced_fights)
    
    print(f"Balanced dataset shape: {fights_df.shape}")
    print(f"Fighter 1 wins distribution: {fights_df['fighter_1_wins'].value_counts()}")
    
    # Extract numeric round information
    if 'round' in fights_df.columns:
        fights_df['round_num'] = pd.to_numeric(fights_df['round'], errors='coerce')
        fights_df['round_num'] = fights_df['round_num'].fillna(3)
    
    return fights_df

def create_fight_features(fights_df, fighters_df):
    """Merge fight data with fighter stats and create comparative features"""
    print(f"Fights before merge: {len(fights_df)}")
    print(f"Fighters data: {len(fighters_df)}")

    # Merge fighter 1 stats
    fighter1_stats = fighters_df.add_suffix('_f1')
    fights_merged = fights_df.merge(
        fighter1_stats,
        left_on='fighter_1',
        right_on='name_f1',
        how='left'
    )
    print(f"After fighter 1 merge: {len(fights_merged)}")
    print(f"Missing fighter 1 data: {fights_merged['name_f1'].isna().sum()}")

    # Merge fighter 2 stats
    fighter2_stats = fighters_df.add_suffix('_f2')
    fights_merged = fights_merged.merge(
        fighter2_stats,
        left_on='fighter_2',
        right_on='name_f2',
        how='left'
    )
    print(f"After fighter 2 merge: {len(fights_merged)}")
    print(f"Missing fighter 2 data: {fights_merged['name_f2'].isna().sum()}")

    # Create comparative features (Fighter 1 - Fighter 2)
    stat_columns = [
        'height_in', 'reach_in', 'weight_lbs', 'age', 'slpm',
        'str_acc', 'sapm', 'str_def', 'td_avg', 'td_acc',
        'td_def', 'sub_avg', 'stance_encoded', 'reach_height_ratio'
    ]
    for stat in stat_columns:
        c1, c2 = f'{stat}_f1', f'{stat}_f2'
        if c1 in fights_merged.columns and c2 in fights_merged.columns:
            fights_merged[f'{stat}_diff'] = fights_merged[c1] - fights_merged[c2]

    # Create ratio features for key stats
    ratio_stats = ['slpm', 'str_acc', 'str_def', 'td_acc', 'td_def']
    for stat in ratio_stats:
        c1, c2 = f'{stat}_f1', f'{stat}_f2'
        if c1 in fights_merged.columns and c2 in fights_merged.columns:
            f2_stat = fights_merged[c2].replace(0, 0.001)
            fights_merged[f'{stat}_ratio'] = fights_merged[c1] / f2_stat

    print(f"Final merged data shape: {fights_merged.shape}")
    return fights_merged

def prepare_features_and_target(fights_merged):
    """Prepare final feature set and target variable"""
    
    # Select feature columns
    feature_cols = [col for col in fights_merged.columns if 
                   ('_diff' in col or '_ratio' in col or col == 'round_num')]
    
    # Add individual fighter stats that might be important
    individual_stats = ['height_in_f1', 'reach_in_f1', 'weight_lbs_f1', 'age_f1',
                       'slpm_f1', 'str_acc_f1', 'str_def_f1', 'td_acc_f1', 'td_def_f1',
                       'height_in_f2', 'reach_in_f2', 'weight_lbs_f2', 'age_f2',
                       'slpm_f2', 'str_acc_f2', 'str_def_f2', 'td_acc_f2', 'td_def_f2']
    
    available_individual_stats = [col for col in individual_stats if col in fights_merged.columns]
    feature_cols.extend(available_individual_stats)
    
    # Remove duplicates and ensure all columns exist
    feature_cols = list(set([col for col in feature_cols if col in fights_merged.columns]))
    
    X = fights_merged[feature_cols].copy()
    y = fights_merged['fighter_1_wins'].copy()
    
    # Handle missing values in features
    X = X.fillna(X.median())
    
    # Remove any rows where target is missing
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    return X, y, feature_cols

def train_xgboost_model(X, y):
    """Train XGBoost model with hyperparameter tuning"""
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Scale features (XGBoost doesn't always need this, but it can help)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initial XGBoost model
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss'
    )
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, 
                              scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Train final model
    best_model.fit(X_train_scaled, y_train)
    
    return best_model, scaler, X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(model, scaler, X_test, y_test, feature_cols, show_plots=False):
    """Evaluate model performance"""
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix (text only)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")
    
    # Feature importance (text only)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Feature Importances:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:20s} {row['importance']:.4f}")
    
    # Optional plots (only if requested)
    if show_plots:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    return accuracy, feature_importance

def predict_fight_outcome(model, scaler, fighter1_stats, fighter2_stats, feature_cols):
    """Predict outcome of a specific fight"""
    
    # Create comparison features
    fight_features = {}
    
    # Add round number (default to 3)
    fight_features['round_num'] = 3
    
    # Create difference and ratio features
    stat_columns = ['height_in', 'reach_in', 'weight_lbs', 'age', 'slpm', 
                   'str_acc', 'sapm', 'str_def', 'td_avg', 'td_acc', 
                   'td_def', 'sub_avg', 'stance_encoded', 'reach_height_ratio']
    
    for stat in stat_columns:
        if stat in fighter1_stats and stat in fighter2_stats:
            fight_features[f'{stat}_diff'] = fighter1_stats[stat] - fighter2_stats[stat]
    
    ratio_stats = ['slpm', 'str_acc', 'str_def', 'td_acc', 'td_def']
    for stat in ratio_stats:
        if stat in fighter1_stats and stat in fighter2_stats and fighter2_stats[stat] != 0:
            fight_features[f'{stat}_ratio'] = fighter1_stats[stat] / fighter2_stats[stat]
    
    # Add individual stats
    for stat in stat_columns:
        if stat in fighter1_stats:
            fight_features[f'{stat}_f1'] = fighter1_stats[stat]
        if stat in fighter2_stats:
            fight_features[f'{stat}_f2'] = fighter2_stats[stat]
    
    # Create DataFrame with same structure as training data
    fight_df = pd.DataFrame([fight_features])
    
    # Ensure all required features are present
    for col in feature_cols:
        if col not in fight_df.columns:
            fight_df[col] = 0  # Default value for missing features
    
    fight_df = fight_df[feature_cols]  # Ensure same column order
    
    # Scale features
    fight_scaled = scaler.transform(fight_df)
    
    # Predict
    prediction = model.predict(fight_scaled)[0]
    probability = model.predict_proba(fight_scaled)[0]
    
    return prediction, probability

def main():
    """Main execution function"""
    print("Loading UFC fight data...")
    
    try:
        # Option 1: Files in same directory
        fighters_df, fights_df = load_data()
        
        # Option 2: Custom file paths (uncomment if needed)
        # fighters_df, fights_df = load_data(
        #     fighters_path='/path/to/your/ufc_fights_master.csv',
        #     fights_path='/path/to/your/ufc_fights_all.csv'
        # )
        
        print("Preprocessing data...")
        fighters_df = preprocess_fighter_data(fighters_df)
        fights_df = preprocess_fights_data(fights_df)
        
        print("Creating fight features...")
        fights_merged = create_fight_features(fights_df, fighters_df)
        
        print("Preparing features and target...")
        X, y, feature_cols = prepare_features_and_target(fights_merged)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of features: {len(feature_cols)}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        print("\nTraining XGBoost model...")
        model, scaler, X_train, X_test, y_train, y_test = train_xgboost_model(X, y)
        
        print("\nEvaluating model...")
        accuracy, feature_importance = evaluate_model(model, scaler, X_test, y_test, feature_cols, show_plots=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Example prediction
        print("\n" + "="*50)
        print("EXAMPLE PREDICTION")
        print("="*50)
        
        # Example fighter stats (you would replace these with actual fighter data)
        example_fighter1 = {
            'height_in': 72, 'reach_in': 74, 'weight_lbs': 185, 'age': 28,
            'slpm': 4.5, 'str_acc': 0.52, 'sapm': 3.2, 'str_def': 0.58,
            'td_avg': 2.1, 'td_acc': 0.45, 'td_def': 0.75, 'sub_avg': 0.5,
            'stance_encoded': 0, 'reach_height_ratio': 1.03
        }
        
        example_fighter2 = {
            'height_in': 70, 'reach_in': 72, 'weight_lbs': 185, 'age': 30,
            'slpm': 3.8, 'str_acc': 0.48, 'sapm': 4.1, 'str_def': 0.52,
            'td_avg': 1.5, 'td_acc': 0.38, 'td_def': 0.68, 'sub_avg': 0.3,
            'stance_encoded': 0, 'reach_height_ratio': 1.03
        }
        
        prediction, probability = predict_fight_outcome(model, scaler, example_fighter1, 
                                                      example_fighter2, feature_cols)
        
        winner = "Fighter 1" if prediction == 1 else "Fighter 2"
        confidence = max(probability)
        
        print(f"Predicted Winner: {winner}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Fighter 1 win probability: {probability[1]:.3f}")
        print(f"Fighter 2 win probability: {probability[0]:.3f}")
        
        return model, scaler, feature_cols, accuracy
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the CSV files are in the correct format and location.")
        return None, None, None, None

if __name__ == "__main__":
    model, scaler, feature_cols, accuracy = main()
    if model is not None:
        import joblib
        joblib.dump(model, "model_xgb.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(feature_cols, "feature_cols.pkl")
        print("Saved model_xgb.pkl, scaler.pkl, and feature_cols.pkl")