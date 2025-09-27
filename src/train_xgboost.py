# src/train_xgboost.py
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Data loading
# ----------------------------
def load_data():
    fighters_path = DATA_DIR / "ufc_fighters_master.csv"
    fights_path = DATA_DIR / "ufc_fights_all.csv"
    if not fighters_path.exists() or not fights_path.exists():
        raise FileNotFoundError(f"Expected CSVs at:\n{fighters_path}\n{fights_path}")
    fighters_df = pd.read_csv(fighters_path)
    fights_df = pd.read_csv(fights_path)
    return fighters_df, fights_df

# ----------------------------
# Fighter preprocessing
# ----------------------------
def preprocess_fighter_data(fighters_df: pd.DataFrame) -> pd.DataFrame:
    df = fighters_df.copy()
    df.columns = df.columns.str.strip()

    numeric_cols = [
        "height_in","reach_in","weight_lbs","slpm","str_acc",
        "sapm","str_def","td_avg","td_acc","td_def","sub_avg"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
            df[c] = df[c].fillna(df[c].median())

    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        current_date = datetime.now()
        df["age"] = (current_date - df["dob"]).dt.days / 365.25
        df["age"] = df["age"].fillna(df["age"].median())
    else:
        if "age" not in df.columns:
            df["age"] = 30.0
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(30.0)

    if "stance" in df.columns:
        le_stance = LabelEncoder()
        df["stance_encoded"] = le_stance.fit_transform(df["stance"].fillna("Orthodox"))
    else:
        df["stance_encoded"] = 0

    if "height_in" in df.columns and "reach_in" in df.columns:
        df["reach_height_ratio"] = df["reach_in"] / df["height_in"]
        df["reach_height_ratio"] = df["reach_height_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    else:
        df["reach_height_ratio"] = 1.0

    return df

# ----------------------------
# Fights preprocessing
# ----------------------------
def preprocess_fights_data(fights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns expected to include at least:
      - fighter_red, fighter_blue, result, round
    Keeps only rows where result == 'win'.
    Mirrors each fight into two rows:
      - red as fighter_1 (label 1)
      - blue as fighter_1 (label 0)
    Adds round_num and fight_id.
    """
    df = fights_df.copy()
    need_cols = ["fighter_red", "fighter_blue", "result"]
    for c in need_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    df["result"] = df["result"].astype(str).str.strip().str.lower()
    df_win = df[df["result"].eq("win")].copy()

    # Drop any invalid names
    df_win = df_win.dropna(subset=["fighter_red","fighter_blue"])

    # Winner perspective
    r1 = pd.DataFrame({
        "fighter_1": df_win["fighter_red"].values,
        "fighter_2": df_win["fighter_blue"].values,
        "fighter_1_wins": 1,
        "round": df_win.get("round", pd.Series([np.nan]*len(df_win))).values
    })

    # Loser perspective
    r2 = pd.DataFrame({
        "fighter_1": df_win["fighter_blue"].values,
        "fighter_2": df_win["fighter_red"].values,
        "fighter_1_wins": 0,
        "round": df_win.get("round", pd.Series([np.nan]*len(df_win))).values
    })

    fights_bal = pd.concat([r1, r2], ignore_index=True)

    # Numeric round with default 3
    fights_bal["round_num"] = pd.to_numeric(fights_bal["round"], errors="coerce").fillna(3).astype(float)

    # Optional stable id per original bout for grouped splits
    df_win = df_win.reset_index(drop=False).rename(columns={"index": "orig_fight_id"})
    r1_ids = df_win["orig_fight_id"].values
    r2_ids = df_win["orig_fight_id"].values
    fights_bal["fight_id"] = np.concatenate([r1_ids, r2_ids])

    return fights_bal

# ----------------------------
# Merge fighter stats and engineer features
# ----------------------------
def create_fight_features(fights_df: pd.DataFrame, fighters_df: pd.DataFrame) -> pd.DataFrame:
    # Merge fighter 1
    f1 = fighters_df.add_suffix("_f1")
    merged = fights_df.merge(f1, left_on="fighter_1", right_on="name_f1", how="left")

    # Merge fighter 2
    f2 = fighters_df.add_suffix("_f2")
    merged = merged.merge(f2, left_on="fighter_2", right_on="name_f2", how="left")

    # Differences
    stat_columns = [
        "height_in","reach_in","weight_lbs","age","slpm",
        "str_acc","sapm","str_def","td_avg","td_acc","td_def","sub_avg",
        "stance_encoded","reach_height_ratio"
    ]
    for s in stat_columns:
        c1, c2 = f"{s}_f1", f"{s}_f2"
        if c1 in merged.columns and c2 in merged.columns:
            merged[f"{s}_diff"] = merged[c1] - merged[c2]

    # Ratios
    ratio_stats = ["slpm","str_acc","str_def","td_acc","td_def"]
    for s in ratio_stats:
        c1, c2 = f"{s}_f1", f"{s}_f2"
        if c1 in merged.columns and c2 in merged.columns:
            denom = merged[c2].replace(0, 0.001)
            merged[f"{s}_ratio"] = merged[c1] / denom

    # Drop rows that failed to merge on both sides for key stats
    key_pairs = [("slpm_f1","slpm_f2"), ("str_acc_f1","str_acc_f2"), ("td_acc_f1","td_acc_f2")]
    mask_any = None
    for a, b in key_pairs:
        if a in merged.columns and b in merged.columns:
            cur = merged[a].notna() & merged[b].notna()
            mask_any = cur if mask_any is None else (mask_any | cur)
    if mask_any is None:
        mask_any = pd.Series([True] * len(merged), index=merged.index)
    merged = merged[mask_any].reset_index(drop=True)

    return merged

# ----------------------------
# Build X, y, feature list, and groups
# ----------------------------
def prepare_features_and_target(fights_merged: pd.DataFrame):
    feature_cols = [
        c for c in fights_merged.columns
        if c.endswith("_diff") or c.endswith("_ratio") or c == "round_num"
    ]

    extra = [
        "height_in_f1","reach_in_f1","weight_lbs_f1","age_f1",
        "slpm_f1","str_acc_f1","str_def_f1","td_acc_f1","td_def_f1",
        "height_in_f2","reach_in_f2","weight_lbs_f2","age_f2",
        "slpm_f2","str_acc_f2","str_def_f2","td_acc_f2","td_def_f2"
    ]
    feature_cols.extend([c for c in extra if c in fights_merged.columns])
    feature_cols = list(dict.fromkeys([c for c in feature_cols if c in fights_merged.columns]))

    X = fights_merged[feature_cols].copy()
    y = fights_merged["fighter_1_wins"].copy()
    groups = fights_merged["fight_id"] if "fight_id" in fights_merged.columns else pd.Series(range(len(fights_merged)))

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    valid = ~y.isna()
    X, y, groups = X[valid], y[valid], groups[valid]
    return X, y, feature_cols, groups

# ----------------------------
# Train with grouped split and grouped CV
# ----------------------------
def train_xgboost_model(X, y, groups):
    # Grouped train test split keeps mirrored pairs together
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = xgb.XGBClassifier(random_state=42, eval_metric="logloss")

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    cv = GroupKFold(n_splits=5)
    gs = GridSearchCV(
        model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
    )
    gs.fit(X_train_s, y_train, groups=groups_train)
    best = gs.best_estimator_
    best.fit(X_train_s, y_train)

    return best, scaler, X_train_s, X_test_s, y_train, y_test

# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(model, X_test_s, y_test, feature_cols, show_plots=False):
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")

    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 10 Feature Importances:")
    for i, (_, row) in enumerate(importances.head(10).iterrows(), start=1):
        print(f"{i:2d}. {row['feature']:20s} {row['importance']:.4f}")

    return acc, importances

# ----------------------------
# Inference for a single matchup
# ----------------------------
def predict_fight_outcome(model, scaler, fighter1_stats: dict, fighter2_stats: dict, feature_cols: list):
    fight_features = {}
    fight_features["round_num"] = 3

    stat_columns = [
        "height_in","reach_in","weight_lbs","age","slpm",
        "str_acc","sapm","str_def","td_avg","td_acc","td_def","sub_avg",
        "stance_encoded","reach_height_ratio"
    ]

    # Differences
    for s in stat_columns:
        if s in fighter1_stats and s in fighter2_stats:
            fight_features[f"{s}_diff"] = float(fighter1_stats[s]) - float(fighter2_stats[s])

    # Ratios with safe denom
    for s in ["slpm","str_acc","str_def","td_acc","td_def"]:
        if s in fighter1_stats and s in fighter2_stats:
            denom = float(fighter2_stats[s]) if float(fighter2_stats[s]) != 0 else 0.001
            fight_features[f"{s}_ratio"] = float(fighter1_stats[s]) / denom

    # Absolutes
    for s in stat_columns:
        if s in fighter1_stats:
            fight_features[f"{s}_f1"] = float(fighter1_stats[s])
        if s in fighter2_stats:
            fight_features[f"{s}_f2"] = float(fighter2_stats[s])

    df = pd.DataFrame([fight_features])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_cols]

    Xs = scaler.transform(df)
    pred = model.predict(Xs)[0]
    proba = model.predict_proba(Xs)[0]
    return pred, proba

# ----------------------------
# Main
# ----------------------------
def main():
    print("Loading UFC fight data...")
    fighters_df, fights_df = load_data()

    print("Preprocessing data...")
    fighters_df = preprocess_fighter_data(fighters_df)
    fights_df = preprocess_fights_data(fights_df)

    print("Creating fight features...")
    fights_merged = create_fight_features(fights_df, fighters_df)

    print("Preparing features and target...")
    X, y, feature_cols, groups = prepare_features_and_target(fights_merged)

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    print("\nTraining XGBoost model...")
    model, scaler, X_train_s, X_test_s, y_train, y_test = train_xgboost_model(X, y, groups)

    print("\nEvaluating model...")
    accuracy, feat_imp = evaluate_model(model, X_test_s, y_test, feature_cols, show_plots=False)

    # Example prediction with placeholder stats
    example_f1 = {
        "height_in": 72, "reach_in": 74, "weight_lbs": 185, "age": 28,
        "slpm": 4.5, "str_acc": 0.52, "sapm": 3.2, "str_def": 0.58,
        "td_avg": 2.1, "td_acc": 0.45, "td_def": 0.75, "sub_avg": 0.5,
        "stance_encoded": 0, "reach_height_ratio": 1.03
    }
    example_f2 = {
        "height_in": 70, "reach_in": 72, "weight_lbs": 185, "age": 30,
        "slpm": 3.8, "str_acc": 0.48, "sapm": 4.1, "str_def": 0.52,
        "td_avg": 1.5, "td_acc": 0.38, "td_def": 0.68, "sub_avg": 0.3,
        "stance_encoded": 0, "reach_height_ratio": 1.03
    }
    pred, proba = predict_fight_outcome(model, scaler, example_f1, example_f2, feature_cols)
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    print(f"Predicted Winner: {'Fighter 1' if pred == 1 else 'Fighter 2'}")
    print(f"Fighter 1 win probability: {proba[1]:.3f}")
    print(f"Fighter 2 win probability: {proba[0]:.3f}")

    # Save artifacts
    joblib.dump(model, MODELS_DIR / "model_xgb.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
    print(f"\nSaved artifacts to: {MODELS_DIR.resolve()}")

    return model, scaler, feature_cols, accuracy

if __name__ == "__main__":
    main()
