import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Import your functions from your module if you split files
# from your_module import preprocess_fighter_data, predict_fight_outcome

from datetime import datetime

@st.cache_data
def load_and_prep():
    fighters_df = pd.read_csv('data/ufc_fighters_master.csv')
    # minimal preprocess to match your training
    fighters_df.columns = fighters_df.columns.str.strip()
    numeric_cols = ['height_in','reach_in','weight_lbs','slpm','str_acc','sapm','str_def','td_avg','td_acc','td_def','sub_avg']
    for c in numeric_cols:
        if c in fighters_df.columns:
            fighters_df[c] = pd.to_numeric(fighters_df[c], errors='coerce')
            fighters_df[c] = fighters_df[c].fillna(fighters_df[c].median())
    if 'dob' in fighters_df.columns:
        fighters_df['dob'] = pd.to_datetime(fighters_df['dob'], errors='coerce')
        current_date = datetime.now()
        fighters_df['age'] = (current_date - fighters_df['dob']).dt.days / 365.25
        fighters_df['age'] = fighters_df['age'].fillna(fighters_df['age'].median())
    if 'stance' in fighters_df.columns:
        # encode stance the same way you did in training
        # If you saved the encoder, load and use it. Otherwise map common stances:
        stance_map = {s:i for i,s in enumerate(sorted(fighters_df['stance'].fillna('Orthodox').unique()))}
        fighters_df['stance_encoded'] = fighters_df['stance'].fillna('Orthodox').map(stance_map)
    if {'height_in','reach_in'}.issubset(fighters_df.columns):
        fighters_df['reach_height_ratio'] = fighters_df['reach_in'] / fighters_df['height_in']
        fighters_df['reach_height_ratio'] = fighters_df['reach_height_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return fighters_df

@st.cache_resource
def load_model_bundle():
    # Load what you saved after training
    model = joblib.load('model_xgb.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    return model, scaler, feature_cols

def row_to_stats(row):
    needed = [
        'height_in','reach_in','weight_lbs','age','slpm','str_acc','sapm','str_def',
        'td_avg','td_acc','td_def','sub_avg','stance_encoded','reach_height_ratio'
    ]
    d = {}
    for k in needed:
        d[k] = float(row.get(k, 0.0))
    if d['height_in'] and not d.get('reach_height_ratio'):
        d['reach_height_ratio'] = d['reach_in'] / d['height_in']
    return d

def predict_pair(f1_row, f2_row, model, scaler, feature_cols):
    f1 = row_to_stats(f1_row)
    f2 = row_to_stats(f2_row)

    # Recreate the same feature build your predict_fight_outcome uses
    # You can import and call your existing function instead of duplicating
    from train_xgboost import predict_fight_outcome  # or inline it if simpler
    pred, proba = predict_fight_outcome(model, scaler, f1, f2, feature_cols)
    return pred, proba

st.title("UFC Fight Predictor")

fighters_df = load_and_prep()
model, scaler, feature_cols = load_model_bundle()

names = fighters_df['name'].dropna().sort_values().unique().tolist()

col1, col2 = st.columns(2)
with col1:
    f1_name = st.selectbox("Fighter 1", names, index=names.index("Jon Jones") if "Jon Jones" in names else 0)
with col2:
    f2_name = st.selectbox("Fighter 2", names, index=names.index("Alex Pereira") if "Alex Pereira" in names else 1)

if f1_name == f2_name:
    st.warning("Pick two different fighters.")
else:
    f1_row = fighters_df.loc[fighters_df['name'] == f1_name].iloc[0]
    f2_row = fighters_df.loc[fighters_df['name'] == f2_name].iloc[0]
    if st.button("Predict"):
        pred, proba = predict_pair(f1_row, f2_row, model, scaler, feature_cols)
        winner = f1_name if pred == 1 else f2_name
        st.subheader(f"Predicted winner: {winner}")
        st.write(f"{f1_name} win probability: {proba[1]:.3f}")
        st.write(f"{f2_name} win probability: {proba[0]:.3f}")

        # Show a few key stat diffs
        show = ['slpm','str_acc','sapm','str_def','td_avg','td_acc','td_def','reach_height_ratio','age','height_in','reach_in']
        diffs = []
        for k in show:
            v1 = float(f1_row.get(k, np.nan))
            v2 = float(f2_row.get(k, np.nan))
            diffs.append((k, v1, v2, v1 - v2, (v1 / v2) if v2 not in [0, np.nan] else np.nan))
        st.dataframe(pd.DataFrame(diffs, columns=['metric','fighter1','fighter2','diff','ratio']))
