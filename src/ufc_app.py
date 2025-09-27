# src/ufc_app.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime

# Import from your training script
ROOT = Path(__file__).resolve().parents[1]   # project root
DATA_DIR = ROOT / "data"

from train_xgboost import (
    ROOT, DATA_DIR,
    preprocess_fighter_data,
)

MODELS_DIR = ROOT / "models"


@st.cache_data
def load_and_prep_fighters():
    fighters_path = DATA_DIR / "ufc_fighters_master.csv"
    if not fighters_path.exists():
        st.error(f"Missing fighters CSV at {fighters_path}")
        st.stop()

    fighters_df = pd.read_csv(fighters_path)
    fighters_df = preprocess_fighter_data(fighters_df)
    if "name" not in fighters_df.columns:
        st.error("fighters CSV must have a 'name' column")
        st.stop()
    return fighters_df


@st.cache_resource
def load_model_bundle():
    model_path = MODELS_DIR / "model_xgb.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feat_path = MODELS_DIR / "feature_cols.pkl"

    if not model_path.exists() or not scaler_path.exists() or not feat_path.exists():
        st.error(f"Missing model files in {MODELS_DIR}. Train and save them first.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(feat_path)
    return model, scaler, feature_cols


def row_to_stats(row: pd.Series) -> dict:
    needed = [
        "height_in","reach_in","weight_lbs","age","slpm","str_acc","sapm","str_def",
        "td_avg","td_acc","td_def","sub_avg","stance_encoded","reach_height_ratio"
    ]
    d = {}
    for k in needed:
        d[k] = float(row.get(k, 0.0)) if pd.notna(row.get(k, np.nan)) else 0.0
    if d.get("height_in", 0) and not d.get("reach_height_ratio", 0):
        d["reach_height_ratio"] = d["reach_in"] / d["height_in"]
    return d


def build_feature_row_from_stats(f1: dict, f2: dict, feature_cols: list) -> pd.DataFrame:
    """
    Recreate the exact feature row used for training and prediction.
    Returns a 1xN DataFrame with columns in feature_cols order.
    """
    fight_features = {}
    fight_features["round_num"] = 3  # default planned rounds

    stat_columns = [
        "height_in","reach_in","weight_lbs","age","slpm",
        "str_acc","sapm","str_def","td_avg","td_acc","td_def","sub_avg",
        "stance_encoded","reach_height_ratio"
    ]

    # diffs
    for stat in stat_columns:
        if stat in f1 and stat in f2:
            fight_features[f"{stat}_diff"] = f1[stat] - f2[stat]

    # ratios
    for stat in ["slpm","str_acc","str_def","td_acc","td_def"]:
        if stat in f1 and stat in f2:
            denom = f2[stat] if f2[stat] != 0 else 0.001
            fight_features[f"{stat}_ratio"] = f1[stat] / denom

    # absolutes
    for stat in stat_columns:
        if stat in f1:
            fight_features[f"{stat}_f1"] = f1[stat]
        if stat in f2:
            fight_features[f"{stat}_f2"] = f2[stat]

    df = pd.DataFrame([fight_features])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_cols]
    return df


def main():
    st.title("UFC Fight Predictor")

    fighters_df = load_and_prep_fighters()
    model, scaler, feature_cols = load_model_bundle()

    names = fighters_df["name"].dropna().sort_values().unique().tolist()
    col1, col2 = st.columns(2)
    with col1:
        f1_name = st.selectbox("Fighter 1", names, index=names.index("Jon Jones") if "Jon Jones" in names else 0)
    with col2:
        f2_name = st.selectbox("Fighter 2", names, index=names.index("Alex Pereira") if "Alex Pereira" in names else 1)

    if f1_name == f2_name:
        st.warning("Pick two different fighters.")
        return

    f1_row = fighters_df.loc[fighters_df["name"] == f1_name].iloc[0]
    f2_row = fighters_df.loc[fighters_df["name"] == f2_name].iloc[0]

    if st.button("Predict"):
        # Build per-fighter dicts
        f1 = row_to_stats(f1_row)
        f2 = row_to_stats(f2_row)

        # Build the model feature row and scale it
        X_row = build_feature_row_from_stats(f1, f2, feature_cols)
        Xs_row = scaler.transform(X_row)

        # Predict
        proba = model.predict_proba(Xs_row)[0]
        pred = int(proba[1] >= 0.5)
        winner = f1_name if pred == 1 else f2_name

        st.subheader(f"Predicted winner: {winner}")
        st.write(f"{f1_name} win probability: {proba[1]:.3f}")
        st.write(f"{f2_name} win probability: {proba[0]:.3f}")

        # Show key stat comparisons
        show = [
            "slpm","str_acc","sapm","str_def","td_avg","td_acc","td_def",
            "reach_height_ratio","age","height_in","reach_in","weight_lbs"
        ]
        rows = []
        for k in show:
            v1 = float(f1_row.get(k, np.nan)) if pd.notna(f1_row.get(k, np.nan)) else np.nan
            v2 = float(f2_row.get(k, np.nan)) if pd.notna(f2_row.get(k, np.nan)) else np.nan
            ratio = (v1 / v2) if (isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v2 not in [0, np.nan]) else np.nan
            rows.append((k, v1, v2, v1 - v2 if pd.notna(v1) and pd.notna(v2) else np.nan, ratio))

        st.caption("Key stat comparisons")
        st.dataframe(pd.DataFrame(rows, columns=["metric","fighter1","fighter2","diff","ratio"]), use_container_width=True)

        # SHAP breakdown
        st.markdown("### Why this prediction? SHAP breakdown")

        try:
            import shap

            # TreeExplainer works with XGBoost models. SHAP is on the model output margin (log-odds) by default.
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Xs_row)  # shape (1, n_features)
            phi = shap_values[0]                         # 1D array
            feature_names = feature_cols

            # Base value (log-odds). Convert to base probability for reference.
            base_logit = explainer.expected_value
            base_prob = 1 / (1 + np.exp(-base_logit))

            # Final logit and prob
            final_logit = base_logit + phi.sum()
            final_prob = 1 / (1 + np.exp(-final_logit))

            # Approximate probability impact for each feature around final prob
            # d(sigmoid)/dz = p*(1-p). Use final_prob for a local linearization.
            slope = final_prob * (1 - final_prob)
            approx_delta_prob = phi * slope

            # Assemble a dataframe of contributions
            contrib = pd.DataFrame({
                "feature": feature_names,
                "shap_logit": phi,                 # contribution in log-odds space
                "approx_delta_prob": approx_delta_prob  # approx contribution in probability
            })
            contrib["abs_delta"] = contrib["approx_delta_prob"].abs()
            contrib = contrib.sort_values("abs_delta", ascending=False)

            top_k = 12
            st.write(f"Base win probability for Fighter 1 (before features): **{base_prob:.3f}**")
            st.write(f"Model win probability for Fighter 1 (after features): **{final_prob:.3f}**")

            st.caption(f"Top {top_k} feature contributions (positive helps Fighter 1)")
            st.dataframe(
                contrib.head(top_k)[["feature","approx_delta_prob","shap_logit"]]
                .rename(columns={
                    "feature":"feature",
                    "approx_delta_prob":"approx Δ win prob",
                    "shap_logit":"log-odds contribution"
                }),
                use_container_width=True
            )

            # Simple bar chart for the same top features
            chart_df = contrib.head(top_k)[["feature","approx_delta_prob"]].set_index("feature")
            st.bar_chart(chart_df)

            with st.expander("Details about SHAP math"):
                st.write(
                    "Tree SHAP explains the model output in log-odds space. "
                    "We show an approximate probability change by multiplying each log-odds SHAP value by p*(1-p), "
                    "where p is the final predicted probability for Fighter 1. "
                    "This is a local linearization and not an exact decomposition in probability space."
                )

        except ImportError:
            st.warning("Install SHAP to view per-fight explanations: `pip install shap`")
        except Exception as e:
            st.error(f"Could not compute SHAP values: {e}")


if __name__ == "__main__":
    # If someone runs this file with plain python, give a friendly hint
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            import sys
            sys.exit("Run this app with:  streamlit run src/ufc_app.py")
    except Exception:
        pass
    main()
