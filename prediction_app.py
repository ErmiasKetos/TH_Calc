
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math
import io

st.set_page_config(page_title="Total Hardness Predictor", page_icon="💧", layout="centered")

st.title("💧 Total Hardness Predictor")
st.write("Predict **Total Hardness (ppm as CaCO₃)** from **Conductivity (µS/cm)** and **pH**.")

with st.expander("ℹ️ How it works / Notes"):
    st.markdown("""
    - The default model is a **Random Forest** trained on your dataset (conductivity-only performed best in the sample).
    - You can **upload a CSV** with columns like: `Total Hardness (ppm)`, `Conductivity (µS/cm)`, `pH` to train a new model inside the app.
    - pH typically adds little signal unless data spans a wide range; **conductivity** tends to be the dominant predictor.
    - This is an empirical model. For production/QA decisions, validate with your own data (e.g., time-based splits).
    """)

def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return math.sqrt(mean_squared_error(y_true, y_pred))

@st.cache_resource
def load_default_model():
    try:
        blob = open("hardness_model.pkl", "rb").read()
        obj = joblib.load(io.BytesIO(blob))
        return obj
    except Exception as e:
        return None

def train_from_csv(df, use_ph: bool):
    # Cast/clean
    cols = ['Total Hardness (ppm)', 'Conductivity (µS/cm)']
    if use_ph:
        cols.append('pH')
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    d = df.copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=cols).copy()
    d = d[(d['Total Hardness (ppm)'] > 0) & (d['Conductivity (µS/cm)'] > 0)]
    d = d[(~d['pH'].isna())] if use_ph else d
    d = d[(d['pH'] >= 0) & (d['pH'] <= 14)] if use_ph else d
    if len(d) < 10:
        raise ValueError("Not enough rows after cleaning (need at least 10).")

    X_cols = ['Conductivity (µS/cm)'] + (['pH'] if use_ph else [])
    X = d[X_cols].values
    y = d['Total Hardness (ppm)'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=max(3, int(0.2*len(y))), random_state=42
    )
    model = RandomForestRegressor(n_estimators=400, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "Test_R2": float(r2_score(y_test, y_pred)),
        "Test_MAE": float(mean_absolute_error(y_test, y_pred)),
        "Test_RMSE": float(rmse(y_test, y_pred)),
        "n": int(len(d)),
        "features": X_cols
    }
    return model, metrics

def rf_std_prediction(model: RandomForestRegressor, X: np.ndarray):
    # Approximate uncertainty: std across trees
    try:
        preds = np.stack([t.predict(X) for t in model.estimators_], axis=0)  # [n_trees, n_samples]
        mean = preds.mean(axis=0)
        std = preds.std(axis=0, ddof=1)
        return mean, std
    except Exception:
        return model.predict(X), np.full((X.shape[0],), np.nan)

st.sidebar.header("Model Source")
mode = st.sidebar.radio(
    "Choose how to get a model:",
    ["Use packaged model", "Upload CSV and train here"]
)

use_ph_flag = st.sidebar.checkbox("Include pH in training (when uploading CSV)", value=False)

model_obj = None
feature_cols = None
metrics_info = None

if mode == "Use packaged model":
    model_obj = load_default_model()
    if model_obj is None:
        st.error("Packaged model not found. Please switch to 'Upload CSV and train here'.")
    else:
        model = model_obj['model']
        feature_cols = model_obj['feature_columns']
        metrics_info = model_obj.get('metrics', {})
        st.sidebar.success("Loaded packaged model.")
        st.sidebar.write("Features:", ", ".join(feature_cols))
else:
    st.sidebar.info("Upload a CSV with columns including 'Total Hardness (ppm)', 'Conductivity (µS/cm)' and optionally 'pH'.")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            model, metrics_info = train_from_csv(df, use_ph=use_ph_flag)
            model_obj = {"model": model, "feature_columns": (['Conductivity (µS/cm)'] + (['pH'] if use_ph_flag else []))}
            feature_cols = model_obj['feature_columns']
            st.sidebar.success("Model trained from your CSV.")
            st.sidebar.write("Features:", ", ".join(feature_cols))
            st.sidebar.write("Metrics:", metrics_info)
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

st.markdown("---")
st.subheader("🔢 Make a prediction")

c = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=4500.0, step=10.0, help="Unit: microsiemens per centimeter")
ph = st.number_input("pH (0–14)", min_value=0.0, max_value=14.0, value=8.0, step=0.01)

predict_btn = st.button("Predict Total Hardness")

if predict_btn:
    if model_obj is None:
        st.error("No model is loaded. Load packaged model or train from CSV first.")
    else:
        # Align feature vector
        X_list = []
        for f in feature_cols:
            if f.lower().startswith("conductivity"):
                X_list.append([c])
            elif f.lower() == "pH".lower():
                X_list.append([ph])
        X_vec = np.array([v[0] for v in X_list], dtype=float).reshape(1, -1)

        model = model_obj['model']
        y_mean, y_std = rf_std_prediction(model, X_vec)
        y_pred = float(y_mean[0])
        st.success(f"Predicted Total Hardness: **{y_pred:,.0f} ppm as CaCO₃**")
        if not np.isnan(y_std[0]):
            lo = y_pred - 1.96 * float(y_std[0])
            hi = y_pred + 1.96 * float(y_std[0])
            st.caption(f"Approx. uncertainty band (not a formal CI): {lo:,.0f} – {hi:,.0f} ppm")

        if metrics_info:
            with st.expander("Model metrics"):
                st.json(metrics_info)

st.markdown("---")
st.subheader("📄 Data dictionary")
st.markdown("""
- **Total Hardness (ppm as CaCO₃)**: Target variable.
- **Conductivity (µS/cm)**: Primary predictor; higher conductivity often correlates with higher hardness depending on ions present.
- **pH (0–14)**: Optional predictor; limited direct correlation in many datasets.
""")

st.caption("Tip: For reliable results, periodically retrain with updated lab data and validate with a time-based split.")
