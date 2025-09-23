
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os, io, math, datetime, json

st.set_page_config(page_title="Total Hardness Predictor (Persistent v2.1)", page_icon="💧", layout="centered")
st.title("💧 Total Hardness Predictor — Persistent & Self-Improving (v2.1)")

st.write("""
Predict **Total Hardness (ppm as CaCO₃)** from **Conductivity (µS/cm)** and **pH**.
This version persists data/model and **handles legacy `.pkl` files** that contain only a raw estimator.
""")

# -----------------------------
# Constants & Paths
# -----------------------------
DATA_PATH = "data/hardness_data.csv"          # persistent dataset (features + target when available)
MODEL_PATH = "hardness_model.pkl"             # persisted model bundle (model + metadata)
FEATURES_SIDE_CAR = "feature_columns.json"    # optional sidecar for legacy models
os.makedirs("data", exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return math.sqrt(mean_squared_error(y_true, y_pred))

def now_iso():
    return datetime.datetime.utcnow().isoformat()

def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            pass
    cols = ["timestamp", "Conductivity (µS/cm)", "pH", "Total Hardness (ppm)", "source"]
    df = pd.DataFrame(columns=cols)
    df.to_csv(DATA_PATH, index=False)
    return df

def save_data(df: pd.DataFrame):
    df.to_csv(DATA_PATH, index=False)

def clean_training_dataframe(df: pd.DataFrame, use_ph: bool) -> pd.DataFrame:
    d = df.copy()
    for c in ["Conductivity (µS/cm)", "pH", "Total Hardness (ppm)"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    req_cols = ["Total Hardness (ppm)", "Conductivity (µS/cm)"]
    if use_ph:
        req_cols.append("pH")
    d = d.dropna(subset=req_cols)
    d = d[(d["Total Hardness (ppm)"] > 0) & (d["Conductivity (µS/cm)"] > 0)]
    if use_ph:
        d = d[(d["pH"] >= 0) & (d["pH"] <= 14)]
    return d

def package_model(model, feature_columns, metrics: dict):
    return {"model": model, "feature_columns": feature_columns, "metrics": metrics}

def _read_sidecar_features():
    if os.path.exists(FEATURES_SIDE_CAR):
        try:
            with open(FEATURES_SIDE_CAR, "r") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "feature_columns" in obj:
                return obj["feature_columns"]
        except Exception:
            pass
    return None

def load_model():
    """Load model; support dict-bundle or legacy raw estimator.
    Returns a dict bundle: {'model': estimator, 'feature_columns': [...], 'metrics': {...}}
    """
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        obj = joblib.load(MODEL_PATH)
    except Exception:
        return None

    # New format: dict bundle
    if isinstance(obj, dict) and "model" in obj:
        return obj

    # Legacy: raw estimator saved directly
    feat = _read_sidecar_features()
    if not feat:
        # sensible default for legacy case
        feat = ["Conductivity (µS/cm)"]
    return {"model": obj, "feature_columns": feat, "metrics": {}}

def save_model(bundle):
    with open(MODEL_PATH, "wb") as f:
        joblib.dump(bundle, f)
    # also write sidecar for portability
    try:
        with open(FEATURES_SIDE_CAR, "w") as f:
            json.dump({"feature_columns": bundle.get("feature_columns", [])}, f)
    except Exception:
        pass

def train_model(df_all: pd.DataFrame, use_ph: bool):
    d = clean_training_dataframe(df_all, use_ph=use_ph)
    if len(d) < 10:
        raise ValueError("Not enough labeled rows to train (need at least 10).")
    X_cols = ["Conductivity (µS/cm)"] + (["pH"] if use_ph else [])
    X = d[X_cols].values
    y = d["Total Hardness (ppm)"].values

    test_size = max(3, int(0.2 * len(d)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "Test_R2": float(r2_score(y_test, y_pred)),
        "Test_MAE": float(mean_absolute_error(y_test, y_pred)),
        "Test_RMSE": float(rmse(y_test, y_pred)),
        "n_trainable_rows": int(len(d)),
        "features": X_cols
    }
    return model, X_cols, metrics

def rf_std_prediction(model: RandomForestRegressor, X: np.ndarray):
    try:
        preds = np.stack([t.predict(X) for t in model.estimators_], axis=0)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0, ddof=1)
        return mean, std
    except Exception:
        return model.predict(X), np.full((X.shape[0],), np.nan)

# -----------------------------
# Sidebar: Model management
# -----------------------------
st.sidebar.header("Model Management")
use_ph_flag = st.sidebar.checkbox("Include pH in training", value=False, help="If enabled, the model uses both Conductivity and pH.")

bundle = load_model()
data_df = load_data()

if bundle is None:
    st.sidebar.warning("No packaged model found yet. Train a model below to create one.")
else:
    st.sidebar.success("Loaded model from disk.")
    st.sidebar.write("Features:", ", ".join(bundle.get("feature_columns", [])))
    if "metrics" in bundle and bundle["metrics"]:
        with st.sidebar.expander("Stored model metrics"):
            st.json(bundle["metrics"])

st.sidebar.markdown("---")
st.sidebar.subheader("Train / Retrain")
if st.sidebar.button("Train now from stored data"):
    try:
        model, feat_cols, metrics = train_model(data_df, use_ph=use_ph_flag)
        bundle = package_model(model, feat_cols, metrics)
        save_model(bundle)
        st.sidebar.success("Model trained and saved.")
    except Exception as e:
        st.sidebar.error(f"Training failed: {e}")

auto_retrain = st.sidebar.checkbox("Auto-retrain when new labeled samples are saved", value=True)

# -----------------------------
# Prediction UI
# -----------------------------
st.markdown("---")
st.subheader("🔢 Predict Total Hardness")

c = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=4500.0, step=10.0, help="Unit: microsiemens per centimeter")
ph = st.number_input("pH (0–14)", min_value=0.0, max_value=14.0, value=8.0, step=0.01)

colA, colB = st.columns(2)
with colA:
    predict_btn = st.button("Predict")
with colB:
    clear_btn = st.button("Clear inputs")

if clear_btn:
    st.experimental_rerun()

if predict_btn:
    if bundle is None:
        st.error("No model found. Please train a model from stored data first.")
    else:
        feat_cols = bundle["feature_columns"]
        X_vals = []
        for f in feat_cols:
            if f.lower().startswith("conductivity"):
                X_vals.append(c)
            elif f.lower() == "pH".lower():
                X_vals.append(ph)
        X_vec = np.array(X_vals, dtype=float).reshape(1, -1)
        model = bundle["model"]
        y_mean, y_std = rf_std_prediction(model, X_vec)
        y_pred = float(y_mean[0])
        st.success(f"Predicted Total Hardness: **{y_pred:,.0f} ppm as CaCO₃**")
        if not np.isnan(y_std[0]):
            lo = y_pred - 1.96 * float(y_std[0])
            hi = y_pred + 1.96 * float(y_std[0])
            st.caption(f"Approx. uncertainty band (informal): {lo:,.0f} – {hi:,.0f} ppm")

        # Append unlabeled prediction for future labeling
        df = load_data()
        row = {
            "timestamp": now_iso(),
            "Conductivity (µS/cm)": c,
            "pH": ph,
            "Total Hardness (ppm)": np.nan,
            "source": "prediction_only"
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        save_data(df)

# -----------------------------
# Labeling UI
# -----------------------------
st.markdown("---")
st.subheader("🧪 Add the actual lab result (optional)")
st.caption("When you receive the lab-measured Total Hardness, enter it below to improve the model.")

with st.form("label_form"):
    measured = st.number_input("Measured Total Hardness (ppm as CaCO₃)", min_value=0.0, value=0.0, step=1.0)
    label_note = st.text_input("Note (optional)", value="")
    label_submit = st.form_submit_button("Save labeled sample")

if label_submit:
    if measured <= 0:
        st.error("Measured value must be > 0.")
    else:
        df = load_data()
        row = {
            "timestamp": now_iso(),
            "Conductivity (µS/cm)": c,
            "pH": ph,
            "Total Hardness (ppm)": measured,
            "source": "labeled"
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        save_data(df)
        st.success("Labeled sample saved.")
        if auto_retrain:
            try:
                model, feat_cols, metrics = train_model(df, use_ph=use_ph_flag)
                bundle = package_model(model, feat_cols, metrics)
                save_model(bundle)
                st.info("Auto-retrained model saved.")
                with st.expander("New model metrics"):
                    st.json(metrics)
            except Exception as e:
                st.warning(f"Auto-retrain failed: {e}")

# -----------------------------
# Data Browser
# -----------------------------
st.markdown("---")
st.subheader("📁 Stored Dataset (tail)")
df_view = load_data()
st.dataframe(df_view.tail(50), use_container_width=True)

with st.expander("View entire dataset"):
    st.dataframe(df_view, use_container_width=True)

with st.expander("Where files are stored"):
    st.code(f"Dataset CSV: {DATA_PATH}\nModel: {MODEL_PATH}\nSidecar features: {FEATURES_SIDE_CAR}")
