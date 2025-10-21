
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os, io, math, datetime, json

st.set_page_config(page_title="Total Hardness Predictor (v3)", page_icon="💧", layout="wide")
st.title("💧 Total Hardness Predictor — v3 (Persistent, Import, Column Mapper)")

DATA_PATH = "data/hardness_data.csv"
MODEL_PATH = "hardness_model.pkl"
FEATURES_SIDE_CAR = "feature_columns.json"
os.makedirs("data", exist_ok=True)

def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return math.sqrt(mean_squared_error(y_true, y_pred))

def now_iso():
    import datetime
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
    req = ["Total Hardness (ppm)", "Conductivity (µS/cm)"]
    if use_ph:
        req.append("pH")
    d = d.dropna(subset=req)
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
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        obj = joblib.load(MODEL_PATH)
    except Exception:
        return None
    if isinstance(obj, dict) and "model" in obj:
        return obj
    feat = _read_sidecar_features() or ["Conductivity (µS/cm)"]
    return {"model": obj, "feature_columns": feat, "metrics": {}}

def save_model(bundle):
    with open(MODEL_PATH, "wb") as f:
        joblib.dump(bundle, f)
    try:
        with open(FEATURES_SIDE_CAR, "w") as f:
            json.dump({"feature_columns": bundle.get("feature_columns", [])}, f)
    except Exception:
        pass

def train_model(df_all: pd.DataFrame, use_ph: bool):
    from sklearn.ensemble import RandomForestRegressor
    d = clean_training_dataframe(df_all, use_ph=use_ph)
    if len(d) < 10:
        raise ValueError("Not enough labeled rows to train (need at least 10).")
    X_cols = ["Conductivity (µS/cm)"] + (["pH"] if use_ph else [])
    X = d[X_cols].values
    y = d["Total Hardness (ppm)"].values
    test_size = max(3, int(0.2*len(d)))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
    model = RandomForestRegressor(n_estimators=600, random_state=42)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    metrics = {
        "Test_R2": float(r2_score(yte, yhat)),
        "Test_MAE": float(mean_absolute_error(yte, yhat)),
        "Test_RMSE": float(rmse(yte, yhat)),
        "n_trainable_rows": int(len(d)),
        "features": X_cols
    }
    # Fit on all for deployment
    model.fit(X, y)
    return model, X_cols, metrics

def rf_std_prediction(model, X: np.ndarray):
    try:
        preds = np.stack([t.predict(X) for t in model.estimators_], axis=0)
        return preds.mean(axis=0), preds.std(axis=0, ddof=1)
    except Exception:
        return model.predict(X), np.full((X.shape[0],), np.nan)

# Sidebar
st.sidebar.header("Model & Data")
use_ph_flag = st.sidebar.checkbox("Include pH in training", value=True)
bundle = load_model()
data_df = load_data()

if bundle is None:
    st.sidebar.warning("No model yet. Train after importing or adding labeled data.")
else:
    st.sidebar.success("Loaded model from disk.")
    st.sidebar.write("Features:", ", ".join(bundle.get("feature_columns", [])))
    if bundle.get("metrics"):
        with st.sidebar.expander("Stored model metrics"):
            st.json(bundle["metrics"])

if st.sidebar.button("Train / Retrain now"):
    try:
        model, feat_cols, metrics = train_model(data_df, use_ph=use_ph_flag)
        bundle = package_model(model, feat_cols, metrics)
        save_model(bundle)
        st.sidebar.success("Model trained and saved.")
    except Exception as e:
        st.sidebar.error(f"Training failed: {e}")

auto_retrain = st.sidebar.checkbox("Auto-retrain after saving a labeled sample", value=True)

st.markdown("---")
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("🔢 Predict")
    c = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=4500.0, step=10.0)
    ph = st.number_input("pH (0–14)", min_value=0.0, max_value=14.0, value=8.0, step=0.01)
    if st.button("Predict Total Hardness"):
        if bundle is None:
            st.error("No model found. Train a model first.")
        else:
            feat_cols = bundle["feature_columns"]
            X_vals = []
            for f in feat_cols:
                if f.lower().startswith("conductivity"):
                    X_vals.append(c)
                elif f.lower() == "pH".lower():
                    X_vals.append(ph)
            X = np.array(X_vals, dtype=float).reshape(1, -1)
            model = bundle["model"]
            y_mean, y_std = rf_std_prediction(model, X)
            y_pred = float(y_mean[0])
            st.success(f"Predicted Total Hardness: **{y_pred:,.0f} ppm as CaCO₃**")
            if not np.isnan(y_std[0]):
                lo = y_pred - 1.96 * float(y_std[0]); hi = y_pred + 1.96 * float(y_std[0])
                st.caption(f"Approx. uncertainty band: {lo:,.0f} – {hi:,.0f} ppm")
            # log unlabeled
            df = load_data()
            row = {"timestamp": now_iso(), "Conductivity (µS/cm)": c, "pH": ph, "Total Hardness (ppm)": np.nan, "source": "prediction_only"}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_data(df)

with col2:
    st.subheader("🧪 Add Measured Result")
    measured = st.number_input("Measured Total Hardness (ppm as CaCO₃)", min_value=0.0, value=0.0, step=1.0)
    if st.button("Save labeled sample"):
        if measured <= 0:
            st.error("Measured value must be > 0.")
        else:
            df = load_data()
            row = {"timestamp": now_iso(), "Conductivity (µS/cm)": c, "pH": ph, "Total Hardness (ppm)": measured, "source": "labeled"}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_data(df)
            st.success("Labeled sample saved.")
            if auto_retrain:
                try:
                    model, feat_cols, metrics = train_model(df, use_ph=use_ph_flag)
                    bundle = package_model(model, feat_cols, metrics)
                    save_model(bundle)
                    st.info("Auto-retrained and saved.")
                    with st.expander("New model metrics"):
                        st.json(metrics)
                except Exception as e:
                    st.warning(f"Auto-retrain failed: {e}")

st.markdown("---")
st.subheader("📥 Import Dataset & Map Columns")
st.caption("Upload a CSV and map its columns to required fields. Imported rows with target become training data; others are stored for future labeling.")
up = st.file_uploader("Upload CSV", type=["csv"])
if up is not None:
    try:
        dfu = pd.read_csv(up)
        st.write("Uploaded columns:", list(dfu.columns))
        # Column mapping UI
        col_cond = st.selectbox("Map conductivity column", dfu.columns, index=0)
        col_ph = st.selectbox("Map pH column (optional)", dfu.columns, index=min(1, len(dfu.columns)-1))
        col_th = st.selectbox("Map Total Hardness column (target, optional)", dfu.columns, index=min(2, len(dfu.columns)-1))
        if st.button("Import rows"):
            # Build standardized frame
            d = pd.DataFrame({
                "timestamp": now_iso(),
                "Conductivity (µS/cm)": pd.to_numeric(dfu[col_cond], errors="coerce"),
                "pH": pd.to_numeric(dfu[col_ph], errors="coerce"),
                "Total Hardness (ppm)": pd.to_numeric(dfu[col_th], errors="coerce")
            })
            # Determine source label
            d["source"] = np.where(d["Total Hardness (ppm)"].notna(), "import_labeled", "import_unlabeled")
            base = load_data()
            merged = pd.concat([base, d], ignore_index=True)
            save_data(merged)
            st.success(f"Imported {len(d)} rows into persistent store.")
    except Exception as e:
        st.error(f"Import failed: {e}")

st.markdown("---")
st.subheader("📊 Data Browser (tail)")
st.dataframe(load_data().tail(50), use_container_width=True)
