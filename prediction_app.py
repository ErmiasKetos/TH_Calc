
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os, io, math, json, datetime

st.set_page_config(page_title="Total & Calcium Hardness Predictor (v4)", page_icon="💧", layout="wide")
st.title("💧 Total & Calcium Hardness Predictor — v4 (Persistent, Dual-Target)")

# ---- Paths ----
DATA_PATH = "data/hardness_data.csv"
MODEL_PATH = "hardness_dual_models.pkl"
os.makedirs("data", exist_ok=True)

# ---- Helpers ----
def now_iso():
    return datetime.datetime.utcnow().isoformat()

def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return math.sqrt(mean_squared_error(y_true, y_pred))

def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            pass
    # Initialize with expected columns
    cols = ["timestamp","Conductivity","pH","Total Hardness ppm","Calculated Hardness (Calcium Hardness)","source"]
    df = pd.DataFrame(columns=cols)
    df.to_csv(DATA_PATH, index=False)
    return df

def save_data(df: pd.DataFrame):
    df.to_csv(DATA_PATH, index=False)

def clean(df: pd.DataFrame, need_total: bool, need_ca: bool) -> pd.DataFrame:
    d = df.copy()
    # ensure numerics
    for c in ["Conductivity","pH","Total Hardness ppm","Calculated Hardness (Calcium Hardness)"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    req = ["Conductivity","pH"]
    if need_total:
        req.append("Total Hardness ppm")
    if need_ca:
        req.append("Calculated Hardness (Calcium Hardness)")
    d = d.dropna(subset=req)
    d = d[(d["Conductivity"]>0) & (d["pH"]>=0) & (d["pH"]<=14)]
    if need_total: d = d[d["Total Hardness ppm"]>0]
    if need_ca: d = d[d["Calculated Hardness (Calcium Hardness)"]>0]
    return d

def train_dual_models(df_all: pd.DataFrame):
    d_total = clean(df_all, need_total=True, need_ca=False)
    d_ca = clean(df_all, need_total=False, need_ca=True)

    metrics = {}
    models = {}

    # total hardness model
    if len(d_total) >= 20:
        X = d_total[["Conductivity","pH"]].values
        y = d_total["Total Hardness ppm"].values
        test_size = max(20, int(0.2*len(y)))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
        m_total = RandomForestRegressor(n_estimators=700, random_state=42)
        m_total.fit(Xtr, ytr)
        yhat = m_total.predict(Xte)
        metrics["total_hardness"] = {
            "Test_R2": float(r2_score(yte, yhat)),
            "Test_MAE": float(mean_absolute_error(yte, yhat)),
            "Test_RMSE": float(rmse(yte, yhat)),
            "n_test": int(len(yte)),
            "n_trainable_rows": int(len(d_total))
        }
        m_total.fit(X, y)  # fit on all for deployment
        models["total_hardness"] = {"model": m_total, "target": "Total Hardness ppm"}
    else:
        metrics["total_hardness"] = {"warning": f"Not enough labeled rows to train Total Hardness (have {len(d_total)}, need ≥20)."}

    # calcium hardness model
    if len(d_ca) >= 20:
        X = d_ca[["Conductivity","pH"]].values
        y = d_ca["Calculated Hardness (Calcium Hardness)"].values
        test_size = max(20, int(0.2*len(y)))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
        m_ca = RandomForestRegressor(n_estimators=700, random_state=42)
        m_ca.fit(Xtr, ytr)
        yhat = m_ca.predict(Xte)
        metrics["calcium_hardness"] = {
            "Test_R2": float(r2_score(yte, yhat)),
            "Test_MAE": float(mean_absolute_error(yte, yhat)),
            "Test_RMSE": float(rmse(yte, yhat)),
            "n_test": int(len(yte)),
            "n_trainable_rows": int(len(d_ca))
        }
        m_ca.fit(X, y)
        models["calcium_hardness"] = {"model": m_ca, "target": "Calculated Hardness (Calcium Hardness)"}
    else:
        metrics["calcium_hardness"] = {"warning": f"Not enough labeled rows to train Calcium Hardness (have {len(d_ca)}, need ≥20)."}

    bundle = {"models": models, "feature_columns": ["Conductivity","pH"], "metrics": metrics}
    return bundle

def load_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

def save_bundle(bundle):
    joblib.dump(bundle, MODEL_PATH)

def predict_with_bundle(bundle, conductivity: float, ph: float):
    res = {}
    X = np.array([[conductivity, ph]], dtype=float)
    if "total_hardness" in bundle["models"]:
        m = bundle["models"]["total_hardness"]["model"]
        th = float(m.predict(X)[0])
        res["Total Hardness (ppm as CaCO3)"] = th
    if "calcium_hardness" in bundle["models"]:
        m = bundle["models"]["calcium_hardness"]["model"]
        cah = float(m.predict(X)[0])
        res["Calcium Hardness (ppm as CaCO3)"] = cah
        res["Calcium (ppm) (derived = Ca hardness / 2.5)"] = cah / 2.5
    return res

# ---- Sidebar controls ----
st.sidebar.header("Model & Data")
bundle = load_bundle()
data_df = load_data()

if bundle is None:
    st.sidebar.warning("No trained model yet. Import data and train.")
else:
    st.sidebar.success("Loaded model bundle.")
    with st.sidebar.expander("Model metrics"):
        st.json(bundle.get("metrics", {}))

if st.sidebar.button("Train / Retrain now"):
    new_bundle = train_dual_models(data_df)
    save_bundle(new_bundle)
    st.sidebar.success("Model(s) trained and saved.")
    bundle = new_bundle

auto_retrain = st.sidebar.checkbox("Auto-retrain when saving labeled rows", value=True)

# ---- Main UI: Predict / Label ----
colL, colR = st.columns(2)

with colL:
    st.subheader("🔢 Predict")
    c = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=4500.0, step=10.0)
    ph = st.number_input("pH (0–14)", min_value=0.0, max_value=14.0, value=8.0, step=0.01)
    if st.button("Predict now"):
        if bundle is None or not bundle.get("models"):
            st.error("No model found. Train a model first.")
        else:
            results = predict_with_bundle(bundle, c, ph)
            if not results:
                st.error("Model bundle contains no trained targets. Train again after importing labeled data.")
            else:
                st.success("Prediction results")
                st.json(results)
            # log unlabeled
            df = load_data()
            row = {"timestamp": now_iso(),"Conductivity": c,"pH": ph,
                   "Total Hardness ppm": np.nan,
                   "Calculated Hardness (Calcium Hardness)": np.nan,
                   "source":"prediction_only"}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_data(df)

with colR:
    st.subheader("🧪 Save Labeled Row (when lab results arrive)")
    th_meas = st.number_input("Measured Total Hardness (ppm as CaCO3)", min_value=0.0, value=0.0, step=1.0)
    cah_meas = st.number_input("Measured Calcium Hardness (ppm as CaCO3)", min_value=0.0, value=0.0, step=1.0)
    if st.button("Save labeled row"):
        if th_meas <= 0 and cah_meas <= 0:
            st.error("Enter at least one measured value (>0).")
        else:
            df = load_data()
            row = {"timestamp": now_iso(),"Conductivity": c,"pH": ph,
                   "Total Hardness ppm": (th_meas if th_meas>0 else np.nan),
                   "Calculated Hardness (Calcium Hardness)": (cah_meas if cah_meas>0 else np.nan),
                   "source": "labeled"}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_data(df)
            st.success("Labeled row saved.")
            if auto_retrain:
                new_bundle = train_dual_models(df)
                save_bundle(new_bundle)
                st.info("Auto-retrained and saved.")
                with st.expander("New model metrics"):
                    st.json(new_bundle.get("metrics", {}))

st.markdown("---")
st.subheader("📥 Import Dataset")
st.caption("Upload a CSV with columns (any names): Conductivity, pH, Total Hardness ppm, Calculated Hardness (Calcium Hardness). Map them below.")
up = st.file_uploader("Upload CSV", type=["csv"])
if up is not None:
    try:
        dfu = pd.read_csv(up)
        st.write("Uploaded columns:", list(dfu.columns))
        # try to guess defaults
        def guess(name, fallback_idx=0):
            for cand in dfu.columns:
                if name.lower() in cand.lower():
                    return cand
            return dfu.columns[min(fallback_idx, len(dfu.columns)-1)]
        col_cond = st.selectbox("Map Conductivity column", dfu.columns, index=dfu.columns.get_loc(guess("conduct")))
        col_ph = st.selectbox("Map pH column", dfu.columns, index=dfu.columns.get_loc(guess("pH",1)))
        col_th = st.selectbox("Map Total Hardness column", dfu.columns, index=dfu.columns.get_loc(guess("total hardness",2)))
        col_cah = st.selectbox("Map Calcium Hardness column", dfu.columns, index=dfu.columns.get_loc(guess("calcium hardness",3)))

        if st.button("Import rows"):
            d = pd.DataFrame({
                "timestamp": now_iso(),
                "Conductivity": pd.to_numeric(dfu[col_cond], errors="coerce"),
                "pH": pd.to_numeric(dfu[col_ph], errors="coerce"),
                "Total Hardness ppm": pd.to_numeric(dfu[col_th], errors="coerce"),
                "Calculated Hardness (Calcium Hardness)": pd.to_numeric(dfu[col_cah], errors="coerce"),
                "source": "import"
            })
            base = load_data()
            merged = pd.concat([base, d], ignore_index=True)
            save_data(merged)
            st.success(f"Imported {len(d)} rows into persistent store.")
    except Exception as e:
        st.error(f"Import failed: {e}")

st.markdown("---")
st.subheader("📊 Data Browser (tail)")
st.dataframe(load_data().tail(50), use_container_width=True)

st.caption("Note: Derived Calcium (ppm) = Predicted Calcium Hardness (as CaCO₃) ÷ 2.5")
