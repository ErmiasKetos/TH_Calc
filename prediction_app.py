
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os, io, math, json, datetime, re

st.set_page_config(page_title="Total & Calcium Hardness Predictor (v4.1)", page_icon="💧", layout="wide")
st.title("💧 Total & Calcium Hardness Predictor — v4.1")

# Canonical column names used everywhere
CANON = {'cond': 'Conductivity (µS/cm)', 'ph': 'pH', 'th': 'Total Hardness ppm', 'cah': 'Calculated Hardness (Calcium Hardness)'}

DATA_PATH = "data/hardness_data.csv"
MODEL_PATH = "hardness_dual_models.pkl"
os.makedirs("data", exist_ok=True)

def now_iso():
    return datetime.datetime.utcnow().isoformat()

def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return math.sqrt(mean_squared_error(y_true, y_pred))

def to_num(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return np.nan
    try:
        if isinstance(s, (int, float, np.number)):
            return float(s)
        x = str(s).strip()
        x = x.replace(",", "")  # remove thousand separators
        m = re.search(r"-?\d+(\.\d+)?", x)
        if not m:
            return np.nan
        return float(m.group(0))
    except Exception:
        return np.nan

def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            pass
    cols = ["timestamp", CANON["cond"], CANON["ph"], CANON["th"], CANON["cah"], "source"]
    df = pd.DataFrame(columns=cols)
    df.to_csv(DATA_PATH, index=False)
    return df

def save_data(df: pd.DataFrame):
    df.to_csv(DATA_PATH, index=False)

def clean(df: pd.DataFrame, need_th: bool, need_cah: bool) -> pd.DataFrame:
    d = df.copy()
    for c in [CANON["cond"], CANON["ph"], CANON["th"], CANON["cah"]]:
        if c in d.columns:
            d[c] = d[c].map(to_num)
    req = [CANON["cond"], CANON["ph"]]
    if need_th: req.append(CANON["th"])
    if need_cah: req.append(CANON["cah"])
    d = d.dropna(subset=req)
    d = d[(d[CANON["cond"]]>0) & (d[CANON["ph"]]>=0) & (d[CANON["ph"]]<=14)]
    if need_th: d = d[d[CANON["th"]]>0]
    if need_cah: d = d[d[CANON["cah"]]>0]
    return d

def train_dual_models(df_all: pd.DataFrame):
    d_th = clean(df_all, need_th=True, need_cah=False)
    d_cah = clean(df_all, need_th=False, need_cah=True)

    metrics = {}
    models = {}

    if len(d_th) >= 20:
        X = d_th[[CANON["cond"], CANON["ph"]]].values
        y = d_th[CANON["th"]].values
        test_size = max(20, int(0.2*len(y)))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
        m_th = RandomForestRegressor(n_estimators=700, random_state=42)
        m_th.fit(Xtr, ytr)
        yhat = m_th.predict(Xte)
        metrics["total_hardness"] = {"Test_R2": float(r2_score(yte, yhat)),
                                      "Test_MAE": float(mean_absolute_error(yte, yhat)),
                                      "Test_RMSE": float(rmse(yte, yhat)),
                                      "n_test": int(len(yte)), "n_trainable_rows": int(len(d_th))}
        m_th.fit(X, y)
        models["total_hardness"] = {"model": m_th, "target": CANON["th"]}
    else:
        metrics["total_hardness"] = {"warning": f"Not enough labeled rows (have {len(d_th)}; need ≥20)"}

    if len(d_cah) >= 20:
        X = d_cah[[CANON["cond"], CANON["ph"]]].values
        y = d_cah[CANON["cah"]].values
        test_size = max(20, int(0.2*len(y)))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
        m_cah = RandomForestRegressor(n_estimators=700, random_state=42)
        m_cah.fit(Xtr, ytr)
        yhat = m_cah.predict(Xte)
        metrics["calcium_hardness"] = {"Test_R2": float(r2_score(yte, yhat)),
                                        "Test_MAE": float(mean_absolute_error(yte, yhat)),
                                        "Test_RMSE": float(rmse(yte, yhat)),
                                        "n_test": int(len(yte)), "n_trainable_rows": int(len(d_cah))}
        m_cah.fit(X, y)
        models["calcium_hardness"] = {"model": m_cah, "target": CANON["cah"]}
    else:
        metrics["calcium_hardness"] = {"warning": f"Not enough labeled rows (have {len(d_cah)}; need ≥20)"}

    return {"models": models, "feature_columns": [CANON["cond"], CANON["ph"]], "metrics": metrics}

def load_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

def save_bundle(bundle):
    joblib.dump(bundle, MODEL_PATH)

def predict(bundle, cond, ph):
    out = {}
    X = np.array([[cond, ph]], dtype=float)
    if "calcium_hardness" in bundle["models"]:
        cah = float(bundle["models"]["calcium_hardness"]["model"].predict(X)[0])
        out["Calcium Hardness (ppm as CaCO3)"] = cah
        out["Calcium (ppm) = Ca Hardness / 2.5"] = cah / 2.5
    if "total_hardness" in bundle["models"]:
        th = float(bundle["models"]["total_hardness"]["model"].predict(X)[0])
        out["Total Hardness (ppm as CaCO3)"] = th
    return out

# ---- Sidebar: model controls ----
st.sidebar.header("Model & Data")
bundle = load_bundle()
data_df = load_data()

if bundle is None:
    st.sidebar.warning("No trained model yet. Import data and train.")
else:
    st.sidebar.success("Model bundle loaded.")
    with st.sidebar.expander("Current metrics"):
        st.json(bundle.get("metrics", {}))

if st.sidebar.button("Train / Retrain now"):
    new_bundle = train_dual_models(data_df)
    save_bundle(new_bundle)
    st.sidebar.success("Model(s) trained and saved.")
    bundle = new_bundle

auto_retrain = st.sidebar.checkbox("Auto-retrain when saving labeled rows", value=True)

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔢 Predict")
    cond = st.number_input(f'{CANON["cond"]}', min_value=0.0, value=4500.0, step=10.0)
    ph = st.number_input("pH (0–14)", min_value=0.0, max_value=14.0, value=8.0, step=0.01)
    if st.button("Predict now"):
        if bundle is None or not bundle.get("models"):
            st.error("No trained model available. Train a model first.")
        else:
            res = predict(bundle, cond, ph)
            if not res:
                st.error("Bundle has no trained targets. Train again after importing labeled data.")
            else:
                st.success("Predictions")
                st.json(res)
            # Log unlabeled
            df = load_data()
            row = {"timestamp": now_iso(), CANON["cond"]: cond, CANON["ph"]: ph,
                   CANON["th"]: np.nan, CANON["cah"]: np.nan, "source": "prediction_only"}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_data(df)

with col2:
    st.subheader("🧪 Save Labeled Row")
    th_val = st.number_input('Measured Total Hardness (ppm as CaCO3)', min_value=0.0, value=0.0, step=1.0)
    cah_val = st.number_input('Measured Calcium Hardness (ppm as CaCO3)', min_value=0.0, value=0.0, step=1.0)
    if st.button("Save labeled row"):
        if th_val <= 0 and cah_val <= 0:
            st.error("Enter at least one measured value (>0).")
        else:
            df = load_data()
            row = {"timestamp": now_iso(), CANON["cond"]: cond, CANON["ph"]: ph,
                    CANON["th"]: (th_val if th_val>0 else np.nan),
                    CANON["cah"]: (cah_val if cah_val>0 else np.nan),
                    "source": "labeled"}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_data(df)
            st.success("Labeled row saved.")
            if auto_retrain:
                new_bundle = train_dual_models(df)
                save_bundle(new_bundle)
                st.info("Auto-retrained.")
                with st.expander("New metrics"):
                    st.json(new_bundle.get("metrics", {}))

st.markdown("---")
st.subheader("📥 Import Dataset (Map Columns)")
up = st.file_uploader("Upload CSV", type=["csv"])
if up is not None:
    try:
        dfu = pd.read_csv(up)
        dfu.columns = [c.strip() for c in dfu.columns]
        st.write("Uploaded columns:", list(dfu.columns))

        # column mapping UI
        def guess(substrs, fallback=None):
            for c in dfu.columns:
                cl = c.lower()
                if any(s in cl for s in substrs):
                    return c
            return fallback or dfu.columns[0]

        m_cond = st.selectbox(f"Map to {CANON['cond']}", dfu.columns, index=dfu.columns.get_loc(guess(['conduct','µs'])))
        m_ph   = st.selectbox(f"Map to {CANON['ph']}", dfu.columns, index=dfu.columns.get_loc(guess(['ph'])))
        m_th   = st.selectbox(f"Map to {CANON['th']} (optional)", dfu.columns, index=min(2, len(dfu.columns)-1))
        m_cah  = st.selectbox(f"Map to {CANON['cah']} (optional)", dfu.columns, index=min(3, len(dfu.columns)-1))

        # Preview mapped columns with cleaning
        prev = pd.DataFrame({
            CANON["cond"]: dfu[m_cond].map(to_num),
            CANON["ph"]: dfu[m_ph].map(to_num),
            CANON["th"]: dfu[m_th].map(to_num) if m_th else np.nan,
            CANON["cah"]: dfu[m_cah].map(to_num) if m_cah else np.nan,
        }).head(10)
        st.write("Preview (cleaned numeric):")
        st.dataframe(prev)

        if st.button("Import rows"):
            d = pd.DataFrame({
                "timestamp": now_iso(),
                CANON["cond"]: dfu[m_cond].map(to_num),
                CANON["ph"]: dfu[m_ph].map(to_num),
                CANON["th"]: dfu[m_th].map(to_num) if m_th else np.nan,
                CANON["cah"]: dfu[m_cah].map(to_num) if m_cah else np.nan,
                "source": "import"
            })
            base = load_data()
            before = len(base)
            merged = pd.concat([base, d], ignore_index=True)
            save_data(merged)
            added = len(merged) - before
            n_labeled = int((d[CANON["th"]].notna() | d[CANON["cah"]].notna()).sum())
            st.success(f"Imported {added} rows ({n_labeled} labeled).")
    except Exception as e:
        st.error(f"Import failed: {e}")

st.markdown("---")
st.subheader("📊 Data Browser (tail)")
st.dataframe(load_data().tail(50), use_container_width=True)
st.caption("Derived Calcium (ppm) = Predicted Calcium Hardness (as CaCO₃) ÷ 2.5")

