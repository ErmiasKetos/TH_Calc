import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = 'water_quality_data.csv'
MODELS_FILE = 'trained_models.pkl'
CONFIG_FILE = 'best_configs.json'

# ============================================================
# COMPLETE EMBEDDED DATA - All 37 water testing records
# ============================================================
INITIAL_DATA = {
    'Date': [
        '2/1/25', '9/1/25', '14-01-25', '20-01-25', '4/2/25', '13-02-25', '19-02-25', '26-02-25',
        '6/3/25', '14-03-25', '20-03-25', '28-03-25', '3/4/25', '8/4/25', '16-04-25', '25-04-25',
        '30-04-25', '9/5/25', '15-05-25', '23-05-25', '30-05-25', '6/6/25', '11/6/25', '18-06-25',
        '23-06-25', '4/7/25', '9/7/25', '14-07-25', '23-07-25', '31-07-25', '8/8/25', '21-08-25',
        '28-08-25', '4/9/25', '11/9/25', '16-09-25', '22-09-25'
    ],
    'Calculated_Hardness_ppm': [
        1740, 1710, 1720, 1670, 1490, 1620, 1450, 1470, 1610, 1700, 1630, 1680, 1070, 1190, 1300,
        1320, 1390, 1490, 1500, 1490, 1000, 1380, 1480, 1100, 1130, 1050, 1050, 1030, 1090, 1060,
        1110, 1070, 1140, 910, 950, 1010, 1100
    ],
    'Calcium_ppm': [
        696, 684, 688, 668, 596, 648, 580, 588, 644, 680, 652, 672, 428, 476, 520, 528, 556, 596,
        600, 596, 400, 552, 592, 440, 452, 420, 420, 412, 436, 424, 444, 428, 456, 364, 380, 404, 440
    ],
    'Total_Hardness_ppm': [
        2090, 2000, 2010, 2100, 1990, 1900, 169, 1930, 1900, 1900, 1970, 1980, 1480, 1480, 1640,
        1680, 1650, 1760, 1910, 1990, 1670, 1750, 1920, 1400, 1590, 1390, 1460, 1430, 1430, 1400,
        1500, 1430, 1480, 1270, 1190, 1390, 1390
    ],
    'Total_Alkalinity_ppm': [
        120, 58, 50, 98, 36, 66, 56, 62, 38, 36, 24, 20, 88, 66, 94, 56, 48, 56, 66, 54, 90, 70,
        72, 86, 78, 86, 78, 84, 46, 70, 74, 58, 75, 62, 43, 68, 52
    ],
    'Conductivity_uS_cm': [
        4930, 5070, 4980, 5010, 4740, 4740, 4390, 4750, 4790, 4860, 5090, 5240, 3740, 4070, 4470,
        4490, 4710, 4520, 4490, 4450, 4250, 4330, 5190, 4210, 4520, 4230, 4311, 4320, 4400, 4340,
        4349, 4360, 5490, 4240, 4250, 4640, 4920
    ],
    'pH': [
        7.55, 7.6, 8.1, 8.1, 7.96, 7.92, 7.91, 7.81, 7.68, 8.07, 7.75, 7.77, 8, 8.09, 8.23, 7.86,
        7.5, 8.1, 7.93, 7.89, 7.96, 7.96, 7.87, 7.93, 7.74, 8.14, 7.95, 8.16, 8.24, 8.76, 7.79,
        8, 8, 7.68, 8.12, 8.22, 7.97
    ],
    'Sulfate_ppm': [
        600, 593.33, 516.67, 580, 600, 500, 416.67, 500, 600, 513.33, 540, 500, 350, 416.67,
        473.33, 453.33, 373.33, 400, 500, 473.33, 516.67, 400, 750, 416.67, 400, 433.33, 403.33,
        500, 500, 383.33, 406.67, 350, 366.67, 346.67, 346.67, 366.67, 403.33
    ],
    'Iron_ppm': [
        0.2, 0.16, 0.24, 0.15, 0.18, 0.25, 0.08, 0.41, 0.24, 0.28, 0.32, 0.34, 0.36, 0.38, 0.37,
        0.28, 0.36, 0.26, 0.17, 0.18, 0.29, 0.24, 0.1, 0.22, 0.15, 0.25, 0.19, 0.24, 0.24, 0.34,
        0.3, 0.06, 0.14, 0.26, 0.18, 0.16, 0.17
    ]
}

# Algorithm descriptions
ALGORITHM_INFO = {
    'Random Forest': {
        'description': 'Ensemble of decision trees. Great for non-linear relationships.',
        'pros': 'Handles complex patterns, resistant to overfitting, shows feature importance',
        'cons': 'Can be slow with large data, less interpretable',
        'best_for': 'Complex, non-linear relationships'
    },
    'Gradient Boosting': {
        'description': 'Builds trees sequentially, each correcting previous errors.',
        'pros': 'Often highest accuracy, handles mixed data types well',
        'cons': 'Can overfit, slower to train, sensitive to parameters',
        'best_for': 'When maximum accuracy is needed'
    },
    'Ridge': {
        'description': 'Linear regression with regularization to prevent overfitting.',
        'pros': 'Fast, interpretable, good with correlated features',
        'cons': 'Assumes linear relationships',
        'best_for': 'Linear relationships, quick baseline'
    },
    'SVR': {
        'description': 'Support Vector Regression - finds optimal hyperplane.',
        'pros': 'Works well with small datasets, handles non-linear data',
        'cons': 'Slower, requires feature scaling, less interpretable',
        'best_for': 'Small datasets with complex patterns'
    },
    'Linear': {
        'description': 'Simple linear regression - finds best straight line fit.',
        'pros': 'Very fast, highly interpretable, good baseline',
        'cons': 'Cannot capture non-linear relationships',
        'best_for': 'Simple linear relationships, interpretability'
    }
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_initial_dataframe():
    """Create DataFrame from embedded data - ALL 37 RECORDS"""
    df = pd.DataFrame(INITIAL_DATA)
    return df

def load_data():
    """Load data from file or use initial embedded data"""
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            # Ensure we have at least the initial data
            if len(df) >= 37:
                return df
        except:
            pass
    
    # Use initial data and save it
    df = get_initial_dataframe()
    save_data(df)
    return df

def save_data(df):
    """Save data to CSV file for persistence"""
    df.to_csv(DATA_FILE, index=False)

def load_models():
    """Load trained models from pickle file"""
    if os.path.exists(MODELS_FILE):
        try:
            with open(MODELS_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return {}

def save_models(models):
    """Save trained models to pickle file"""
    with open(MODELS_FILE, 'wb') as f:
        pickle.dump(models, f)

def load_configs():
    """Load best configurations from JSON file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_configs(configs):
    """Save best configurations to JSON file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(configs, f)

def create_features(df, predictor_cols):
    """Create engineered features from predictors"""
    X = df[predictor_cols].copy()
    
    if len(predictor_cols) == 2:
        X['interaction'] = df[predictor_cols[0]] * df[predictor_cols[1]]
        X['ratio'] = df[predictor_cols[0]] / (df[predictor_cols[1]] + 1e-6)
    
    for col in predictor_cols:
        X[f'{col}_squared'] = df[col] ** 2
        if df[col].min() > 0:
            X[f'{col}_log'] = np.log(df[col] + 1)
    
    return X

def get_model_instance(algorithm):
    """Get a fresh model instance"""
    models_dict = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
        'Linear': LinearRegression()
    }
    return models_dict.get(algorithm)

def evaluate_combination(df, predictors, target, algorithm):
    """Evaluate a predictor/algorithm combination using cross-validation"""
    required_cols = predictors + [target]
    valid_data = df[required_cols].dropna()
    
    # Filter out obvious outliers (like the 169 value in Total Hardness)
    if target == 'Total_Hardness_ppm':
        valid_data = valid_data[valid_data[target] > 500]
    
    if len(valid_data) < 3:
        return None
    
    X = create_features(valid_data, predictors)
    y = valid_data[target].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = get_model_instance(algorithm)
    if model is None:
        return None
    
    try:
        # Use 5-fold CV for larger datasets, LOO for smaller
        if len(valid_data) >= 10:
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            r2 = np.mean(cv_scores)
            
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
        else:
            loo = LeaveOneOut()
            predictions, actuals = [], []
            
            for train_idx, test_idx in loo.split(X_scaled):
                model_clone = get_model_instance(algorithm)
                model_clone.fit(X_scaled[train_idx], y[train_idx])
                predictions.append(model_clone.predict(X_scaled[test_idx])[0])
                actuals.append(y[test_idx][0])
            
            r2 = r2_score(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'n_samples': len(valid_data)}
    except Exception as e:
        return None

def train_model(df, predictors, target, algorithm):
    """Train final model on all available data"""
    required_cols = predictors + [target]
    valid_data = df[required_cols].dropna()
    
    # Filter out obvious outliers
    if target == 'Total_Hardness_ppm':
        valid_data = valid_data[valid_data[target] > 500]
    
    if len(valid_data) < 2:
        return None
    
    X = create_features(valid_data, predictors)
    y = valid_data[target].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = get_model_instance(algorithm)
    model.fit(X_scaled, y)
    
    y_pred = model.predict(X_scaled)
    
    return {
        'model': model,
        'scaler': scaler,
        'predictors': predictors,
        'feature_names': X.columns.tolist(),
        'algorithm': algorithm,
        'metrics': {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'n_samples': len(valid_data)
        },
        'y_true': y,
        'y_pred': y_pred
    }

def find_best_configuration(df, target, user_algorithm=None):
    """Find best predictor/algorithm combination"""
    predictor_options = {
        'Conductivity Only': ['Conductivity_uS_cm'],
        'pH Only': ['pH'],
        'Conductivity + pH': ['Conductivity_uS_cm', 'pH']
    }
    
    if user_algorithm:
        algorithms = [user_algorithm]
    else:
        algorithms = ['Random Forest', 'Gradient Boosting', 'Ridge', 'SVR', 'Linear']
    
    best_result = None
    best_config = None
    
    for pred_name, pred_cols in predictor_options.items():
        for algorithm in algorithms:
            result = evaluate_combination(df, pred_cols, target, algorithm)
            if result and (best_result is None or result['r2'] > best_result['r2']):
                best_result = result
                best_config = {
                    'predictors_name': pred_name,
                    'predictors': pred_cols,
                    'algorithm': algorithm,
                    'r2': result['r2'],
                    'rmse': result['rmse'],
                    'mae': result['mae']
                }
    
    return best_config

def run_full_analysis(df, user_algorithm_total=None, user_algorithm_calcium=None):
    """Run complete analysis and model training"""
    targets = ['Total_Hardness_ppm', 'Calculated_Hardness_ppm']
    predictors = ['Conductivity_uS_cm', 'pH']
    
    # Calculate correlations
    correlations = {}
    for target in targets:
        corrs = {}
        pvals = {}
        
        # Filter valid data
        target_data = df.copy()
        if target == 'Total_Hardness_ppm':
            target_data = target_data[target_data[target] > 500]
        
        for predictor in predictors:
            if predictor in target_data.columns and target in target_data.columns:
                valid_data = target_data[[predictor, target]].dropna()
                if len(valid_data) >= 3:
                    r, p = stats.pearsonr(valid_data[predictor], valid_data[target])
                    corrs[predictor] = r
                    pvals[predictor] = p
        correlations[target] = {'correlations': corrs, 'p_values': pvals}
    
    # Find best configurations
    best_configs = {}
    user_algorithms = {
        'Total_Hardness_ppm': user_algorithm_total,
        'Calculated_Hardness_ppm': user_algorithm_calcium
    }
    
    for target in targets:
        config = find_best_configuration(df, target, user_algorithms.get(target))
        if config:
            best_configs[target] = config
    
    # Train models
    trained_models = {}
    for target, config in best_configs.items():
        result = train_model(df, config['predictors'], target, config['algorithm'])
        if result:
            trained_models[target] = result
    
    return correlations, best_configs, trained_models

# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Water Hardness Predictor", page_icon="💧", layout="wide")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.data = None
    st.session_state.correlations = None
    st.session_state.best_configs = None
    st.session_state.models = None

# Auto-initialize on startup - ALWAYS train with all data
if not st.session_state.initialized:
    with st.spinner("🚀 Loading all 37 records and training models..."):
        # Always start fresh with all embedded data
        st.session_state.data = get_initial_dataframe()
        save_data(st.session_state.data)
        
        # Run full analysis
        correlations, best_configs, models = run_full_analysis(st.session_state.data)
        st.session_state.correlations = correlations
        st.session_state.best_configs = best_configs
        st.session_state.models = models
        
        save_models(models)
        save_configs(best_configs)
        
        st.session_state.initialized = True

# ============================================================
# MAIN UI
# ============================================================

st.title("💧 Water Hardness Prediction System")
st.markdown("**Complete ML System with All 37 Training Records**")

# Status bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.success(f"✅ Data: {len(st.session_state.data)} records")
with col2:
    if st.session_state.models:
        st.success(f"✅ Models: {len(st.session_state.models)}")
with col3:
    if st.session_state.best_configs:
        for target, config in st.session_state.best_configs.items():
            name = "TH" if "Total" in target else "Ca"
            st.caption(f"{name}: {config['algorithm']}")
with col4:
    st.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🤖 Algorithm Selection")

algo_total = st.sidebar.selectbox(
    "Total Hardness Algorithm",
    ['Auto (Best)', 'Random Forest', 'Gradient Boosting', 'Ridge', 'SVR', 'Linear'],
    index=0
)

algo_calcium = st.sidebar.selectbox(
    "Calcium Hardness Algorithm",
    ['Auto (Best)', 'Random Forest', 'Gradient Boosting', 'Ridge', 'SVR', 'Linear'],
    index=0
)

if st.sidebar.button("🔄 Retrain with Selected Algorithms", type="primary"):
    user_algo_total = None if algo_total == 'Auto (Best)' else algo_total
    user_algo_calcium = None if algo_calcium == 'Auto (Best)' else algo_calcium
    
    with st.spinner("Retraining..."):
        correlations, best_configs, models = run_full_analysis(
            st.session_state.data,
            user_algorithm_total=user_algo_total,
            user_algorithm_calcium=user_algo_calcium
        )
        st.session_state.correlations = correlations
        st.session_state.best_configs = best_configs
        st.session_state.models = models
        save_models(models)
        save_configs(best_configs)
    
    st.sidebar.success("✅ Retrained!")
    st.rerun()

st.sidebar.divider()
st.sidebar.header("➕ Add New Data")

with st.sidebar.form("add_data_form"):
    new_date = st.text_input("Date", value=datetime.now().strftime('%d-%m-%y'))
    new_conductivity = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=4500.0, step=50.0)
    new_ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=8.0, step=0.1)
    new_total_hardness = st.number_input("Total Hardness (ppm)", min_value=0.0, value=1500.0, step=10.0)
    new_calcium_hardness = st.number_input("Calculated Hardness (ppm)", min_value=0.0, value=1200.0, step=10.0)
    new_calcium_ppm = st.number_input("Calcium (ppm)", min_value=0.0, value=480.0, step=10.0)
    new_alkalinity = st.number_input("Total Alkalinity (ppm)", min_value=0.0, value=60.0, step=5.0)
    new_sulfate = st.number_input("Sulfate (ppm)", min_value=0.0, value=450.0, step=10.0)
    new_iron = st.number_input("Iron (ppm)", min_value=0.0, value=0.2, step=0.01)
    
    submitted = st.form_submit_button("➕ Add & Retrain")
    
    if submitted:
        new_row = pd.DataFrame({
            'Date': [new_date],
            'Calculated_Hardness_ppm': [new_calcium_hardness],
            'Calcium_ppm': [new_calcium_ppm],
            'Total_Hardness_ppm': [new_total_hardness],
            'Total_Alkalinity_ppm': [new_alkalinity],
            'Conductivity_uS_cm': [new_conductivity],
            'pH': [new_ph],
            'Sulfate_ppm': [new_sulfate],
            'Iron_ppm': [new_iron]
        })
        
        st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
        save_data(st.session_state.data)
        
        user_algo_total = None if algo_total == 'Auto (Best)' else algo_total
        user_algo_calcium = None if algo_calcium == 'Auto (Best)' else algo_calcium
        
        with st.spinner("Retraining with new data..."):
            correlations, best_configs, models = run_full_analysis(
                st.session_state.data,
                user_algorithm_total=user_algo_total,
                user_algorithm_calcium=user_algo_calcium
            )
            st.session_state.correlations = correlations
            st.session_state.best_configs = best_configs
            st.session_state.models = models
            save_models(models)
            save_configs(best_configs)
        
        st.sidebar.success(f"✅ Added! Now {len(st.session_state.data)} records")
        st.rerun()

# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Predictions", "📈 Correlations", "🤖 Models", "📊 Data", "⚙️ Compare All", "📚 Help"
])

# TAB 1: PREDICTIONS
with tab1:
    st.header("🔮 Make Predictions")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        pred_conductivity = st.number_input(
            "Conductivity (µS/cm)",
            min_value=3000.0, max_value=7000.0,
            value=float(st.session_state.data['Conductivity_uS_cm'].mean()),
            step=50.0, key="pred_cond"
        )
        
        pred_ph = st.number_input(
            "pH",
            min_value=6.0, max_value=10.0,
            value=float(st.session_state.data['pH'].mean()),
            step=0.1, key="pred_ph"
        )
        
        predict_btn = st.button("🔮 Predict Hardness", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Prediction Results")
        
        if predict_btn and st.session_state.models:
            input_df = pd.DataFrame({
                'Conductivity_uS_cm': [pred_conductivity],
                'pH': [pred_ph]
            })
            
            cols = st.columns(2)
            
            for i, (target, model_data) in enumerate(st.session_state.models.items()):
                X = create_features(input_df, model_data['predictors'])
                X_scaled = model_data['scaler'].transform(X)
                prediction = model_data['model'].predict(X_scaled)[0]
                
                display_name = "Total Hardness" if "Total" in target else "Calculated Hardness (Ca)"
                
                with cols[i]:
                    st.metric(label=f"**{display_name}**", value=f"{prediction:.0f} ppm")
                    
                    config = st.session_state.best_configs.get(target, {})
                    st.caption(f"🤖 Algorithm: **{config.get('algorithm', 'N/A')}**")
                    st.caption(f"📊 R² Score: **{config.get('r2', 0):.3f}**")
                    st.caption(f"📐 RMSE: **{config.get('rmse', 0):.1f} ppm**")
            
            st.success("✅ Predictions complete!")
            
            # Data range info
            st.divider()
            st.subheader("📊 Training Data Ranges")
            
            range_cols = st.columns(4)
            with range_cols[0]:
                st.metric("Conductivity", f"{st.session_state.data['Conductivity_uS_cm'].min():.0f} - {st.session_state.data['Conductivity_uS_cm'].max():.0f}")
            with range_cols[1]:
                st.metric("pH", f"{st.session_state.data['pH'].min():.2f} - {st.session_state.data['pH'].max():.2f}")
            with range_cols[2]:
                st.metric("Total Hardness", f"{st.session_state.data['Total_Hardness_ppm'].min():.0f} - {st.session_state.data['Total_Hardness_ppm'].max():.0f}")
            with range_cols[3]:
                st.metric("Calc Hardness", f"{st.session_state.data['Calculated_Hardness_ppm'].min():.0f} - {st.session_state.data['Calculated_Hardness_ppm'].max():.0f}")

# TAB 2: CORRELATIONS
with tab2:
    st.header("📈 Correlation Analysis")
    st.info(f"📊 Analysis based on **{len(st.session_state.data)} records**")
    
    if st.session_state.correlations:
        # Heatmap
        st.subheader("Correlation Heatmap")
        
        numeric_cols = ['Conductivity_uS_cm', 'pH', 'Total_Hardness_ppm', 'Calculated_Hardness_ppm', 'Calcium_ppm', 'Total_Alkalinity_ppm']
        available_cols = [c for c in numeric_cols if c in st.session_state.data.columns]
        
        corr_matrix = st.session_state.data[available_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        for target in ['Total_Hardness_ppm', 'Calculated_Hardness_ppm']:
            display_name = "Total Hardness" if "Total" in target else "Calculated Hardness (Calcium)"
            st.subheader(f"📊 {display_name}")
            
            if target in st.session_state.correlations:
                corr_data = st.session_state.correlations[target]
                correlations = corr_data['correlations']
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    colors = ['green' if abs(v) > 0.5 else 'orange' if abs(v) > 0.3 else 'red' 
                             for v in correlations.values()]
                    
                    fig = go.Figure(data=[
                        go.Bar(x=list(correlations.keys()), y=list(correlations.values()),
                              marker_color=colors,
                              text=[f"{v:.3f}" for v in correlations.values()],
                              textposition='auto')
                    ])
                    fig.update_layout(title="Correlation Coefficients", yaxis_range=[-1, 1], height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    best = max(correlations.items(), key=lambda x: abs(x[1]))
                    strength = "Strong" if abs(best[1]) > 0.5 else "Moderate" if abs(best[1]) > 0.3 else "Weak"
                    st.info(f"**Best Predictor:** {best[0]} (r={best[1]:.3f}, {strength})")
                
                with col2:
                    fig = make_subplots(rows=1, cols=2, subplot_titles=('Conductivity vs Hardness', 'pH vs Hardness'))
                    
                    # Filter outliers for visualization
                    plot_data = st.session_state.data.copy()
                    if target == 'Total_Hardness_ppm':
                        plot_data = plot_data[plot_data[target] > 500]
                    
                    fig.add_trace(go.Scatter(
                        x=plot_data['Conductivity_uS_cm'],
                        y=plot_data[target],
                        mode='markers', name='Conductivity',
                        marker=dict(color='blue', size=8)
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=plot_data['pH'],
                        y=plot_data[target],
                        mode='markers', name='pH',
                        marker=dict(color='green', size=8)
                    ), row=1, col=2)
                    
                    fig.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            st.divider()

# TAB 3: MODELS
with tab3:
    st.header("🤖 Model Performance")
    st.info(f"📊 Models trained on **{len(st.session_state.data)} records**")
    
    # Metrics explanation
    with st.expander("📚 What do these metrics mean?"):
        st.markdown("""
        | Metric | Name | Meaning | Good Value |
        |--------|------|---------|------------|
        | **R²** | Coefficient of Determination | How well the model explains variance (0-1) | > 0.7 good, > 0.9 excellent |
        | **RMSE** | Root Mean Square Error | Average error in ppm (penalizes large errors) | Lower is better |
        | **MAE** | Mean Absolute Error | Average absolute error in ppm | Lower is better |
        """)
    
    if st.session_state.models and st.session_state.best_configs:
        for target, model_data in st.session_state.models.items():
            display_name = "Total Hardness" if "Total" in target else "Calculated Hardness (Calcium)"
            config = st.session_state.best_configs.get(target, {})
            
            st.subheader(f"📊 {display_name}")
            
            algo = config.get('algorithm', 'N/A')
            if algo in ALGORITHM_INFO:
                with st.expander(f"ℹ️ About {algo}"):
                    info = ALGORITHM_INFO[algo]
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown(f"**✅ Pros:** {info['pros']}")
                    st.markdown(f"**❌ Cons:** {info['cons']}")
            
            # Metrics
            cols = st.columns(5)
            with cols[0]:
                st.metric("Algorithm", algo)
            with cols[1]:
                r2_val = model_data['metrics']['r2']
                delta = "Excellent" if r2_val > 0.9 else "Good" if r2_val > 0.7 else "Fair"
                st.metric("R² Score", f"{r2_val:.4f}", delta=delta)
            with cols[2]:
                st.metric("RMSE", f"{model_data['metrics']['rmse']:.1f} ppm")
            with cols[3]:
                st.metric("MAE", f"{model_data['metrics']['mae']:.1f} ppm")
            with cols[4]:
                st.metric("Training Samples", model_data['metrics']['n_samples'])
            
            st.info(f"**Features Used:** {config.get('predictors_name', 'N/A')}")
            
            # Plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                min_val = min(model_data['y_true'].min(), model_data['y_pred'].min())
                max_val = max(model_data['y_true'].max(), model_data['y_pred'].max())
                
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                        mode='lines', name='Perfect', line=dict(dash='dash', color='red')))
                fig.add_trace(go.Scatter(x=model_data['y_true'], y=model_data['y_pred'],
                                        mode='markers', name='Predictions', marker=dict(size=10, color='blue')))
                
                fig.update_layout(title="Predicted vs Actual", xaxis_title="Actual (ppm)", 
                                yaxis_title="Predicted (ppm)", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                residuals = model_data['y_true'] - model_data['y_pred']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=model_data['y_pred'], y=residuals,
                                        mode='markers', marker=dict(size=10, color='blue')))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                
                fig.update_layout(title="Residuals (Prediction Errors)", xaxis_title="Predicted (ppm)",
                                yaxis_title="Error (ppm)", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if hasattr(model_data['model'], 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                importances = model_data['model'].feature_importances_
                feature_names = model_data['feature_names']
                indices = np.argsort(importances)[::-1]
                
                fig = go.Figure(data=[go.Bar(x=[feature_names[i] for i in indices],
                                            y=[importances[i] for i in indices],
                                            marker_color='steelblue')])
                fig.update_layout(title="Which features matter most?", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()

# TAB 4: DATA
with tab4:
    st.header("📊 Training Data")
    st.success(f"✅ **{len(st.session_state.data)} records** loaded and used for training")
    
    # Summary stats
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Records", len(st.session_state.data))
    with cols[1]:
        st.metric("Avg Total Hardness", f"{st.session_state.data['Total_Hardness_ppm'].mean():.0f} ppm")
    with cols[2]:
        st.metric("Avg Calc Hardness", f"{st.session_state.data['Calculated_Hardness_ppm'].mean():.0f} ppm")
    with cols[3]:
        st.metric("Avg Conductivity", f"{st.session_state.data['Conductivity_uS_cm'].mean():.0f} µS/cm")
    
    # Data table
    st.subheader("All Training Data")
    st.dataframe(st.session_state.data, use_container_width=True, height=500)
    
    # Download
    csv = st.session_state.data.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, f"water_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    st.divider()
    st.subheader("⚠️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Delete Last Row"):
            if len(st.session_state.data) > 10:
                st.session_state.data = st.session_state.data.iloc[:-1]
                save_data(st.session_state.data)
                
                correlations, best_configs, models = run_full_analysis(st.session_state.data)
                st.session_state.correlations = correlations
                st.session_state.best_configs = best_configs
                st.session_state.models = models
                save_models(models)
                save_configs(best_configs)
                
                st.success("Deleted & retrained!")
                st.rerun()
            else:
                st.error("Cannot delete - minimum 10 records needed")
    
    with col2:
        if st.button("🔄 Reset to Original 37 Records"):
            st.session_state.data = get_initial_dataframe()
            save_data(st.session_state.data)
            
            correlations, best_configs, models = run_full_analysis(st.session_state.data)
            st.session_state.correlations = correlations
            st.session_state.best_configs = best_configs
            st.session_state.models = models
            save_models(models)
            save_configs(best_configs)
            
            st.success("✅ Reset to original 37 records!")
            st.rerun()

# TAB 5: COMPARE ALL
with tab5:
    st.header("⚙️ Compare All Configurations")
    st.info(f"📊 Testing with **{len(st.session_state.data)} records**")
    
    if st.button("🔍 Run Full Comparison", type="primary"):
        predictor_options = {
            'Conductivity Only': ['Conductivity_uS_cm'],
            'pH Only': ['pH'],
            'Conductivity + pH': ['Conductivity_uS_cm', 'pH']
        }
        algorithms = ['Random Forest', 'Gradient Boosting', 'Ridge', 'SVR', 'Linear']
        
        for target in ['Total_Hardness_ppm', 'Calculated_Hardness_ppm']:
            display_name = "Total Hardness" if "Total" in target else "Calculated Hardness"
            st.subheader(f"📊 {display_name}")
            
            results = []
            progress = st.progress(0)
            total = len(predictor_options) * len(algorithms)
            current = 0
            
            for pred_name, pred_cols in predictor_options.items():
                for algorithm in algorithms:
                    result = evaluate_combination(st.session_state.data, pred_cols, target, algorithm)
                    if result:
                        results.append({
                            'Predictors': pred_name,
                            'Algorithm': algorithm,
                            'R² Score': result['r2'],
                            'RMSE': result['rmse'],
                            'MAE': result['mae'],
                            'Samples': result['n_samples']
                        })
                    current += 1
                    progress.progress(current / total)
            
            progress.empty()
            
            if results:
                results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
                
                st.dataframe(
                    results_df.style.background_gradient(subset=['R² Score'], cmap='RdYlGn')
                              .format({'R² Score': '{:.4f}', 'RMSE': '{:.2f}', 'MAE': '{:.2f}'}),
                    use_container_width=True
                )
                
                best = results_df.iloc[0]
                st.success(f"🏆 **Best:** {best['Predictors']} + {best['Algorithm']} | R² = {best['R² Score']:.4f} | Trained on {best['Samples']} samples")
                
                fig = go.Figure()
                for pred_name in predictor_options.keys():
                    subset = results_df[results_df['Predictors'] == pred_name]
                    fig.add_trace(go.Bar(x=subset['Algorithm'], y=subset['R² Score'], name=pred_name))
                
                fig.update_layout(title=f"R² Score Comparison - {display_name}", 
                                barmode='group', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()

# TAB 6: HELP
with tab6:
    st.header("📚 Help & Documentation")
    
    st.subheader("📊 Understanding the Metrics")
    st.markdown("""
    ### R² Score (Coefficient of Determination)
    - **Range:** 0 to 1
    - **Meaning:** How much of the variance in hardness is explained by the model
    - **> 0.9:** Excellent | **0.7-0.9:** Good | **0.5-0.7:** Moderate | **< 0.5:** Poor
    
    ### RMSE (Root Mean Square Error)
    - **Units:** ppm
    - **Meaning:** Average prediction error (larger errors penalized more)
    - **Example:** RMSE = 50 means predictions are typically off by ~50 ppm
    
    ### MAE (Mean Absolute Error)
    - **Units:** ppm
    - **Meaning:** Simple average of absolute errors
    - **More intuitive** than RMSE for understanding typical error
    """)
    
    st.divider()
    st.subheader("🤖 Algorithm Guide")
    
    for algo, info in ALGORITHM_INFO.items():
        with st.expander(f"**{algo}**"):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**✅ Pros:** {info['pros']}")
            st.markdown(f"**❌ Cons:** {info['cons']}")
            st.markdown(f"**🎯 Best for:** {info['best_for']}")
    
    st.divider()
    st.subheader("📋 Data Summary")
    st.markdown(f"""
    - **Initial Records:** 37 water quality measurements
    - **Date Range:** January 2025 - September 2025
    - **Key Variables:**
        - Conductivity: {st.session_state.data['Conductivity_uS_cm'].min():.0f} - {st.session_state.data['Conductivity_uS_cm'].max():.0f} µS/cm
        - pH: {st.session_state.data['pH'].min():.2f} - {st.session_state.data['pH'].max():.2f}
        - Total Hardness: {st.session_state.data['Total_Hardness_ppm'].min():.0f} - {st.session_state.data['Total_Hardness_ppm'].max():.0f} ppm
        - Calculated Hardness: {st.session_state.data['Calculated_Hardness_ppm'].min():.0f} - {st.session_state.data['Calculated_Hardness_ppm'].max():.0f} ppm
    """)

# Footer
st.divider()
st.caption(f"💧 Water Hardness Prediction | {len(st.session_state.data)} Training Records | Auto-Training | Persistent Storage")
