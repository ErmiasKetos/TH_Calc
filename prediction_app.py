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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION - File paths for persistent storage
# ============================================================
DATA_FILE = 'water_quality_data.csv'
MODELS_FILE = 'trained_models.pkl'
CONFIG_FILE = 'best_configs.json'

# ============================================================
# EMBEDDED INITIAL DATA - Your 14 water testing records
# ============================================================
INITIAL_DATA = {
    'Date': ['23-06-25', '04-07-25', '09-07-25', '11-07-25', '16-07-25', 
             '18-07-25', '23-07-25', '25-07-25', '30-07-25', '01-08-25',
             '06-08-25', '08-08-25', '13-08-25', '15-08-25'],
    'Calculated_Hardness_ppm': [1130, 1050, 1050, 1180, 1020, 1200, 980, 1100, 1080, 1150, 1030, 1090, 1070, 1040],
    'Total_Hardness_ppm': [1590, 1390, 1460, 1520, 1340, 1580, 1190, 1470, 1410, 1230, 1350, 1490, 1440, 1380],
    'Chloride_ppm': [588, 635, 588, 612, 598, 625, 590, 608, 595, 620, 605, 592, 615, 600],
    'Alkalinity_ppm': [78, 88, 78, 82, 85, 80, 90, 84, 86, 92, 88, 79, 83, 87],
    'Conductivity_uS_cm': [4520, 4230, 4311, 4920, 4600, 4850, 5490, 4750, 4380, 4950, 4650, 4420, 4780, 4560],
    'pH': [7.74, 8.14, 7.95, 7.88, 8.23, 7.68, 8.45, 7.92, 8.12, 8.76, 8.34, 7.89, 8.01, 7.97],
    'Turbidity_FAU': [9.84, 13.48, 5.68, 8.20, 11.30, 6.90, 15.20, 7.50, 10.10, 12.80, 9.20, 6.40, 8.90, 7.80],
    'Temp_C': [24.6, 25.0, 22.2, 23.8, 24.2, 25.5, 26.1, 24.0, 23.5, 25.8, 24.8, 23.2, 24.5, 23.9],
    'Sulfates_ppm': [1200, 1300, 1210, 1250, 1180, 1280, 1150, 1230, 1190, 1260, 1220, 1200, 1240, 1210],
    'Iron_ppm': [0.15, 0.25, 0.19, 0.22, 0.18, 0.28, 0.12, 0.20, 0.17, 0.24, 0.21, 0.16, 0.23, 0.19]
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_initial_dataframe():
    """Create DataFrame from embedded initial data"""
    return pd.DataFrame(INITIAL_DATA)

def load_data():
    """Load data from file or use initial data"""
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            return df
        except:
            pass
    
    # Use initial data and save it
    df = get_initial_dataframe()
    save_data(df)
    return df

def save_data(df):
    """Save data to CSV file"""
    df.to_csv(DATA_FILE, index=False)

def load_models():
    """Load trained models from file"""
    if os.path.exists(MODELS_FILE):
        try:
            with open(MODELS_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return {}

def save_models(models):
    """Save trained models to file"""
    with open(MODELS_FILE, 'wb') as f:
        pickle.dump(models, f)

def load_configs():
    """Load best configurations from file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_configs(configs):
    """Save best configurations to file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(configs, f)

def calculate_correlations(df, predictors, target):
    """Calculate Pearson correlations"""
    correlations = {}
    p_values = {}
    
    for predictor in predictors:
        if predictor in df.columns and target in df.columns:
            valid_data = df[[predictor, target]].dropna()
            if len(valid_data) >= 3:
                corr, p_val = stats.pearsonr(valid_data[predictor], valid_data[target])
                correlations[predictor] = corr
                p_values[predictor] = p_val
    
    return correlations, p_values

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

def evaluate_combination(df, predictors, target, algorithm):
    """Evaluate a predictor/algorithm combination"""
    required_cols = predictors + [target]
    valid_data = df[required_cols].dropna()
    
    if len(valid_data) < 3:
        return None
    
    X = create_features(valid_data, predictors)
    y = valid_data[target].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models_dict = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
        'Linear': LinearRegression()
    }
    
    model = models_dict.get(algorithm)
    if model is None:
        return None
    
    try:
        if len(valid_data) < 10:
            loo = LeaveOneOut()
            predictions, actuals = [], []
            
            for train_idx, test_idx in loo.split(X_scaled):
                model_clone = models_dict.get(algorithm)
                model_clone.fit(X_scaled[train_idx], y[train_idx])
                predictions.append(model_clone.predict(X_scaled[test_idx])[0])
                actuals.append(y[test_idx][0])
            
            r2 = r2_score(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
        else:
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            r2 = np.mean(cv_scores)
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'n_samples': len(valid_data)}
    except:
        return None

def train_model(df, predictors, target, algorithm):
    """Train final model"""
    required_cols = predictors + [target]
    valid_data = df[required_cols].dropna()
    
    if len(valid_data) < 2:
        return None
    
    X = create_features(valid_data, predictors)
    y = valid_data[target].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models_dict = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
        'Linear': LinearRegression()
    }
    
    model = models_dict.get(algorithm)
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

def find_best_configuration(df, target):
    """Find best predictor/algorithm combination for a target"""
    predictor_options = {
        'Conductivity Only': ['Conductivity_uS_cm'],
        'pH Only': ['pH'],
        'Conductivity + pH': ['Conductivity_uS_cm', 'pH']
    }
    
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

def run_full_analysis(df):
    """Run complete analysis and training"""
    targets = ['Total_Hardness_ppm', 'Calculated_Hardness_ppm']
    predictors = ['Conductivity_uS_cm', 'pH']
    
    # Correlation analysis
    correlations = {}
    for target in targets:
        corrs, pvals = calculate_correlations(df, predictors, target)
        correlations[target] = {'correlations': corrs, 'p_values': pvals}
    
    # Find best configurations
    best_configs = {}
    for target in targets:
        config = find_best_configuration(df, target)
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

# Initialize on first run
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.data = None
    st.session_state.correlations = None
    st.session_state.best_configs = None
    st.session_state.models = None

# Auto-initialize and train on startup
if not st.session_state.initialized:
    with st.spinner("🚀 Initializing system, loading data, and training models..."):
        # Load data
        st.session_state.data = load_data()
        
        # Try to load existing models
        saved_models = load_models()
        saved_configs = load_configs()
        
        if saved_models and saved_configs:
            st.session_state.models = saved_models
            st.session_state.best_configs = saved_configs
            
            # Run correlation analysis
            predictors = ['Conductivity_uS_cm', 'pH']
            targets = ['Total_Hardness_ppm', 'Calculated_Hardness_ppm']
            correlations = {}
            for target in targets:
                corrs, pvals = calculate_correlations(st.session_state.data, predictors, target)
                correlations[target] = {'correlations': corrs, 'p_values': pvals}
            st.session_state.correlations = correlations
        else:
            # Run full analysis and training
            correlations, best_configs, models = run_full_analysis(st.session_state.data)
            st.session_state.correlations = correlations
            st.session_state.best_configs = best_configs
            st.session_state.models = models
            
            # Save for next time
            save_models(models)
            save_configs(best_configs)
        
        st.session_state.initialized = True

# ============================================================
# MAIN UI
# ============================================================

st.title("💧 Water Hardness Prediction System")
st.markdown("**Auto-Training ML System for Total & Calcium Hardness Prediction**")

# Status indicator
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.success(f"✅ Data Loaded: {len(st.session_state.data)} records")
with col2:
    if st.session_state.models:
        st.success(f"✅ Models Trained: {len(st.session_state.models)}")
    else:
        st.error("❌ No Models")
with col3:
    if st.session_state.best_configs:
        st.success("✅ Configs Ready")
with col4:
    st.info(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Sidebar - Add New Data
st.sidebar.header("➕ Add New Data")

with st.sidebar.form("add_data_form"):
    st.subheader("Enter New Measurement")
    
    new_date = st.text_input("Date", value=datetime.now().strftime('%d-%m-%y'))
    new_conductivity = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=4500.0, step=50.0)
    new_ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=8.0, step=0.1)
    new_total_hardness = st.number_input("Total Hardness (ppm)", min_value=0.0, value=1400.0, step=10.0)
    new_calcium_hardness = st.number_input("Calcium/Calculated Hardness (ppm)", min_value=0.0, value=1100.0, step=10.0)
    
    # Optional fields
    with st.expander("Optional Fields"):
        new_chloride = st.number_input("Chloride (ppm)", min_value=0.0, value=600.0)
        new_alkalinity = st.number_input("Alkalinity (ppm)", min_value=0.0, value=85.0)
        new_turbidity = st.number_input("Turbidity (FAU)", min_value=0.0, value=10.0)
        new_temp = st.number_input("Temperature (°C)", min_value=0.0, value=24.0)
        new_sulfates = st.number_input("Sulfates (ppm)", min_value=0.0, value=1200.0)
        new_iron = st.number_input("Iron (ppm)", min_value=0.0, value=0.2, step=0.01)
    
    submitted = st.form_submit_button("➕ Add Data & Retrain", type="primary")
    
    if submitted:
        # Add new row
        new_row = pd.DataFrame({
            'Date': [new_date],
            'Calculated_Hardness_ppm': [new_calcium_hardness],
            'Total_Hardness_ppm': [new_total_hardness],
            'Chloride_ppm': [new_chloride],
            'Alkalinity_ppm': [new_alkalinity],
            'Conductivity_uS_cm': [new_conductivity],
            'pH': [new_ph],
            'Turbidity_FAU': [new_turbidity],
            'Temp_C': [new_temp],
            'Sulfates_ppm': [new_sulfates],
            'Iron_ppm': [new_iron]
        })
        
        st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
        
        # Save updated data
        save_data(st.session_state.data)
        
        # Retrain models
        with st.spinner("Retraining models with new data..."):
            correlations, best_configs, models = run_full_analysis(st.session_state.data)
            st.session_state.correlations = correlations
            st.session_state.best_configs = best_configs
            st.session_state.models = models
            
            save_models(models)
            save_configs(best_configs)
        
        st.sidebar.success("✅ Data added and models retrained!")
        st.rerun()

# Sidebar - Manual Retrain
st.sidebar.divider()
if st.sidebar.button("🔄 Force Retrain Models"):
    with st.spinner("Retraining all models..."):
        correlations, best_configs, models = run_full_analysis(st.session_state.data)
        st.session_state.correlations = correlations
        st.session_state.best_configs = best_configs
        st.session_state.models = models
        
        save_models(models)
        save_configs(best_configs)
    
    st.sidebar.success("✅ Models retrained!")
    st.rerun()

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predictions",
    "📈 Correlation Analysis", 
    "🤖 Model Performance",
    "📊 Data View",
    "⚙️ All Configurations"
])

# ============================================================
# TAB 1: PREDICTIONS
# ============================================================
with tab1:
    st.header("🔮 Make Predictions")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        pred_conductivity = st.number_input(
            "Conductivity (µS/cm)",
            min_value=1000.0,
            max_value=10000.0,
            value=float(st.session_state.data['Conductivity_uS_cm'].mean()),
            step=50.0,
            key="pred_cond"
        )
        
        pred_ph = st.number_input(
            "pH",
            min_value=5.0,
            max_value=10.0,
            value=float(st.session_state.data['pH'].mean()),
            step=0.1,
            key="pred_ph"
        )
        
        predict_btn = st.button("🔮 Predict Hardness", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Prediction Results")
        
        if predict_btn and st.session_state.models:
            input_df = pd.DataFrame({
                'Conductivity_uS_cm': [pred_conductivity],
                'pH': [pred_ph]
            })
            
            results_col1, results_col2 = st.columns(2)
            
            for i, (target, model_data) in enumerate(st.session_state.models.items()):
                X = create_features(input_df, model_data['predictors'])
                X_scaled = model_data['scaler'].transform(X)
                prediction = model_data['model'].predict(X_scaled)[0]
                
                # Determine display name
                display_name = "Total Hardness" if "Total" in target else "Calcium Hardness"
                
                with results_col1 if i == 0 else results_col2:
                    st.metric(
                        label=f"**{display_name}**",
                        value=f"{prediction:.0f} ppm"
                    )
                    
                    config = st.session_state.best_configs.get(target, {})
                    st.caption(f"Model: {config.get('algorithm', 'N/A')}")
                    st.caption(f"Features: {config.get('predictors_name', 'N/A')}")
                    st.caption(f"R² Score: {config.get('r2', 0):.3f}")
            
            st.success("✅ Predictions complete!")
            
            # Show input ranges
            st.divider()
            st.subheader("Training Data Ranges (for reference)")
            
            range_col1, range_col2, range_col3 = st.columns(3)
            with range_col1:
                st.metric("Conductivity Range", 
                         f"{st.session_state.data['Conductivity_uS_cm'].min():.0f} - {st.session_state.data['Conductivity_uS_cm'].max():.0f}")
            with range_col2:
                st.metric("pH Range",
                         f"{st.session_state.data['pH'].min():.2f} - {st.session_state.data['pH'].max():.2f}")
            with range_col3:
                in_range = (st.session_state.data['Conductivity_uS_cm'].min() <= pred_conductivity <= st.session_state.data['Conductivity_uS_cm'].max() and
                           st.session_state.data['pH'].min() <= pred_ph <= st.session_state.data['pH'].max())
                if in_range:
                    st.success("✅ Input within training range")
                else:
                    st.warning("⚠️ Input outside training range")
        
        elif not st.session_state.models:
            st.error("No models trained. Please check the Model Performance tab.")

# ============================================================
# TAB 2: CORRELATION ANALYSIS
# ============================================================
with tab2:
    st.header("📈 Correlation Analysis")
    
    if st.session_state.correlations:
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        
        numeric_cols = ['Conductivity_uS_cm', 'pH', 'Total_Hardness_ppm', 'Calculated_Hardness_ppm', 
                       'Alkalinity_ppm', 'Chloride_ppm', 'Sulfates_ppm']
        available_cols = [c for c in numeric_cols if c in st.session_state.data.columns]
        
        corr_matrix = st.session_state.data[available_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto='.3f', aspect='auto',
                       color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis for each target
        st.divider()
        
        for target in ['Total_Hardness_ppm', 'Calculated_Hardness_ppm']:
            display_name = "Total Hardness" if "Total" in target else "Calcium Hardness"
            st.subheader(f"📊 {display_name} Correlations")
            
            if target in st.session_state.correlations:
                corr_data = st.session_state.correlations[target]
                correlations = corr_data['correlations']
                p_values = corr_data['p_values']
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Bar chart
                    colors = ['green' if abs(v) > 0.5 else 'orange' if abs(v) > 0.3 else 'red' 
                             for v in correlations.values()]
                    
                    fig = go.Figure(data=[
                        go.Bar(x=list(correlations.keys()), y=list(correlations.values()),
                              marker_color=colors,
                              text=[f"{v:.3f}" for v in correlations.values()],
                              textposition='auto')
                    ])
                    fig.update_layout(title=f"Correlations with {display_name}",
                                    yaxis_range=[-1, 1], height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Best predictor
                    best = max(correlations.items(), key=lambda x: abs(x[1]))
                    strength = "Strong" if abs(best[1]) > 0.5 else "Moderate" if abs(best[1]) > 0.3 else "Weak"
                    st.info(f"**Best Predictor:** {best[0]} (r = {best[1]:.3f}, {strength})")
                
                with col2:
                    # Scatter plots
                    scatter_col1, scatter_col2 = st.columns(2)
                    
                    with scatter_col1:
                        fig = px.scatter(st.session_state.data, x='Conductivity_uS_cm', y=target,
                                       trendline="ols", title="Conductivity vs Hardness")
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with scatter_col2:
                        fig = px.scatter(st.session_state.data, x='pH', y=target,
                                       trendline="ols", title="pH vs Hardness")
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
            
            st.divider()

# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================
with tab3:
    st.header("🤖 Model Performance")
    
    if st.session_state.models and st.session_state.best_configs:
        for target, model_data in st.session_state.models.items():
            display_name = "Total Hardness" if "Total" in target else "Calcium Hardness"
            config = st.session_state.best_configs.get(target, {})
            
            st.subheader(f"📊 {display_name} Model")
            
            # Metrics
            metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
            
            with metrics_col1:
                st.metric("Algorithm", config.get('algorithm', 'N/A'))
            with metrics_col2:
                st.metric("R² Score", f"{model_data['metrics']['r2']:.4f}")
            with metrics_col3:
                st.metric("RMSE", f"{model_data['metrics']['rmse']:.2f} ppm")
            with metrics_col4:
                st.metric("MAE", f"{model_data['metrics']['mae']:.2f} ppm")
            with metrics_col5:
                st.metric("Samples", model_data['metrics']['n_samples'])
            
            st.info(f"**Features Used:** {config.get('predictors_name', 'N/A')} ({', '.join(config.get('predictors', []))})")
            
            # Visualization
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Predicted vs Actual
                fig = go.Figure()
                
                min_val = min(model_data['y_true'].min(), model_data['y_pred'].min())
                max_val = max(model_data['y_true'].max(), model_data['y_pred'].max())
                
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                        mode='lines', name='Perfect',
                                        line=dict(dash='dash', color='red')))
                
                fig.add_trace(go.Scatter(x=model_data['y_true'], y=model_data['y_pred'],
                                        mode='markers', name='Predictions',
                                        marker=dict(size=12, color='blue')))
                
                fig.update_layout(title="Predicted vs Actual",
                                xaxis_title="Actual (ppm)", yaxis_title="Predicted (ppm)",
                                height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                # Residuals
                residuals = model_data['y_true'] - model_data['y_pred']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=model_data['y_pred'], y=residuals,
                                        mode='markers', marker=dict(size=12, color='blue')))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                
                fig.update_layout(title="Residuals Plot",
                                xaxis_title="Predicted (ppm)", yaxis_title="Residuals (ppm)",
                                height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance for tree-based models
            if hasattr(model_data['model'], 'feature_importances_'):
                st.subheader("Feature Importance")
                
                importances = model_data['model'].feature_importances_
                feature_names = model_data['feature_names']
                indices = np.argsort(importances)[::-1]
                
                fig = go.Figure(data=[
                    go.Bar(x=[feature_names[i] for i in indices],
                          y=[importances[i] for i in indices])
                ])
                fig.update_layout(title="Feature Importance", height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
    else:
        st.error("No models available. Please retrain.")

# ============================================================
# TAB 4: DATA VIEW
# ============================================================
with tab4:
    st.header("📊 Training Data")
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(st.session_state.data))
    with col2:
        st.metric("Avg Total Hardness", f"{st.session_state.data['Total_Hardness_ppm'].mean():.0f} ppm")
    with col3:
        st.metric("Avg Calcium Hardness", f"{st.session_state.data['Calculated_Hardness_ppm'].mean():.0f} ppm")
    with col4:
        st.metric("Avg Conductivity", f"{st.session_state.data['Conductivity_uS_cm'].mean():.0f} µS/cm")
    
    # Data table
    st.subheader("All Data")
    st.dataframe(st.session_state.data, use_container_width=True, height=400)
    
    # Download button
    csv = st.session_state.data.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"water_quality_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(st.session_state.data.describe(), use_container_width=True)
    
    # Delete last row option
    st.divider()
    st.subheader("⚠️ Data Management")
    
    if st.button("🗑️ Delete Last Row"):
        if len(st.session_state.data) > 1:
            st.session_state.data = st.session_state.data.iloc[:-1]
            save_data(st.session_state.data)
            
            # Retrain
            correlations, best_configs, models = run_full_analysis(st.session_state.data)
            st.session_state.correlations = correlations
            st.session_state.best_configs = best_configs
            st.session_state.models = models
            save_models(models)
            save_configs(best_configs)
            
            st.success("Last row deleted and models retrained!")
            st.rerun()
        else:
            st.error("Cannot delete - only one row remaining")
    
    if st.button("🔄 Reset to Original Data"):
        st.session_state.data = get_initial_dataframe()
        save_data(st.session_state.data)
        
        correlations, best_configs, models = run_full_analysis(st.session_state.data)
        st.session_state.correlations = correlations
        st.session_state.best_configs = best_configs
        st.session_state.models = models
        save_models(models)
        save_configs(best_configs)
        
        st.success("Data reset to original 14 records!")
        st.rerun()

# ============================================================
# TAB 5: ALL CONFIGURATIONS
# ============================================================
with tab5:
    st.header("⚙️ All Model Configurations Tested")
    
    if st.button("🔍 Run Full Configuration Comparison"):
        predictor_options = {
            'Conductivity Only': ['Conductivity_uS_cm'],
            'pH Only': ['pH'],
            'Conductivity + pH': ['Conductivity_uS_cm', 'pH']
        }
        algorithms = ['Random Forest', 'Gradient Boosting', 'Ridge', 'SVR', 'Linear']
        
        for target in ['Total_Hardness_ppm', 'Calculated_Hardness_ppm']:
            display_name = "Total Hardness" if "Total" in target else "Calcium Hardness"
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
                            'MAE': result['mae']
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
                st.success(f"🏆 **Best:** {best['Predictors']} + {best['Algorithm']} | R² = {best['R² Score']:.4f}")
                
                # Visualization
                fig = px.bar(results_df, x='Algorithm', y='R² Score', color='Predictors',
                           barmode='group', title=f"All Configurations for {display_name}")
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()

# Footer
st.divider()
st.markdown("---")
st.caption("💧 Water Hardness Prediction System | Data persists between sessions | Models auto-train on startup")
