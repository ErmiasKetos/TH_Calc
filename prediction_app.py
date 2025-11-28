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
    return pd.DataFrame(INITIAL_DATA)

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            return df
        except:
            pass
    df = get_initial_dataframe()
    save_data(df)
    return df

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def load_models():
    if os.path.exists(MODELS_FILE):
        try:
            with open(MODELS_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return {}

def save_models(models):
    with open(MODELS_FILE, 'wb') as f:
        pickle.dump(models, f)

def load_configs():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_configs(configs):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(configs, f)

def create_features(df, predictor_cols):
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
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
        'Linear': LinearRegression()
    }
    return models_dict.get(algorithm)

def evaluate_combination(df, predictors, target, algorithm):
    required_cols = predictors + [target]
    valid_data = df[required_cols].dropna()
    
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
        if len(valid_data) < 10:
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
    required_cols = predictors + [target]
    valid_data = df[required_cols].dropna()
    
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
    """Find best configuration, optionally using user-selected algorithm"""
    predictor_options = {
        'Conductivity Only': ['Conductivity_uS_cm'],
        'pH Only': ['pH'],
        'Conductivity + pH': ['Conductivity_uS_cm', 'pH']
    }
    
    # If user selected an algorithm, only test that one
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
    """Run analysis with optional user-selected algorithms"""
    targets = ['Total_Hardness_ppm', 'Calculated_Hardness_ppm']
    predictors = ['Conductivity_uS_cm', 'pH']
    
    # Correlations
    correlations = {}
    for target in targets:
        corrs = {}
        pvals = {}
        for predictor in predictors:
            if predictor in df.columns and target in df.columns:
                valid_data = df[[predictor, target]].dropna()
                if len(valid_data) >= 3:
                    r, p = stats.pearsonr(valid_data[predictor], valid_data[target])
                    corrs[predictor] = r
                    pvals[predictor] = p
        correlations[target] = {'correlations': corrs, 'p_values': pvals}
    
    # Find best configs
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
    st.session_state.user_algo_total = None
    st.session_state.user_algo_calcium = None

# Auto-initialize on startup
if not st.session_state.initialized:
    with st.spinner("🚀 Initializing system and training models..."):
        st.session_state.data = load_data()
        
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
                corrs = {}
                pvals = {}
                for predictor in predictors:
                    if predictor in st.session_state.data.columns and target in st.session_state.data.columns:
                        valid_data = st.session_state.data[[predictor, target]].dropna()
                        if len(valid_data) >= 3:
                            r, p = stats.pearsonr(valid_data[predictor], valid_data[target])
                            corrs[predictor] = r
                            pvals[predictor] = p
                correlations[target] = {'correlations': corrs, 'p_values': pvals}
            st.session_state.correlations = correlations
        else:
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
st.markdown("**ML System with User-Selectable Algorithms**")

# Status bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.success(f"✅ Data: {len(st.session_state.data)} records")
with col2:
    if st.session_state.models:
        st.success(f"✅ Models: {len(st.session_state.models)}")
with col3:
    if st.session_state.best_configs:
        st.success("✅ Ready")
with col4:
    st.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🤖 Algorithm Selection")

st.sidebar.markdown("**Choose algorithms for each target:**")

# Algorithm selection for Total Hardness
algo_total = st.sidebar.selectbox(
    "Total Hardness Algorithm",
    ['Auto (Best)', 'Random Forest', 'Gradient Boosting', 'Ridge', 'SVR', 'Linear'],
    index=0,
    key='algo_total_select'
)

# Algorithm selection for Calcium Hardness
algo_calcium = st.sidebar.selectbox(
    "Calcium Hardness Algorithm",
    ['Auto (Best)', 'Random Forest', 'Gradient Boosting', 'Ridge', 'SVR', 'Linear'],
    index=0,
    key='algo_calcium_select'
)

# Retrain button with selected algorithms
if st.sidebar.button("🔄 Retrain with Selected Algorithms", type="primary"):
    user_algo_total = None if algo_total == 'Auto (Best)' else algo_total
    user_algo_calcium = None if algo_calcium == 'Auto (Best)' else algo_calcium
    
    with st.spinner("Retraining with selected algorithms..."):
        correlations, best_configs, models = run_full_analysis(
            st.session_state.data,
            user_algorithm_total=user_algo_total,
            user_algorithm_calcium=user_algo_calcium
        )
        st.session_state.correlations = correlations
        st.session_state.best_configs = best_configs
        st.session_state.models = models
        st.session_state.user_algo_total = user_algo_total
        st.session_state.user_algo_calcium = user_algo_calcium
        save_models(models)
        save_configs(best_configs)
    
    st.sidebar.success("✅ Retrained!")
    st.rerun()

st.sidebar.divider()

# Add new data section
st.sidebar.header("➕ Add New Data")

with st.sidebar.form("add_data_form"):
    new_date = st.text_input("Date", value=datetime.now().strftime('%d-%m-%y'))
    new_conductivity = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=4500.0, step=50.0)
    new_ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=8.0, step=0.1)
    new_total_hardness = st.number_input("Total Hardness (ppm)", min_value=0.0, value=1400.0, step=10.0)
    new_calcium_hardness = st.number_input("Calcium Hardness (ppm)", min_value=0.0, value=1100.0, step=10.0)
    
    with st.expander("Optional Fields"):
        new_chloride = st.number_input("Chloride (ppm)", min_value=0.0, value=600.0)
        new_alkalinity = st.number_input("Alkalinity (ppm)", min_value=0.0, value=85.0)
        new_turbidity = st.number_input("Turbidity (FAU)", min_value=0.0, value=10.0)
        new_temp = st.number_input("Temperature (°C)", min_value=0.0, value=24.0)
        new_sulfates = st.number_input("Sulfates (ppm)", min_value=0.0, value=1200.0)
        new_iron = st.number_input("Iron (ppm)", min_value=0.0, value=0.2, step=0.01)
    
    submitted = st.form_submit_button("➕ Add & Retrain")
    
    if submitted:
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
        save_data(st.session_state.data)
        
        # Use current algorithm selections
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
        
        st.sidebar.success("✅ Added & retrained!")
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
            min_value=1000.0, max_value=10000.0,
            value=float(st.session_state.data['Conductivity_uS_cm'].mean()),
            step=50.0, key="pred_cond"
        )
        
        pred_ph = st.number_input(
            "pH",
            min_value=5.0, max_value=10.0,
            value=float(st.session_state.data['pH'].mean()),
            step=0.1, key="pred_ph"
        )
        
        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Results")
        
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
                
                display_name = "Total Hardness" if "Total" in target else "Calcium Hardness"
                
                with cols[i]:
                    st.metric(label=f"**{display_name}**", value=f"{prediction:.0f} ppm")
                    
                    config = st.session_state.best_configs.get(target, {})
                    st.caption(f"🤖 Algorithm: **{config.get('algorithm', 'N/A')}**")
                    st.caption(f"📊 R² Score: **{config.get('r2', 0):.3f}**")
                    st.caption(f"📐 Features: {config.get('predictors_name', 'N/A')}")
            
            st.success("✅ Predictions complete!")
            
            # Range check
            st.divider()
            cond_min = st.session_state.data['Conductivity_uS_cm'].min()
            cond_max = st.session_state.data['Conductivity_uS_cm'].max()
            ph_min = st.session_state.data['pH'].min()
            ph_max = st.session_state.data['pH'].max()
            
            in_range = (cond_min <= pred_conductivity <= cond_max and ph_min <= pred_ph <= ph_max)
            
            if in_range:
                st.info(f"✅ Input within training range")
            else:
                st.warning(f"⚠️ Input outside training range - predictions may be less accurate")

# TAB 2: CORRELATIONS
with tab2:
    st.header("📈 Correlation Analysis")
    
    if st.session_state.correlations:
        # Heatmap
        st.subheader("Correlation Heatmap")
        
        numeric_cols = ['Conductivity_uS_cm', 'pH', 'Total_Hardness_ppm', 'Calculated_Hardness_ppm']
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
            textfont={"size": 12}
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        for target in ['Total_Hardness_ppm', 'Calculated_Hardness_ppm']:
            display_name = "Total Hardness" if "Total" in target else "Calcium Hardness"
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
                    st.info(f"**Best:** {best[0]} (r={best[1]:.3f}, {strength})")
                
                with col2:
                    fig = make_subplots(rows=1, cols=2, subplot_titles=('Conductivity', 'pH'))
                    
                    fig.add_trace(go.Scatter(
                        x=st.session_state.data['Conductivity_uS_cm'],
                        y=st.session_state.data[target],
                        mode='markers', name='Conductivity',
                        marker=dict(color='blue', size=10)
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=st.session_state.data['pH'],
                        y=st.session_state.data[target],
                        mode='markers', name='pH',
                        marker=dict(color='green', size=10)
                    ), row=1, col=2)
                    
                    fig.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            st.divider()

# TAB 3: MODELS
with tab3:
    st.header("🤖 Model Performance")
    
    # Metrics explanation
    with st.expander("📚 What do these metrics mean?"):
        st.markdown("""
        | Metric | Name | Meaning | Good Value |
        |--------|------|---------|------------|
        | **R²** | Coefficient of Determination | How well the model explains variance (0-1) | > 0.7 good, > 0.9 excellent |
        | **RMSE** | Root Mean Square Error | Average error in ppm (penalizes large errors) | Lower is better |
        | **MAE** | Mean Absolute Error | Average absolute error in ppm | Lower is better |
        
        **Example:** R²=0.85 means the model explains 85% of the variation in hardness values.
        """)
    
    if st.session_state.models and st.session_state.best_configs:
        for target, model_data in st.session_state.models.items():
            display_name = "Total Hardness" if "Total" in target else "Calcium Hardness"
            config = st.session_state.best_configs.get(target, {})
            
            st.subheader(f"📊 {display_name}")
            
            # Algorithm info
            algo = config.get('algorithm', 'N/A')
            if algo in ALGORITHM_INFO:
                with st.expander(f"ℹ️ About {algo}"):
                    info = ALGORITHM_INFO[algo]
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown(f"**✅ Pros:** {info['pros']}")
                    st.markdown(f"**❌ Cons:** {info['cons']}")
                    st.markdown(f"**🎯 Best for:** {info['best_for']}")
            
            # Metrics
            cols = st.columns(5)
            with cols[0]:
                st.metric("Algorithm", algo)
            with cols[1]:
                r2_val = model_data['metrics']['r2']
                st.metric("R² Score", f"{r2_val:.4f}", 
                         delta="Good" if r2_val > 0.7 else "Fair" if r2_val > 0.5 else "Poor")
            with cols[2]:
                st.metric("RMSE", f"{model_data['metrics']['rmse']:.1f} ppm")
            with cols[3]:
                st.metric("MAE", f"{model_data['metrics']['mae']:.1f} ppm")
            with cols[4]:
                st.metric("Samples", model_data['metrics']['n_samples'])
            
            st.info(f"**Features:** {config.get('predictors_name', 'N/A')}")
            
            # Plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                min_val = min(model_data['y_true'].min(), model_data['y_pred'].min())
                max_val = max(model_data['y_true'].max(), model_data['y_pred'].max())
                
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                        mode='lines', name='Perfect', line=dict(dash='dash', color='red')))
                fig.add_trace(go.Scatter(x=model_data['y_true'], y=model_data['y_pred'],
                                        mode='markers', name='Predictions', marker=dict(size=12, color='blue')))
                
                fig.update_layout(title="Predicted vs Actual", xaxis_title="Actual (ppm)", 
                                yaxis_title="Predicted (ppm)", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                residuals = model_data['y_true'] - model_data['y_pred']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=model_data['y_pred'], y=residuals,
                                        mode='markers', marker=dict(size=12, color='blue')))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                
                fig.update_layout(title="Residuals (Errors)", xaxis_title="Predicted (ppm)",
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
    st.header("📊 Data")
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("Records", len(st.session_state.data))
    with cols[1]:
        st.metric("Avg Total Hardness", f"{st.session_state.data['Total_Hardness_ppm'].mean():.0f}")
    with cols[2]:
        st.metric("Avg Calcium Hardness", f"{st.session_state.data['Calculated_Hardness_ppm'].mean():.0f}")
    with cols[3]:
        st.metric("Avg Conductivity", f"{st.session_state.data['Conductivity_uS_cm'].mean():.0f}")
    
    st.dataframe(st.session_state.data, use_container_width=True, height=400)
    
    csv = st.session_state.data.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, f"water_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    st.divider()
    st.subheader("⚠️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Delete Last Row"):
            if len(st.session_state.data) > 1:
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
    
    with col2:
        if st.button("🔄 Reset to Original"):
            st.session_state.data = get_initial_dataframe()
            save_data(st.session_state.data)
            
            correlations, best_configs, models = run_full_analysis(st.session_state.data)
            st.session_state.correlations = correlations
            st.session_state.best_configs = best_configs
            st.session_state.models = models
            save_models(models)
            save_configs(best_configs)
            
            st.success("Reset to original 14 records!")
            st.rerun()

# TAB 5: COMPARE ALL
with tab5:
    st.header("⚙️ Compare All Configurations")
    
    st.markdown("Compare all combinations of **predictors** and **algorithms** to find the best configuration.")
    
    if st.button("🔍 Run Full Comparison", type="primary"):
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
                
                # Visual comparison
                fig = go.Figure()
                for pred_name in predictor_options.keys():
                    subset = results_df[results_df['Predictors'] == pred_name]
                    fig.add_trace(go.Bar(x=subset['Algorithm'], y=subset['R² Score'], name=pred_name))
                
                fig.update_layout(title=f"R² Score Comparison - {display_name}", 
                                barmode='group', height=400,
                                yaxis_title="R² Score")
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()

# TAB 6: HELP
with tab6:
    st.header("📚 Help & Documentation")
    
    st.subheader("📊 Understanding the Metrics")
    
    st.markdown("""
    ### R² Score (Coefficient of Determination)
    - **Range:** 0 to 1 (can be negative for very poor models)
    - **Meaning:** Proportion of variance in the target explained by the model
    - **Interpretation:**
        - **> 0.9:** Excellent - model explains most variation
        - **0.7 - 0.9:** Good - model is useful for predictions
        - **0.5 - 0.7:** Moderate - model has some predictive power
        - **< 0.5:** Poor - model needs improvement
    
    ### RMSE (Root Mean Square Error)
    - **Units:** Same as target (ppm for hardness)
    - **Meaning:** Average prediction error, with larger errors penalized more
    - **Interpretation:** If RMSE = 50, expect predictions to be off by ~50 ppm on average
    - **Lower is better**
    
    ### MAE (Mean Absolute Error)
    - **Units:** Same as target (ppm for hardness)
    - **Meaning:** Average absolute difference between predicted and actual
    - **Interpretation:** More intuitive than RMSE - direct average error
    - **Lower is better**
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
    
    st.subheader("🔄 How to Use This App")
    
    st.markdown("""
    1. **Make Predictions:** Enter Conductivity and pH values to predict hardness
    2. **View Correlations:** See how predictors relate to hardness values
    3. **Check Model Performance:** Review accuracy metrics and visualizations
    4. **Compare Algorithms:** Run full comparison to find the best configuration
    5. **Select Algorithm:** Use the sidebar to choose a specific algorithm
    6. **Add Data:** Add new measurements to improve model accuracy over time
    """)

# Footer
st.divider()
st.caption("💧 Water Hardness Prediction | User-Selectable Algorithms | Auto-Training | Persistent Data")
