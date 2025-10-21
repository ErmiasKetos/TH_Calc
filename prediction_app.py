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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Water Hardness Predictor", page_icon="💧", layout="wide")

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'best_configs' not in st.session_state:
    st.session_state.best_configs = {}
if 'correlation_analysis' not in st.session_state:
    st.session_state.correlation_analysis = {}

def load_data(uploaded_file):
    """Load and process uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert numeric columns
        numeric_cols = ['Calculated Hardness (ppm)', 'Total Hardness (ppm)', 'Conductivity (µS/cm)', 
                       'pH', 'Chloride (ppm)', 'Alkalinity (ppm)', 'Turbidity (FAU)', 
                       'Aluminum (ppm)', 'Temp (°C)', 'Biocide (ppm)', 'Sulfates (ppm)', 
                       'Iron (ppm)', 'Antiscalant (PTSA ppb)', 'Antiscalant (>0.1)']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def calculate_correlations(df, predictors, target):
    """Calculate correlation coefficients"""
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
    """Create engineered features"""
    X = df[predictor_cols].copy()
    
    # Polynomial features
    if len(predictor_cols) >= 2:
        X['interaction'] = df[predictor_cols[0]] * df[predictor_cols[1]]
        X['ratio'] = df[predictor_cols[0]] / (df[predictor_cols[1]] + 1e-6)
    
    # Add squared and log terms
    for col in predictor_cols:
        X[f'{col}_squared'] = df[col] ** 2
        if df[col].min() > 0:
            X[f'{col}_log'] = np.log(df[col] + 1)
    
    return X

def evaluate_model_combination(df, predictors, target, algorithm):
    """Evaluate a specific predictor combination with an algorithm"""
    
    # Filter valid data
    required_cols = predictors + [target]
    valid_data = df[required_cols].dropna()
    
    if len(valid_data) < 3:
        return None
    
    # Create features
    X = create_features(valid_data, predictors)
    y = valid_data[target].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define model
    models_dict = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_split=2, 
                                               min_samples_leaf=1, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, 
                                                       max_depth=4, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
        'Linear': LinearRegression()
    }
    
    model = models_dict.get(algorithm)
    if model is None:
        return None
    
    try:
        # Use Leave-One-Out cross-validation for small datasets
        if len(valid_data) < 10:
            loo = LeaveOneOut()
            predictions = []
            actuals = []
            
            for train_idx, test_idx in loo.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model_clone = models_dict.get(algorithm)
                model_clone.fit(X_train, y_train)
                pred = model_clone.predict(X_test)
                
                predictions.append(pred[0])
                actuals.append(y_test[0])
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            r2 = r2_score(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
        else:
            # Regular cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            r2 = np.mean(cv_scores)
            
            # Fit for RMSE and MAE
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(valid_data),
            'model': model,
            'scaler': scaler,
            'feature_names': X.columns.tolist()
        }
    
    except Exception as e:
        return None

def train_final_model(df, predictors, target, algorithm):
    """Train the final model on all data"""
    required_cols = predictors + [target]
    valid_data = df[required_cols].dropna()
    
    if len(valid_data) < 2:
        return None
    
    # Create features
    X = create_features(valid_data, predictors)
    y = valid_data[target].values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define and train model
    models_dict = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
        'Linear': LinearRegression()
    }
    
    model = models_dict.get(algorithm)
    if model is None:
        return None
    
    model.fit(X_scaled, y)
    
    # Calculate training metrics
    y_pred = model.predict(X_scaled)
    metrics = {
        'r2': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'n_samples': len(valid_data)
    }
    
    return {
        'model': model,
        'scaler': scaler,
        'predictors': predictors,
        'feature_names': X.columns.tolist(),
        'metrics': metrics,
        'y_true': y,
        'y_pred': y_pred
    }

# ========== MAIN APP ==========

st.title("💧 Water Hardness Prediction System")
st.markdown("**Comprehensive ML System for Total & Calcium Hardness Prediction**")

# Sidebar
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Water Testing CSV", type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.session_state.data = df
        st.sidebar.success(f"✅ Loaded {len(df)} records")
        st.sidebar.info(f"Columns: {len(df.columns)}")

# Main content
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Define predictor and target options
    predictor_options = {
        'Conductivity (µS/cm)': 'Conductivity (µS/cm)',
        'pH': 'pH',
        'Both (Conductivity + pH)': ['Conductivity (µS/cm)', 'pH']
    }
    
    target_options = {
        'Total Hardness (ppm)': 'Total Hardness (ppm)',
        'Calculated Hardness (ppm)': 'Calculated Hardness (ppm)',
        'Both': ['Total Hardness (ppm)', 'Calculated Hardness (ppm)']
    }
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🔍 Correlation Analysis",
        "🤖 Model Comparison",
        "🎯 Best Models & Training",
        "🔮 Predictions"
    ])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Hardness Avg", f"{df['Total Hardness (ppm)'].mean():.0f} ppm" if 'Total Hardness (ppm)' in df.columns else "N/A")
        with col3:
            st.metric("Calcium Hardness Avg", f"{df['Calculated Hardness (ppm)'].mean():.0f} ppm" if 'Calculated Hardness (ppm)' in df.columns else "N/A")
        with col4:
            st.metric("pH Range", f"{df['pH'].min():.2f} - {df['pH'].max():.2f}" if 'pH' in df.columns else "N/A")
        
        st.subheader("Raw Data")
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Distribution plots
        st.subheader("Data Distributions")
        
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=('Total Hardness', 'Calculated Hardness (Calcium)', 
                                         'Conductivity', 'pH'))
        
        if 'Total Hardness (ppm)' in df.columns:
            fig.add_trace(go.Histogram(x=df['Total Hardness (ppm)'], name='Total Hardness'), row=1, col=1)
        
        if 'Calculated Hardness (ppm)' in df.columns:
            fig.add_trace(go.Histogram(x=df['Calculated Hardness (ppm)'], name='Calcium Hardness'), row=1, col=2)
        
        if 'Conductivity (µS/cm)' in df.columns:
            fig.add_trace(go.Histogram(x=df['Conductivity (µS/cm)'], name='Conductivity'), row=2, col=1)
        
        if 'pH' in df.columns:
            fig.add_trace(go.Histogram(x=df['pH'], name='pH'), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Correlation Analysis")
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 2:
            st.subheader("Correlation Heatmap")
            
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix,
                           text_auto='.3f',
                           aspect='auto',
                           color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1)
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed correlation analysis
        st.subheader("Predictor vs Target Correlations")
        
        predictors = ['Conductivity (µS/cm)', 'pH']
        targets = ['Total Hardness (ppm)', 'Calculated Hardness (ppm)']
        
        for target in targets:
            if target in df.columns:
                st.markdown(f"### {target}")
                
                correlations, p_values = calculate_correlations(df, predictors, target)
                
                if correlations:
                    # Create bar chart
                    fig = go.Figure()
                    
                    colors = ['green' if abs(v) > 0.5 else 'orange' if abs(v) > 0.3 else 'red' 
                             for v in correlations.values()]
                    
                    fig.add_trace(go.Bar(
                        x=list(correlations.keys()),
                        y=list(correlations.values()),
                        text=[f"{v:.3f}<br>p={p_values.get(k, 0):.3f}" 
                              for k, v in correlations.items()],
                        textposition='auto',
                        marker_color=colors
                    ))
                    
                    fig.update_layout(
                        title=f"Correlation with {target}",
                        xaxis_title="Predictor",
                        yaxis_title="Pearson Correlation Coefficient",
                        height=400,
                        yaxis_range=[-1, 1]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    best_pred = max(correlations.items(), key=lambda x: abs(x[1]))
                    
                    if abs(best_pred[1]) > 0.7:
                        strength = "Strong"
                        color = "green"
                    elif abs(best_pred[1]) > 0.4:
                        strength = "Moderate"
                        color = "orange"
                    else:
                        strength = "Weak"
                        color = "red"
                    
                    st.markdown(f"**Best Predictor:** {best_pred[0]} | **Correlation:** {best_pred[1]:.3f} | **Strength:** :{color}[{strength}]")
                    
                    # Scatter plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'Conductivity (µS/cm)' in df.columns:
                            fig = px.scatter(df, x='Conductivity (µS/cm)', y=target,
                                           trendline="ols",
                                           title=f"Conductivity vs {target}")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'pH' in df.columns:
                            fig = px.scatter(df, x='pH', y=target,
                                           trendline="ols",
                                           title=f"pH vs {target}")
                            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Model Comparison")
        st.markdown("Test all predictor combinations and algorithms to find the best configuration")
        
        # Target selection
        target_choice = st.selectbox("Select Target Variable", list(target_options.keys()))
        
        if st.button("🚀 Run Comprehensive Analysis", type="primary"):
            targets_to_test = target_options[target_choice]
            if not isinstance(targets_to_test, list):
                targets_to_test = [targets_to_test]
            
            all_results = {}
            
            for target in targets_to_test:
                if target not in df.columns:
                    continue
                
                st.markdown(f"### Analyzing: {target}")
                
                results = []
                
                # Test all combinations
                progress_bar = st.progress(0)
                total_tests = len(predictor_options) * 6  # 6 algorithms
                current = 0
                
                for pred_name, pred_cols in predictor_options.items():
                    if not isinstance(pred_cols, list):
                        pred_cols = [pred_cols]
                    
                    for algorithm in ['Random Forest', 'Gradient Boosting', 'Ridge', 'Lasso', 'SVR', 'Linear']:
                        result = evaluate_model_combination(df, pred_cols, target, algorithm)
                        
                        if result is not None:
                            results.append({
                                'Predictors': pred_name,
                                'Algorithm': algorithm,
                                'R² Score': result['r2'],
                                'RMSE': result['rmse'],
                                'MAE': result['mae'],
                                'Samples': result['n_samples']
                            })
                        
                        current += 1
                        progress_bar.progress(current / total_tests)
                
                progress_bar.empty()
                
                if results:
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values('R² Score', ascending=False)
                    
                    all_results[target] = results_df
                    
                    # Display results
                    st.dataframe(results_df.style.background_gradient(subset=['R² Score'], cmap='RdYlGn')
                                             .format({'R² Score': '{:.4f}', 'RMSE': '{:.2f}', 'MAE': '{:.2f}'}),
                               use_container_width=True)
                    
                    # Highlight best
                    best = results_df.iloc[0]
                    st.success(f"🏆 **Best Configuration:** {best['Predictors']} + {best['Algorithm']} | R² = {best['R² Score']:.4f} | RMSE = {best['RMSE']:.2f}")
                    
                    # Visualization
                    fig = px.bar(results_df.head(10), 
                               x='Algorithm', y='R² Score',
                               color='Predictors',
                               barmode='group',
                               title=f"Top 10 Configurations for {target}")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store best config
                    st.session_state.best_configs[target] = {
                        'predictors': best['Predictors'],
                        'algorithm': best['Algorithm'],
                        'r2': best['R² Score']
                    }
    
    with tab4:
        st.header("Train Best Models")
        
        if st.session_state.best_configs:
            st.subheader("Recommended Configurations")
            
            for target, config in st.session_state.best_configs.items():
                st.info(f"**{target}**: {config['predictors']} + {config['algorithm']} (R² = {config['r2']:.4f})")
            
            st.divider()
            
            if st.button("🎯 Train All Best Models", type="primary"):
                for target, config in st.session_state.best_configs.items():
                    st.markdown(f"### Training: {target}")
                    
                    # Get predictor columns
                    pred_choice = config['predictors']
                    predictors = predictor_options[pred_choice]
                    if not isinstance(predictors, list):
                        predictors = [predictors]
                    
                    # Train
                    with st.spinner(f"Training {config['algorithm']}..."):
                        result = train_final_model(df, predictors, target, config['algorithm'])
                        
                        if result is not None:
                            st.session_state.models[target] = result
                            
                            # Display metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("R² Score", f"{result['metrics']['r2']:.4f}")
                            with col2:
                                st.metric("RMSE", f"{result['metrics']['rmse']:.2f} ppm")
                            with col3:
                                st.metric("MAE", f"{result['metrics']['mae']:.2f} ppm")
                            with col4:
                                st.metric("Samples", result['metrics']['n_samples'])
                            
                            # Predictions vs Actual plot
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = go.Figure()
                                
                                min_val = min(result['y_true'].min(), result['y_pred'].min())
                                max_val = max(result['y_true'].max(), result['y_pred'].max())
                                
                                fig.add_trace(go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode='lines',
                                    name='Perfect Prediction',
                                    line=dict(dash='dash', color='red')
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=result['y_true'],
                                    y=result['y_pred'],
                                    mode='markers',
                                    name='Predictions',
                                    marker=dict(size=12, color='blue')
                                ))
                                
                                fig.update_layout(
                                    title="Predicted vs Actual",
                                    xaxis_title="Actual (ppm)",
                                    yaxis_title="Predicted (ppm)",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                residuals = result['y_true'] - result['y_pred']
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=result['y_pred'],
                                    y=residuals,
                                    mode='markers',
                                    marker=dict(size=12, color='blue')
                                ))
                                fig.add_hline(y=0, line_dash="dash", line_color="red")
                                
                                fig.update_layout(
                                    title="Residuals Plot",
                                    xaxis_title="Predicted (ppm)",
                                    yaxis_title="Residuals (ppm)",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature importance for tree-based models
                            if hasattr(result['model'], 'feature_importances_'):
                                st.subheader("Feature Importance")
                                
                                importances = result['model'].feature_importances_
                                feature_names = result['feature_names']
                                
                                # Sort by importance
                                indices = np.argsort(importances)[::-1]
                                
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=[feature_names[i] for i in indices],
                                        y=[importances[i] for i in indices]
                                    )
                                ])
                                
                                fig.update_layout(
                                    title="Feature Importance",
                                    xaxis_title="Feature",
                                    yaxis_title="Importance",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.success(f"✅ Model trained successfully for {target}!")
                        else:
                            st.error(f"Failed to train model for {target}")
                
                st.balloons()
        else:
            st.info("👈 Run Model Comparison first to identify best configurations")
    
    with tab5:
        st.header("Make Predictions")
        
        if st.session_state.models:
            st.subheader("Input Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                conductivity_input = st.number_input(
                    "Conductivity (µS/cm)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=float(df['Conductivity (µS/cm)'].mean()) if 'Conductivity (µS/cm)' in df.columns else 4500.0,
                    step=50.0
                )
                
                ph_input = st.number_input(
                    "pH",
                    min_value=0.0,
                    max_value=14.0,
                    value=float(df['pH'].mean()) if 'pH' in df.columns else 7.5,
                    step=0.1
                )
            
            with col2:
                if st.button("🔮 Predict All Parameters", type="primary"):
                    st.subheader("Predictions")
                    
                    input_df = pd.DataFrame({
                        'Conductivity (µS/cm)': [conductivity_input],
                        'pH': [ph_input]
                    })
                    
                    for target, model_data in st.session_state.models.items():
                        # Create features
                        X = create_features(input_df, model_data['predictors'])
                        X_scaled = model_data['scaler'].transform(X)
                        
                        # Predict
                        prediction = model_data['model'].predict(X_scaled)[0]
                        
                        # Display
                        st.metric(
                            label=target,
                            value=f"{prediction:.1f} ppm",
                            help=f"Using: {', '.join(model_data['predictors'])}"
                        )
                    
                    st.success("✅ Predictions complete!")
            
            st.divider()
            
            # Add feedback
            st.subheader("📝 Add Actual Measurement (Continuous Learning)")
            
            with st.expander("Provide Feedback to Improve Models"):
                feedback_target = st.selectbox("Parameter", list(st.session_state.models.keys()))
                actual_value = st.number_input("Actual Measured Value (ppm)", min_value=0.0, value=1400.0)
                
                if st.button("✅ Add to Training Data"):
                    new_row = {
                        'Date': datetime.now().strftime('%Y-%m-%d'),
                        'Conductivity (µS/cm)': conductivity_input,
                        'pH': ph_input,
                        feedback_target: actual_value
                    }
                    
                    st.session_state.data = pd.concat([
                        st.session_state.data,
                        pd.DataFrame([new_row])
                    ], ignore_index=True)
                    
                    st.success("✅ Data added! Retrain models to incorporate new data.")
                    st.balloons()
        else:
            st.info("👈 Train models first in the 'Best Models & Training' tab")

else:
    st.info("👈 Please upload your water testing CSV file to begin")
    
    st.markdown("""
    ### Expected CSV Format:
    
    Your CSV should contain columns like:
    - `Conductivity (µS/cm)` or `EC` - Electrical Conductivity
    - `pH` - pH value
    - `Total Hardness (ppm)` - Target variable 1
    - `Calculated Hardness (ppm)` - Target variable 2 (Calcium Hardness)
    
    The system will automatically:
    1. ✅ Analyze correlations between all predictors and targets
    2. ✅ Test multiple ML algorithms (Random Forest, Gradient Boosting, Ridge, etc.)
    3. ✅ Test all predictor combinations (pH only, EC only, pH+EC)
    4. ✅ Recommend the best configuration for each target
    5. ✅ Train optimized models
    6. ✅ Provide predictions with confidence metrics
    """)
