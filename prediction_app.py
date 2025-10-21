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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_DIR = 'models'
DATA_FILE = 'training_data.csv'

class WaterQualityPredictor:
    def __init__(self):
        self.models = {}  # Will store models for each target
        self.scalers = {}
        self.feature_configs = {}  # Best feature combination for each target
        self.model_names = {}
        
    def calculate_correlations(self, df, predictors, target):
        """Calculate correlations between predictors and target"""
        correlations = {}
        for predictor in predictors:
            if predictor in df.columns and target in df.columns:
                valid_data = df[[predictor, target]].dropna()
                if len(valid_data) > 1:
                    corr = valid_data[predictor].corr(valid_data[target])
                    correlations[predictor] = corr
        return correlations
    
    def create_engineered_features(self, df, feature_combo):
        """Create engineered features based on selected combination"""
        X = df[feature_combo].copy()
        
        # Add polynomial features
        if len(feature_combo) == 2:
            # Interaction term
            X[f'{feature_combo[0]}_x_{feature_combo[1]}'] = df[feature_combo[0]] * df[feature_combo[1]]
            # Ratio
            X[f'{feature_combo[0]}_div_{feature_combo[1]}'] = df[feature_combo[0]] / (df[feature_combo[1]] + 1e-6)
        
        # Add squared terms
        for col in feature_combo:
            X[f'{col}_squared'] = df[col] ** 2
            if df[col].min() > 0:
                X[f'{col}_log'] = np.log(df[col] + 1)
        
        return X
    
    def test_all_combinations(self, df, target, available_predictors):
        """Test all combinations of predictors to find the best"""
        from itertools import combinations
        
        results = []
        
        # Test individual predictors
        for predictor in available_predictors:
            if predictor in df.columns:
                score, model_name = self._test_combination(df, [predictor], target)
                results.append({
                    'features': [predictor],
                    'score': score,
                    'model': model_name,
                    'n_features': 1
                })
        
        # Test pairs of predictors
        if len(available_predictors) >= 2:
            for combo in combinations(available_predictors, 2):
                if all(c in df.columns for c in combo):
                    score, model_name = self._test_combination(df, list(combo), target)
                    results.append({
                        'features': list(combo),
                        'score': score,
                        'model': model_name,
                        'n_features': 2
                    })
        
        # Test all predictors together
        if len(available_predictors) >= 3:
            valid_predictors = [p for p in available_predictors if p in df.columns]
            if len(valid_predictors) >= 3:
                score, model_name = self._test_combination(df, valid_predictors, target)
                results.append({
                    'features': valid_predictors,
                    'score': score,
                    'model': model_name,
                    'n_features': len(valid_predictors)
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def _test_combination(self, df, features, target):
        """Test a specific combination of features"""
        # Filter valid data
        valid_cols = features + [target]
        valid_data = df[valid_cols].dropna()
        
        if len(valid_data) < 3:
            return -999, 'Insufficient Data'
        
        # Create features
        X = self.create_engineered_features(valid_data, features)
        y = valid_data[target].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Test multiple algorithms
        algorithms = {
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'SVR': SVR(kernel='rbf'),
            'Linear': LinearRegression()
        }
        
        best_score = -999
        best_model_name = 'None'
        
        for name, model in algorithms.items():
            try:
                if len(valid_data) < 5:
                    # Too few points for cross-validation
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_scaled)
                    score = r2_score(y, y_pred)
                else:
                    # Use cross-validation
                    scores = cross_val_score(model, X_scaled, y, cv=min(5, len(valid_data)), 
                                           scoring='r2', n_jobs=-1)
                    score = np.mean(scores)
                
                if score > best_score:
                    best_score = score
                    best_model_name = name
            except:
                continue
        
        return best_score, best_model_name
    
    def train_model(self, df, target, feature_combo, algorithm='Random Forest'):
        """Train the final model with best configuration"""
        # Filter valid data
        valid_cols = feature_combo + [target]
        valid_data = df[valid_cols].dropna()
        
        if len(valid_data) < 2:
            return None, None, "Insufficient data"
        
        # Create features
        X = self.create_engineered_features(valid_data, feature_combo)
        y = valid_data[target].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Select algorithm
        algorithms = {
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
            'Linear': LinearRegression()
        }
        
        model = algorithms.get(algorithm, RandomForestRegressor(n_estimators=200, random_state=42))
        
        # Train
        model.fit(X_scaled, y)
        
        # Calculate metrics
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # Store
        self.models[target] = model
        self.scalers[target] = scaler
        self.feature_configs[target] = feature_combo
        self.model_names[target] = algorithm
        
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(valid_data)
        }
        
        return model, scaler, metrics
    
    def predict(self, input_data, target):
        """Make prediction for a target variable"""
        if target not in self.models:
            return None
        
        feature_combo = self.feature_configs[target]
        
        # Create input dataframe
        input_df = pd.DataFrame([input_data])
        
        # Create engineered features
        X = self.create_engineered_features(input_df, feature_combo)
        
        # Scale
        X_scaled = self.scalers[target].transform(X)
        
        # Predict
        prediction = self.models[target].predict(X_scaled)
        
        return prediction[0]

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = WaterQualityPredictor()
    st.session_state.training_data = pd.DataFrame()
    st.session_state.correlation_results = {}
    st.session_state.combination_results = {}

def load_data_from_upload(uploaded_file):
    """Load data from uploaded CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Try to standardize column names
        column_mapping = {
            'conductivity': 'EC',
            'conductivity (µs/cm)': 'EC',
            'conductivity (us/cm)': 'EC',
            'ec': 'EC',
            'ph': 'pH',
            'total hardness': 'Total_Hardness',
            'total hardness (ppm)': 'Total_Hardness',
            'total_hardness': 'Total_Hardness',
            'calcium hardness': 'Calcium_Hardness',
            'calcium hardness (ppm)': 'Calcium_Hardness',
            'calcium_hardness': 'Calcium_Hardness',
            'calculated hardness (ppm)': 'Calcium_Hardness',
        }
        
        # Rename columns
        df.columns = df.columns.str.strip().str.lower()
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def save_models():
    """Save all trained models"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    try:
        with open(os.path.join(MODEL_DIR, 'predictor.pkl'), 'wb') as f:
            pickle.dump(st.session_state.predictor, f)
        if not st.session_state.training_data.empty:
            st.session_state.training_data.to_csv(DATA_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving models: {e}")

def load_models():
    """Load previously saved models"""
    try:
        predictor_path = os.path.join(MODEL_DIR, 'predictor.pkl')
        if os.path.exists(predictor_path):
            with open(predictor_path, 'rb') as f:
                st.session_state.predictor = pickle.load(f)
        
        if os.path.exists(DATA_FILE):
            st.session_state.training_data = pd.read_csv(DATA_FILE)
    except Exception as e:
        st.error(f"Error loading models: {e}")

# Main App
def main():
    st.set_page_config(page_title="Water Quality Predictor", page_icon="💧", layout="wide")
    
    st.title("💧 Advanced Water Quality Prediction System")
    st.markdown("**Predict Total Hardness & Calcium Hardness from pH, EC, or Both - Automatic Best Model Selection**")
    
    # Load existing models
    load_models()
    
    # Sidebar
    st.sidebar.header("📊 Data Management")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Water Testing Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        df = load_data_from_upload(uploaded_file)
        if df is not None:
            st.session_state.training_data = df
            st.sidebar.success(f"✅ Loaded {len(df)} records")
            
            # Show detected columns
            st.sidebar.info(f"Detected columns: {', '.join(df.columns.tolist())}")
    
    # Algorithm selection
    st.sidebar.header("🤖 Model Configuration")
    algorithm_choice = st.sidebar.selectbox(
        "Preferred Algorithm",
        ['Random Forest', 'Gradient Boosting', 'Ridge', 'SVR', 'Lasso', 'Linear'],
        index=0
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Correlation Analysis", 
        "🎯 Model Training", 
        "🔮 Predictions",
        "📊 Performance",
        "💾 Data View"
    ])
    
    with tab1:
        st.header("Correlation Analysis")
        
        if not st.session_state.training_data.empty:
            df = st.session_state.training_data
            
            # Identify available columns
            potential_predictors = ['EC', 'pH', 'Conductivity (µS/cm)', 'conductivity', 'ph']
            potential_targets = ['Total_Hardness', 'Calcium_Hardness', 'Total Hardness (ppm)', 'Calculated Hardness (ppm)']
            
            available_predictors = [col for col in df.columns if any(p.lower() in col.lower() for p in ['ec', 'ph', 'conductivity'])]
            available_targets = [col for col in df.columns if any(t.lower() in col.lower() for t in ['hardness', 'calcium'])]
            
            st.subheader("Available Variables")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Predictors:**", available_predictors)
            with col2:
                st.write("**Targets:**", available_targets)
            
            if len(available_predictors) > 0 and len(available_targets) > 0:
                # Calculate correlations
                st.subheader("Correlation Matrix")
                
                all_numeric_cols = available_predictors + available_targets
                numeric_df = df[all_numeric_cols].select_dtypes(include=[np.number])
                
                if not numeric_df.empty:
                    corr_matrix = numeric_df.corr()
                    
                    fig = px.imshow(corr_matrix, 
                                   text_auto='.3f',
                                   aspect="auto",
                                   color_continuous_scale='RdBu_r',
                                   title="Correlation Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Individual correlations for each target
                    for target in available_targets:
                        if target in df.columns:
                            st.subheader(f"Correlations with {target}")
                            correlations = st.session_state.predictor.calculate_correlations(
                                df, available_predictors, target
                            )
                            
                            if correlations:
                                # Create bar chart
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=list(correlations.keys()),
                                        y=list(correlations.values()),
                                        text=[f"{v:.3f}" for v in correlations.values()],
                                        textposition='auto',
                                    )
                                ])
                                fig.update_layout(
                                    title=f"Correlation with {target}",
                                    xaxis_title="Predictor",
                                    yaxis_title="Correlation Coefficient",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Interpretation
                                best_predictor = max(correlations.items(), key=lambda x: abs(x[1]))
                                st.info(f"🎯 **Best single predictor:** {best_predictor[0]} (r = {best_predictor[1]:.3f})")
            else:
                st.warning("Please ensure your data has both predictors (EC, pH) and targets (Total Hardness, Calcium Hardness)")
        else:
            st.info("👆 Upload your water testing data to begin correlation analysis")
    
    with tab2:
        st.header("Model Training & Optimization")
        
        if not st.session_state.training_data.empty:
            df = st.session_state.training_data
            
            # Detect columns
            available_predictors = [col for col in df.columns if any(p.lower() in col.lower() for p in ['ec', 'ph', 'conductivity'])]
            available_targets = [col for col in df.columns if any(t.lower() in col.lower() for t in ['hardness', 'calcium'])]
            
            if available_predictors and available_targets:
                st.subheader("🔍 Test All Feature Combinations")
                
                if st.button("🚀 Find Best Model Configuration", type="primary"):
                    results_dict = {}
                    
                    for target in available_targets:
                        with st.spinner(f"Testing combinations for {target}..."):
                            results = st.session_state.predictor.test_all_combinations(
                                df, target, available_predictors
                            )
                            results_dict[target] = results
                    
                    st.session_state.combination_results = results_dict
                    st.success("✅ Analysis complete!")
                
                # Display results
                if st.session_state.combination_results:
                    for target, results in st.session_state.combination_results.items():
                        st.subheader(f"📊 Results for {target}")
                        
                        if results:
                            # Create results dataframe
                            results_df = pd.DataFrame([
                                {
                                    'Features': ' + '.join(r['features']),
                                    'R² Score': f"{r['score']:.4f}",
                                    'Best Model': r['model'],
                                    'N Features': r['n_features']
                                }
                                for r in results[:10]  # Top 10
                            ])
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Highlight best
                            best = results[0]
                            st.success(f"🏆 **Best Configuration:** {' + '.join(best['features'])} using {best['model']} (R² = {best['score']:.4f})")
                
                st.divider()
                st.subheader("🎯 Train Final Models")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    train_total = st.checkbox("Train Total Hardness Model", value=True)
                with col2:
                    train_calcium = st.checkbox("Train Calcium Hardness Model", value=True)
                
                if st.button("⚡ Train Selected Models"):
                    trained_models = []
                    
                    for target in available_targets:
                        should_train = (train_total and 'total' in target.lower()) or \
                                     (train_calcium and 'calcium' in target.lower())
                        
                        if should_train:
                            # Use best combination if available
                            if target in st.session_state.combination_results and st.session_state.combination_results[target]:
                                best_config = st.session_state.combination_results[target][0]
                                features = best_config['features']
                                best_algorithm = best_config['model']
                            else:
                                features = available_predictors
                                best_algorithm = algorithm_choice
                            
                            with st.spinner(f"Training {target} model..."):
                                model, scaler, metrics = st.session_state.predictor.train_model(
                                    df, target, features, best_algorithm
                                )
                                
                                if model is not None:
                                    trained_models.append({
                                        'target': target,
                                        'features': features,
                                        'algorithm': best_algorithm,
                                        'metrics': metrics
                                    })
                    
                    # Save models
                    save_models()
                    
                    # Display results
                    if trained_models:
                        st.success(f"✅ Successfully trained {len(trained_models)} model(s)")
                        
                        for result in trained_models:
                            with st.expander(f"📈 {result['target']} - {result['algorithm']}"):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("R² Score", f"{result['metrics']['r2']:.3f}")
                                with col2:
                                    st.metric("RMSE", f"{result['metrics']['rmse']:.1f}")
                                with col3:
                                    st.metric("MAE", f"{result['metrics']['mae']:.1f}")
                                with col4:
                                    st.metric("Samples", result['metrics']['n_samples'])
                                
                                st.write(f"**Features used:** {', '.join(result['features'])}")
            else:
                st.warning("Could not detect predictor or target columns. Please check your data format.")
        else:
            st.info("👆 Upload data first to train models")
    
    with tab3:
        st.header("Make Predictions")
        
        if st.session_state.predictor.models:
            st.subheader("Enter Water Quality Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ec_value = st.number_input("EC / Conductivity (µS/cm)", 
                                          min_value=0.0, 
                                          max_value=20000.0, 
                                          value=4500.0,
                                          step=50.0)
                
                ph_value = st.number_input("pH", 
                                          min_value=0.0, 
                                          max_value=14.0, 
                                          value=7.5,
                                          step=0.1)
            
            with col2:
                if st.button("🔮 Predict All Parameters", type="primary"):
                    input_data = {
                        'EC': ec_value,
                        'pH': ph_value,
                        'Conductivity (µS/cm)': ec_value,
                        'conductivity': ec_value,
                        'ph': ph_value
                    }
                    
                    predictions = {}
                    
                    for target in st.session_state.predictor.models.keys():
                        try:
                            pred = st.session_state.predictor.predict(input_data, target)
                            if pred is not None:
                                predictions[target] = pred
                        except Exception as e:
                            st.error(f"Error predicting {target}: {e}")
                    
                    if predictions:
                        st.success("✅ Predictions Complete!")
                        
                        for target, value in predictions.items():
                            features_used = st.session_state.predictor.feature_configs.get(target, [])
                            model_used = st.session_state.predictor.model_names.get(target, 'Unknown')
                            
                            st.metric(
                                label=f"**{target}**",
                                value=f"{value:.1f} ppm",
                                help=f"Model: {model_used} | Features: {', '.join(features_used)}"
                            )
                    else:
                        st.error("No predictions could be made")
            
            # Add actual measurement for retraining
            st.divider()
            st.subheader("📝 Add Actual Measurement for Continuous Learning")
            
            with st.expander("Provide Feedback"):
                feedback_target = st.selectbox("Which parameter?", 
                                              list(st.session_state.predictor.models.keys()))
                actual_value = st.number_input("Actual measured value (ppm)", 
                                              min_value=0.0, 
                                              value=1400.0)
                
                if st.button("✅ Add to Training Data"):
                    new_row = {
                        'EC': ec_value,
                        'pH': ph_value,
                        'Conductivity (µS/cm)': ec_value,
                        feedback_target: actual_value,
                        'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    st.session_state.training_data = pd.concat([
                        st.session_state.training_data,
                        pd.DataFrame([new_row])
                    ], ignore_index=True)
                    
                    # Retrain
                    features = st.session_state.predictor.feature_configs.get(feedback_target, ['EC', 'pH'])
                    algorithm = st.session_state.predictor.model_names.get(feedback_target, 'Random Forest')
                    
                    with st.spinner("Retraining model..."):
                        st.session_state.predictor.train_model(
                            st.session_state.training_data,
                            feedback_target,
                            features,
                            algorithm
                        )
                        save_models()
                    
                    st.success("✅ Model updated with new data!")
                    st.balloons()
        else:
            st.info("👈 Train models first in the 'Model Training' tab")
    
    with tab4:
        st.header("Model Performance Analysis")
        
        if st.session_state.predictor.models:
            for target, model in st.session_state.predictor.models.items():
                st.subheader(f"📊 {target}")
                
                # Get data for this target
                df = st.session_state.training_data
                features = st.session_state.predictor.feature_configs[target]
                
                valid_cols = features + [target]
                valid_data = df[valid_cols].dropna()
                
                if len(valid_data) > 0:
                    # Make predictions
                    X = st.session_state.predictor.create_engineered_features(valid_data, features)
                    X_scaled = st.session_state.predictor.scalers[target].transform(X)
                    
                    y_true = valid_data[target].values
                    y_pred = model.predict(X_scaled)
                    
                    # Metrics
                    r2 = r2_score(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mae = mean_absolute_error(y_true, y_pred)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Model", st.session_state.predictor.model_names[target])
                    with col2:
                        st.metric("R² Score", f"{r2:.3f}")
                    with col3:
                        st.metric("RMSE", f"{rmse:.1f} ppm")
                    with col4:
                        st.metric("MAE", f"{mae:.1f} ppm")
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Predicted vs Actual
                        fig = go.Figure()
                        
                        min_val = min(min(y_true), min(y_pred))
                        max_val = max(max(y_true), max(y_pred))
                        fig.add_trace(go.Scatter(x=[min_val, max_val], 
                                               y=[min_val, max_val],
                                               mode='lines', 
                                               name='Perfect',
                                               line=dict(dash='dash', color='red')))
                        
                        fig.add_trace(go.Scatter(x=y_true, 
                                               y=y_pred,
                                               mode='markers',
                                               name='Predictions',
                                               marker=dict(size=10)))
                        
                        fig.update_layout(
                            title="Predicted vs Actual",
                            xaxis_title="Actual (ppm)",
                            yaxis_title="Predicted (ppm)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Residuals
                        residuals = y_true - y_pred
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=y_pred, 
                                               y=residuals,
                                               mode='markers',
                                               marker=dict(size=10)))
                        fig.add_hline(y=0, line_dash="dash", line_color="red")
                        
                        fig.update_layout(
                            title="Residuals Plot",
                            xaxis_title="Predicted (ppm)",
                            yaxis_title="Residuals (ppm)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance for Random Forest
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("🎯 Feature Importance")
                        
                        feature_names = X.columns.tolist()
                        importances = model.feature_importances_
                        
                        # Sort by importance
                        indices = np.argsort(importances)[::-1][:10]  # Top 10
                        
                        fig = go.Figure(data=[
                            go.Bar(x=[feature_names[i] for i in indices],
                                  y=[importances[i] for i in indices])
                        ])
                        fig.update_layout(
                            title="Top 10 Most Important Features",
                            xaxis_title="Feature",
                            yaxis_title="Importance",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.write(f"**Features used:** {', '.join(features)}")
                    
                st.divider()
        else:
            st.info("👈 Train models first to see performance metrics")
    
    with tab5:
        st.header("Training Data")
        
        if not st.session_state.training_data.empty:
            st.dataframe(st.session_state.training_data, use_container_width=True)
            
            # Download button
            csv = st.session_state.training_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Data",
                data=csv,
                file_name=f"water_quality_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            st.dataframe(st.session_state.training_data.describe(), use_container_width=True)
        else:
            st.info("No data loaded yet")

if __name__ == "__main__":
    main()
