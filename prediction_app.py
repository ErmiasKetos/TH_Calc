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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_FILE = 'hardness_model.pkl'
SCALER_FILE = 'scaler.pkl'
DATA_FILE = 'training_data.csv'
POLY_FILE = 'poly_features.pkl'

class WaterHardnessPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.feature_names = ['Conductivity', 'pH']
        self.use_scaling = False  # Track whether current model needs scaling
        
    def create_features(self, conductivity, ph):
        """Create engineered features from conductivity and pH"""
        # Basic features
        features = np.array([[conductivity, ph]])
        
        # Polynomial features
        poly_features = self.poly_features.fit_transform(features)
        
        # Additional engineered features
        conductivity_ph_ratio = conductivity / ph if ph != 0 else 0
        log_conductivity = np.log(conductivity) if conductivity > 0 else 0
        ph_squared = ph ** 2
        
        # Combine all features
        engineered = np.array([[
            conductivity, ph, 
            conductivity_ph_ratio, 
            log_conductivity, 
            ph_squared,
            conductivity * ph,  # interaction term
        ]])
        
        return engineered
    
    def train_model(self, X, y, preferred_model='Random Forest'):
        """Train the model with Random Forest as preferred algorithm"""
        
        # Scale features (needed for some algorithms)
        X_scaled = self.scaler.fit_transform(X)
        
        # Define models with Random Forest prioritized
        models = {
            'Random Forest': {
                'model': RandomForestRegressor(
                    n_estimators=200, 
                    max_depth=10,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaling': False  # Random Forest doesn't need scaling
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'use_scaling': False
            },
            'Ridge Regression': {
                'model': Ridge(alpha=1.0),
                'use_scaling': True
            },
            'Linear Regression': {
                'model': LinearRegression(),
                'use_scaling': True
            }
        }
        
        # Try preferred model first (Random Forest)
        if preferred_model in models:
            model_config = models[preferred_model]
            try:
                model = model_config['model']
                X_input = X_scaled if model_config['use_scaling'] else X
                
                # Cross-validation for preferred model
                scores = cross_val_score(model, X_input, y, cv=min(5, len(X)), 
                                       scoring='r2', n_jobs=-1)
                avg_score = np.mean(scores)
                
                # Train the preferred model
                self.model = model
                self.model.fit(X_input, y)
                self.use_scaling = model_config['use_scaling']
                
                return preferred_model, avg_score
                
            except Exception as e:
                st.warning(f"Error with {preferred_model}: {e}")
        
        # Test all other algorithms if preferred fails
        best_score = -np.inf
        best_model = None
        best_name = None
        best_use_scaling = True
        
        for name, config in models.items():
            if name == preferred_model:  # Skip already tried
                continue
                
            try:
                model = config['model']
                X_input = X_scaled if config['use_scaling'] else X
                
                # Cross-validation score
                scores = cross_val_score(model, X_input, y, cv=min(5, len(X)), 
                                       scoring='r2', n_jobs=-1)
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_name = name
                    best_use_scaling = config['use_scaling']
            except:
                continue
        
        # Train the best alternative model
        if best_model is not None:
            X_input = X_scaled if best_use_scaling else X
            self.model = best_model
            self.model.fit(X_input, y)
            self.use_scaling = best_use_scaling
            return best_name, best_score
        
        # Final fallback to Random Forest (simplified)
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.model.fit(X, y)
        self.use_scaling = False
        return "Random Forest (Fallback)", 0.0
    
    def predict(self, conductivity, ph):
        """Make a prediction"""
        if self.model is None:
            return None
        
        features = self.create_features(conductivity, ph)
        
        # Use scaling only if the current model requires it
        if hasattr(self, 'use_scaling') and self.use_scaling:
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)
        else:
            prediction = self.model.predict(features)
            
        return prediction[0]
    
    def retrain_with_new_data(self, new_data):
        """Retrain the model with new data points"""
        if len(new_data) == 0:
            return
        
        # Prepare features
        X = []
        y = []
        
        for _, row in new_data.iterrows():
            features = self.create_features(row['Conductivity'], row['pH'])
            X.append(features[0])
            y.append(row['Total_Hardness'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Retrain
        self.train_model(X, y)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = WaterHardnessPredictor()
    st.session_state.training_data = pd.DataFrame()
    st.session_state.predictions_made = []

def load_initial_data():
    """Load your initial water testing data"""
    # Your initial dataset (replace with actual data loading)
    initial_data = {
        'Date': ['23-06-25', '04-07-25', '09-07-25', '11-07-25', '16-07-25', 
                '18-07-25', '23-07-25', '25-07-25', '30-07-25', '01-08-25',
                '06-08-25', '08-08-25', '13-08-25', '15-08-25'],
        'Conductivity': [4520, 4230, 4311, 4920, 4600, 4850, 5490, 4750, 4380, 4950,
                        4650, 4420, 4780, 4560],
        'pH': [7.74, 8.14, 7.95, 7.88, 8.23, 7.68, 8.45, 7.92, 8.12, 8.76,
               8.34, 7.89, 8.01, 7.97],
        'Total_Hardness': [1590, 1390, 1460, 1520, 1340, 1580, 1190, 1470, 1410, 1230,
                          1350, 1490, 1440, 1380]
    }
    return pd.DataFrame(initial_data)

def save_model_and_data():
    """Save the trained model and accumulated data"""
    try:
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(st.session_state.predictor.model, f)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(st.session_state.predictor.scaler, f)
        with open(POLY_FILE, 'wb') as f:
            pickle.dump(st.session_state.predictor.poly_features, f)
        if not st.session_state.training_data.empty:
            st.session_state.training_data.to_csv(DATA_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving model: {e}")

def load_model_and_data():
    """Load previously saved model and data"""
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            with open(MODEL_FILE, 'rb') as f:
                st.session_state.predictor.model = pickle.load(f)
            with open(SCALER_FILE, 'rb') as f:
                st.session_state.predictor.scaler = pickle.load(f)
            if os.path.exists(POLY_FILE):
                with open(POLY_FILE, 'rb') as f:
                    st.session_state.predictor.poly_features = pickle.load(f)
        
        if os.path.exists(DATA_FILE):
            st.session_state.training_data = pd.read_csv(DATA_FILE)
        else:
            st.session_state.training_data = load_initial_data()
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.session_state.training_data = load_initial_data()

def train_initial_model(preferred_model='Random Forest'):
    """Train the model with initial data"""
    if st.session_state.training_data.empty:
        return
    
    # Prepare features
    X = []
    y = st.session_state.training_data['Total_Hardness'].values
    
    for _, row in st.session_state.training_data.iterrows():
        features = st.session_state.predictor.create_features(row['Conductivity'], row['pH'])
        X.append(features[0])
    
    X = np.array(X)
    
    # Train model with preferred algorithm
    model_name, score = st.session_state.predictor.train_model(X, y, preferred_model)
    
    return model_name, score

# Main Streamlit App
def main():
    st.set_page_config(page_title="Water Hardness Predictor", page_icon="💧", layout="wide")
    
    st.title("💧WHPS")
    st.markdown("**Predict TH from EC and pH with Continuous Learning**")
    
    # Load model and data
    load_model_and_data()
    
    # Sidebar for model management
    st.sidebar.header("🔧 Model Management")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "🎯 Preferred Algorithm",
        ['Random Forest', 'Gradient Boosting', 'Ridge Regression', 'Linear Regression'],
        index=0,  # Random Forest as default
        help="Random Forest is recommended for water quality prediction"
    )
    
    # Advanced Random Forest settings
    if model_choice == 'Random Forest':
        with st.sidebar.expander("🌲 Random Forest Settings"):
            n_estimators = st.slider("Number of Trees", 50, 500, 200, 50)
            max_depth = st.slider("Max Depth", 5, 20, 10, 1)
            
    # Initialize/retrain model
    if st.sidebar.button("🚀 Train/Retrain Model"):
        with st.spinner(f"Training {model_choice} model..."):
            model_name, score = train_initial_model(preferred_model=model_choice)
            save_model_and_data()
            st.sidebar.success(f"✅ Model trained: {model_name}")
            st.sidebar.info(f"Cross-validation R² Score: {score:.3f}")
            
            # Show why Random Forest is good for this task
            if model_name == 'Random Forest':
                st.sidebar.info("""
                🌲 **Random Forest Benefits:**
                - Handles non-linear relationships
                - Robust to outliers
                - No feature scaling needed
                - Built-in feature importance
                - Less prone to overfitting
                """)
    
    # Model status
    if st.session_state.predictor.model is not None:
        st.sidebar.success("✅ Model Ready")
        st.sidebar.info(f"Training Data Points: {len(st.session_state.training_data)}")
        
        # Show current model type
        model_type = type(st.session_state.predictor.model).__name__
        st.sidebar.info(f"🤖 Current Model: {model_type}")
    else:
        st.sidebar.warning("❌ Model Not Trained")
        st.sidebar.info("Click 'Train/Retrain Model' to start")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Data Analysis", "📈 Model Performance", "💾 Data Management"])
    
    with tab1:
        st.header("Make Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            conductivity = st.number_input(
                "Conductivity (µS/cm)", 
                min_value=1000.0, 
                max_value=10000.0, 
                value=4500.0,
                step=50.0,
                help="Typical range: 4000-5500 µS/cm"
            )
            
            ph = st.number_input(
                "pH", 
                min_value=6.0, 
                max_value=10.0, 
                value=8.0,
                step=0.1,
                help="Typical range: 7.5-8.5"
            )
        
        with col2:
            if st.button("🔮 Predict Hardness", type="primary"):
                if st.session_state.predictor.model is not None:
                    prediction = st.session_state.predictor.predict(conductivity, ph)
                    
                    st.success(f"**Predicted Total Hardness: {prediction:.0f} ppm**")
                    
                    # Store prediction for potential feedback
                    st.session_state.predictions_made.append({
                        'timestamp': datetime.now(),
                        'conductivity': conductivity,
                        'ph': ph,
                        'predicted_hardness': prediction
                    })
                    
                    # Confidence indicator (simplified)
                    if 4200 <= conductivity <= 5000 and 7.5 <= ph <= 8.5:
                        st.info("🎯 High confidence - values within training range")
                    else:
                        st.warning("⚠️ Lower confidence - extrapolating beyond training data")
                else:
                    st.error("Please train the model first!")
        
        # Feedback section
        if st.session_state.predictions_made:
            st.subheader("📝 Provide Feedback")
            st.write("Help improve the model by providing actual measurements:")
            
            with st.expander("Add Actual Measurement"):
                actual_hardness = st.number_input(
                    "Actual Total Hardness (ppm)", 
                    min_value=500.0, 
                    max_value=3000.0, 
                    value=1400.0
                )
                
                if st.button("✅ Add to Training Data"):
                    if st.session_state.predictions_made:
                        last_prediction = st.session_state.predictions_made[-1]
                        
                        # Add to training data
                        new_row = pd.DataFrame({
                            'Date': [datetime.now().strftime('%d-%m-%y')],
                            'Conductivity': [last_prediction['conductivity']],
                            'pH': [last_prediction['ph']],
                            'Total_Hardness': [actual_hardness]
                        })
                        
                        st.session_state.training_data = pd.concat([
                            st.session_state.training_data, new_row
                        ], ignore_index=True)
                        
                        # Retrain model with new data
                        with st.spinner("Updating model with Random Forest..."):
                            train_initial_model('Random Forest')  # Force Random Forest for updates
                            save_model_and_data()
                        
                        st.success("✅ Data added and model updated!")
                        st.balloons()
    
    with tab2:
        st.header("Data Analysis")
        
        if not st.session_state.training_data.empty:
            # Data overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(st.session_state.training_data))
            
            with col2:
                avg_hardness = st.session_state.training_data['Total_Hardness'].mean()
                st.metric("Average Hardness", f"{avg_hardness:.0f} ppm")
            
            with col3:
                correlation = st.session_state.training_data[['Conductivity', 'Total_Hardness']].corr().iloc[0,1]
                st.metric("Conductivity Correlation", f"{correlation:.3f}")
            
            # Visualizations
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Hardness vs Conductivity', 'Hardness vs pH', 
                              'Data Distribution', 'Hardness Over Time')
            )
            
            # Scatter plots
            fig.add_trace(
                go.Scatter(x=st.session_state.training_data['Conductivity'], 
                          y=st.session_state.training_data['Total_Hardness'],
                          mode='markers', name='Hardness vs Conductivity'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=st.session_state.training_data['pH'], 
                          y=st.session_state.training_data['Total_Hardness'],
                          mode='markers', name='Hardness vs pH'),
                row=1, col=2
            )
            
            # Distribution
            fig.add_trace(
                go.Histogram(x=st.session_state.training_data['Total_Hardness'], 
                           name='Hardness Distribution'),
                row=2, col=1
            )
            
            # Time series
            fig.add_trace(
                go.Scatter(x=st.session_state.training_data.index, 
                          y=st.session_state.training_data['Total_Hardness'],
                          mode='lines+markers', name='Hardness Over Time'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.subheader("Training Data")
            st.dataframe(st.session_state.training_data, use_container_width=True)
    
    with tab3:
        st.header("Model Performance")
        
        if st.session_state.predictor.model is not None and len(st.session_state.training_data) > 1:
            # Make predictions on training data
            X = []
            for _, row in st.session_state.training_data.iterrows():
                features = st.session_state.predictor.create_features(row['Conductivity'], row['pH'])
                X.append(features[0])
            
            X = np.array(X)
            
            # Use appropriate input based on model type
            if hasattr(st.session_state.predictor, 'use_scaling') and st.session_state.predictor.use_scaling:
                X_input = st.session_state.predictor.scaler.transform(X)
            else:
                X_input = X
                
            y_true = st.session_state.training_data['Total_Hardness'].values
            y_pred = st.session_state.predictor.model.predict(X_input)
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Display metrics with model info
            model_type = type(st.session_state.predictor.model).__name__
            st.success(f"🤖 **Current Model: {model_type}**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R² Score", f"{r2:.3f}")
            with col2:
                st.metric("RMSE", f"{rmse:.1f} ppm")
            with col3:
                st.metric("MAE", f"{mae:.1f} ppm")
            with col4:
                st.metric("MSE", f"{mse:.1f}")
            
            # Show Random Forest specific info if applicable
            if model_type == 'RandomForestRegressor':
                st.info("🌲 **Random Forest Advantages:** Excellent for non-linear relationships, robust to outliers, provides feature importance")
                
                # Feature importance (if available)
                if hasattr(st.session_state.predictor.model, 'feature_importances_'):
                    feature_names = ['Conductivity', 'pH', 'Cond/pH Ratio', 'Log Conductivity', 'pH²', 'Cond×pH']
                    importances = st.session_state.predictor.model.feature_importances_
                    
                    fig_importance = go.Figure(data=[
                        go.Bar(x=feature_names, y=importances)
                    ])
                    fig_importance.update_layout(
                        title="🎯 Feature Importance (Random Forest)",
                        xaxis_title="Features",
                        yaxis_title="Importance",
                        height=300
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
            
            # Prediction vs Actual plot
            fig = go.Figure()
            
            # Perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], 
                y=[min_val, max_val],
                mode='lines', 
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            
            # Actual vs predicted
            fig.add_trace(go.Scatter(
                x=y_true, 
                y=y_pred,
                mode='markers',
                name='Predictions',
                text=[f"Point {i+1}" for i in range(len(y_true))],
                hovertemplate="Actual: %{x}<br>Predicted: %{y}<extra></extra>"
            ))
            
            fig.update_layout(
                title="Predicted vs Actual Hardness",
                xaxis_title="Actual Hardness (ppm)",
                yaxis_title="Predicted Hardness (ppm)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals plot
            residuals = y_true - y_pred
            
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals'
            ))
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            fig_residuals.update_layout(
                title="Residuals Plot",
                xaxis_title="Predicted Hardness (ppm)",
                yaxis_title="Residuals (ppm)",
                height=400
            )
            
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        else:
            st.info("Train the model to see performance metrics")
    
    with tab4:
        st.header("Data Management")
        
        # Upload new data
        st.subheader("📤 Upload Training Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="CSV should have columns: Conductivity, pH, Total_Hardness"
        )
        
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(new_data.head())
                
                if st.button("📥 Add to Training Data"):
                    # Validate columns
                    required_cols = ['Conductivity', 'pH', 'Total_Hardness']
                    if all(col in new_data.columns for col in required_cols):
                        st.session_state.training_data = pd.concat([
                            st.session_state.training_data, new_data
                        ], ignore_index=True)
                        st.success(f"Added {len(new_data)} records to training data")
                    else:
                        st.error(f"CSV must contain columns: {required_cols}")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        # Download current data
        st.subheader("📥 Download Data")
        if not st.session_state.training_data.empty:
            csv = st.session_state.training_data.to_csv(index=False)
            st.download_button(
                label="Download Training Data as CSV",
                data=csv,
                file_name=f"water_hardness_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Clear data
        st.subheader("🗑️ Reset")
        if st.button("🔄 Reset to Initial Data", type="secondary"):
            st.session_state.training_data = load_initial_data()
            st.success("Data reset to initial dataset")
        
        if st.button("❌ Clear All Data", type="secondary"):
            st.session_state.training_data = pd.DataFrame()
            st.session_state.predictor = WaterHardnessPredictor()
            st.warning("All data cleared")

if __name__ == "__main__":
    main()
