import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Water Hardness Predictor",
    page_icon="💧",
    layout="centered"
)

# --- Model Training ---
# This section prepares the model. It's best to have it at the top.

# 1. The original dataset provided by the user.
# In a real-world app, this might be loaded from a file or database.
data = {
    'Total Hardness (ppm)': [1590, 1390, 1460, 1430, 1430, 1400, 1500, 1510, 1430, 1480, 1270, 1190, 1390, 1390],
    'Conductivity (µS/cm)': [4520, 4230, 4311, 4320, 4400, 4340, 4348, 4800, 4360, 5490, 4240, 4250, 4840, 4820],
    'pH': [7.74, 8.14, 7.95, 8.16, 8.24, 8.76, 7.79, 8.00, 8.00, 8.00, 7.68, 8.12, 8.22, 7.97]
}
df = pd.DataFrame(data)

# 2. Define features (X) and target (y).
features = ['Conductivity (µS/cm)', 'pH']
target = 'Total Hardness (ppm)'

X = df[features]
y = df[target]

# 3. Initialize and train the Random Forest model.
# We use the entire dataset for training to make the best possible
# predictor based on the limited data available.
# random_state ensures the model is consistent every time it runs.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)


# --- Streamlit User Interface ---

# Header
st.title("💧 Water Total Hardness Predictor")
st.markdown("""
This application predicts the **Total Hardness (ppm)** of a water sample based on its **Conductivity** and **pH**. 
The prediction is made using a Random Forest Regressor model.
""")

# --- User Input Section ---
st.sidebar.header("Input Water Parameters")
st.sidebar.markdown("Use the sliders to input the values from your water test.")

# Create sliders for user input in the sidebar.
conductivity_input = st.sidebar.slider(
    'Conductivity (µS/cm)',
    min_value=float(X['Conductivity (µS/cm)'].min()),
    max_value=float(X['Conductivity (µS/cm)'].max()),
    value=float(X['Conductivity (µS/cm)'].mean()),
    step=1.0
)

ph_input = st.sidebar.slider(
    'pH',
    min_value=float(X['pH'].min()),
    max_value=float(X['pH'].max()),
    value=float(X['pH'].mean()),
    step=0.01
)

# --- Prediction and Display ---

# Create a DataFrame from the user's input
user_input_df = pd.DataFrame({
    'Conductivity (µS/cm)': [conductivity_input],
    'pH': [ph_input]
})

# Make the prediction
prediction = rf_model.predict(user_input_df)

# Display the result
st.header("Prediction Result")
st.metric(
    label="Predicted Total Hardness",
    value=f"{prediction[0]:.2f} ppm"
)

# --- Important Disclaimer ---
st.warning("""
**Disclaimer:** This model is trained on a very small dataset (14 samples). 
While it may provide a reasonable estimate, its predictions should be considered a guideline and not a substitute for precise laboratory analysis. The accuracy will improve significantly with more data.
""")


# --- Expander for more info ---
with st.expander("About the Model"):
    st.markdown("""
    - **Model Used:** `Random Forest Regressor`. This is a powerful machine learning model that works by building a multitude of decision trees and averaging their outputs. This "wisdom of the crowd" approach generally leads to more accurate and stable predictions compared to a single decision tree.
    - **Training Data:** The model was trained on the 14 data points you provided.
    - **Features:** The model uses two features for its prediction: `Conductivity (µS/cm)` and `pH`.
    """)

# Display the training data for reference
st.subheader("Training Data Used")
st.dataframe(df, use_container_width=True)
