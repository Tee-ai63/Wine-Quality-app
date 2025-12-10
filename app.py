import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    layout="wide"
)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# FIXED: Correct load_model function
@st.cache_resource
def load_model():
    """Load trained model and scaler"""
    try:
        if os.path.exists('wine_model.pkl'):
            # Load the model object directly
            model = joblib.load('wine_model.pkl')
            
            # Try to load features from separate files
            feature_names = []
            scaler = None
            
            # Try to load feature information from available files
            feature_files = ['wine_features.pkl', 'model_features.pkl']
            for file in feature_files:
                if os.path.exists(file):
                    features_data = joblib.load(file)
                    if 'features' in features_data:
                        feature_names = features_data['features']
                    break
            
            # If no features file found, use default names
            if not feature_names:
                feature_names = [
                    'fixed acidity', 'volatile acidity', 'citric acid', 
                    'residual sugar', 'chlorides', 'free sulfur dioxide',
                    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
                ]
            
            return {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'loaded': True
            }
        else:
            st.warning("Model file not found. Please train the model first.")
            return {'loaded': False}
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return {'loaded': False}

def load_metrics():
    """Load model performance metrics"""
    try:
        if os.path.exists('wine_model_metrics.json'):
            with open('wine_model_metrics.json', 'r') as f:
                return json.load(f)
        return None
    except:
        return None

def predict_quality(features_dict):
    """Predict wine quality from features dictionary"""
    model_data = load_model()
    
    if not model_data['loaded']:
        return None
    
    try:
        # Convert features dict to DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Ensure columns are in correct order
        if model_data['feature_names']:
            features_df = features_df.reindex(columns=model_data['feature_names'])
        
        # Scale features if scaler exists
        if model_data['scaler']:
            features_scaled = model_data['scaler'].transform(features_df)
            features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
        
        # Make prediction
        prediction = model_data['model'].predict(features_df)
        
        return float(prediction[0])
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Load model and metrics
model_data = load_model()
metrics = load_metrics()

# Sidebar Navigation
st.sidebar.title("Wine Quality Predictor")
st.sidebar.markdown("**Navigation**")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Single Prediction", "Batch Analysis", "Model Insights", "About"]
)

# Display model status
if model_data['loaded']:
    st.sidebar.success("Model Loaded")
    if metrics:
        st.sidebar.metric("R2 Score", f"{metrics.get('r2', 0):.3f}")
else:
    st.sidebar.error("Model Not Loaded")

# Home Page
if page == "Home":
    st.title("Wine Quality Predictor")
    
    if model_data['loaded']:
        st.success("Model is loaded and ready for predictions!")
    
    st.markdown("""
    ## The Quality Prediction System
    
    ### Overview
    This application uses a Random Forest Regressor model trained on chemical properties of wines 
    to predict their quality scores (0-10 scale).
    
    **Key Features:**
    - Real-time quality prediction for single wines
    - Batch analysis for multiple wines
    - Detailed model performance insights
    - Feature importance visualization
    """)
    
    # Quick Actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Sample Data"):
            # Create sample data
            sample_features = {}
            for feature in model_data.get('feature_names', []):
                sample_features[feature] = [np.random.uniform(0, 10)]
            
            sample_df = pd.DataFrame(sample_features)
            st.session_state.sample_data = sample_df
            st.success("Sample data loaded!")
            st.dataframe(sample_df)
    
    with col2:
        if st.button("View Model Details"):
            if model_data['loaded']:
                st.subheader("Model Information")
                st.write(f"**Model Type:** Random Forest Regressor")
                st.write(f"**Number of Trees:** {model_data['model'].n_estimators}")
                st.write(f"**Features:** {len(model_data['feature_names'])}")
                
                if metrics:
                    st.subheader("Model Performance")
                    for key, value in metrics.items():
                        st.write(f"**{key}:** {value}")
            else:
                st.warning("Model not loaded")

# Single Prediction Page
elif page == "Single Prediction":
    st.title("Single Wine Quality Prediction")
    
    if not model_data['loaded']:
        st.error("Model not loaded. Please load the model first from the Home page.")
        st.info("Go to the Home page and make sure the model loads successfully.")
        st.stop()
    
    st.success("Model is loaded and ready for predictions!")
    
    # Get feature names
    feature_names = model_data.get('feature_names', [])
    
    # Create input fields
    input_data = {}
    cols = st.columns(3)
    
    # Default values for a sample wine
    default_values = {
        'fixed acidity': 7.4,
        'volatile acidity': 0.7,
        'citric acid': 0.0,
        'residual sugar': 1.9,
        'chlorides': 0.076,
        'free sulfur dioxide': 11.0,
        'total sulfur dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4
    }
    
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            default_val = default_values.get(feature, 0.0)
            input_data[feature] = st.number_input(
                f"{feature}",
                value=float(default_val),
                step=0.1,
                key=f"input_{feature}"
            )
    
    if st.button("Predict Quality"):
        with st.spinner("Predicting..."):
            prediction = predict_quality(input_data)
            
            if prediction is not None:
                # Store prediction
                prediction_record = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'prediction': prediction
                }
                st.session_state.predictions.append(prediction_record)
                
                # Display result
                st.success(f"**Predicted Quality:** {prediction:.2f}/10")
                
                # Show input values
                st.subheader("Input Values")
                input_df = pd.DataFrame([input_data])
                st.dataframe(input_df)

# Batch Analysis Page
elif page == "Batch Analysis":
    st.title("Batch Analysis")
    st.info("Batch analysis feature coming soon...")

# Model Insights Page
elif page == "Model Insights":
    st.title("Model Insights")
    
    if model_data['loaded']:
        # Try to load feature importance
        feature_files = ['wine_features.pkl', 'model_features.pkl']
        feature_importance = None
        
        for file in feature_files:
            if os.path.exists(file):
                features_data = joblib.load(file)
                if 'feature_importances' in features_data:
                    feature_importance = features_data['feature_importances']
                    break
        
        if feature_importance and feature_names:
            # Create feature importance chart
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            st.subheader("Feature Importance")
            st.bar_chart(importance_df.set_index('Feature'))
        else:
            st.info("Feature importance data not available")
    else:
        st.warning("Model not loaded")

# About Page
elif page == "About":
    st.title("About")
    st.markdown("""
    ### Wine Quality Prediction System
    This application predicts wine quality based on chemical properties.
    
    **Model:** Random Forest Regressor
    **Features:** 11 chemical properties
    **Output:** Quality score (0-10)
    
    ---
    *Created for wine quality analysis and prediction*
    """)
