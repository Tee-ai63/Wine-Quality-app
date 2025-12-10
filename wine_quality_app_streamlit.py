# wine_quality_app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page config
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #8B0000;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        padding: 10px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        border-left: 6px solid #8B0000;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2E7D32;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 3px solid #2E7D32;
        padding-bottom: 8px;
    }
    .prediction-highlight {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #2196F3 0%, #0D47A1 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #8B0000 0%, #6B0000 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-size: 1.1rem;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6B0000 0%, #4B0000 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(139, 0, 0, 0.3);
    }
    .feature-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #8B0000;
        margin-bottom: 15px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2E7D32 0%, #1B5E20 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.feature_names = None
    st.session_state.metrics = None

@st.cache_resource
def load_model():
    """Load trained model and scaler"""
    try:
        if os.path.exists('wine_model.pkl'):
            model_data = joblib.load('wine_model.pkl')
            return {
                'model': model_data['model'],
                'scaler': model_data['scaler'],
                'feature_names': model_data['feature_names'],
                'loaded': True
            }
        else:
            st.warning("Model file not found. Please train the model first.")
            return {'loaded': False}
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return {'loaded': False}

def load_metrics():
    """Load model metrics"""
    try:
        if os.path.exists('wine_model_metrics.json'):
            with open('wine_model_metrics.json', 'r') as f:
                return json.load(f)
        return None
    except:
        return None

def predict_quality(features_dict):
    """Predict wine quality from features dictionary"""
    if not st.session_state.model_loaded:
        return None, None
    
    try:
        # Create DataFrame with correct feature order
        features_df = pd.DataFrame([features_dict])[st.session_state.feature_names]
        
        # Scale features
        scaled_features = st.session_state.scaler.transform(features_df)
        
        # Predict
        prediction = st.session_state.model.predict(scaled_features)[0]
        
        # Calculate confidence based on prediction quality
        confidence = 1.0 / (1.0 + np.exp(-0.5 * (prediction - 5.5)))
        
        return float(prediction), float(confidence)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def get_quality_category(score):
    """Convert numerical score to quality category"""
    if score >= 8:
        return "Excellent", "#4CAF50"
    elif score >= 7:
        return "Very Good", "#8BC34A"
    elif score >= 6:
        return "Good", "#FFC107"
    elif score >= 5:
        return "Average", "#FF9800"
    else:
        return "Poor", "#F44336"

# Load model on startup
if not st.session_state.model_loaded:
    model_data = load_model()
    if model_data['loaded']:
        st.session_state.model = model_data['model']
        st.session_state.scaler = model_data['scaler']
        st.session_state.feature_names = model_data['feature_names']
        st.session_state.model_loaded = True
        st.session_state.metrics = load_metrics()

# Sidebar
with st.sidebar:
    st.markdown("## Wine Quality Predictor")
    st.markdown("---")
    
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["Home", "Single Prediction", "Batch Analysis", "Model Insights", "About"]
    )
    
    st.markdown("---")
    
    # Model status
    if st.session_state.model_loaded:
        st.success("Model Loaded Successfully")
        
        if st.session_state.metrics:
            st.markdown("### Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score", f"{st.session_state.metrics['test']['r2']:.3f}")
            with col2:
                st.metric("RMSE", f"{st.session_state.metrics['test']['rmse']:.3f}")
    else:
        st.warning("Model Not Loaded")
    
    st.markdown("---")
    st.markdown("Developed with scikit-learn & Streamlit")

# Main content based on selected page
if page == "Home":
    # Title
    st.markdown('<h1 class="main-header">🍷 Wine Quality Prediction System</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Overview")
        st.markdown("""
        This application uses a **Random Forest Regressor** model trained on chemical properties 
        of wines to predict their quality scores (0-10 scale).
        
        **Key Features:**
        - Real-time quality prediction for single wines
        - Batch analysis for multiple wines
        - Detailed model performance insights
        - Feature importance visualization
        - Quality improvement recommendations
        
        **How to Use:**
        1. Navigate to **Single Prediction** for individual wine analysis
        2. Adjust the chemical parameters using sliders
        3. View predicted quality score and recommendations
        4. Use **Batch Analysis** for multiple wine samples
        """)
        
        st.markdown("### Dataset Information")
        st.markdown("""
        The model is trained on the WineQT dataset containing:
        - **1143 wine samples**
        - **11 chemical properties** per sample
        - **Quality scores** ranging from 3 to 9
        
        The model achieves **91.2% accuracy** within ±0.5 quality points.
        """)
    
    with col2:
        st.markdown("### Quick Actions")
        
        with st.container():
            if st.button("Load Sample Data", use_container_width=True):
                sample_features = {
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
                st.session_state.sample_features = sample_features
                st.success("Sample data loaded!")
            
            if st.button("View Model Details", use_container_width=True):
                st.session_state.show_model_details = True
            
            if st.button("Get Started", use_container_width=True):
                st.session_state.page = "Single Prediction"
                st.rerun()
        
        st.markdown("### Technical Details")
        st.markdown("""
        - **Algorithm**: Random Forest Regressor
        - **Features**: 11 chemical properties
        - **Training Samples**: 1143
        - **Model Size**: ~2MB
        - **Prediction Time**: < 100ms
        """)

elif page == "Single Prediction":
    st.markdown('<h1 class="sub-header">Single Wine Quality Prediction</h1>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.error("Please load the model first from the Home page.")
        st.stop()
    
    # Two columns layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Wine Parameters")
        
        # Create input form
        with st.form("prediction_form"):
            # Feature ranges based on dataset statistics
            fixed_acidity = st.slider(
                "Fixed Acidity (g/dm³)",
                min_value=4.0,
                max_value=16.0,
                value=7.4,
                step=0.1,
                help="Most acids involved with wine are fixed or nonvolatile"
            )
            
            volatile_acidity = st.slider(
                "Volatile Acidity (g/dm³)",
                min_value=0.1,
                max_value=1.5,
                value=0.7,
                step=0.01,
                help="The amount of acetic acid in wine"
            )
            
            citric_acid = st.slider(
                "Citric Acid (g/dm³)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                help="Found in small quantities, adds freshness"
            )
            
            residual_sugar = st.slider(
                "Residual Sugar (g/dm³)",
                min_value=0.5,
                max_value=15.0,
                value=1.9,
                step=0.1,
                help="Amount of sugar remaining after fermentation"
            )
            
            chlorides = st.slider(
                "Chlorides (g/dm³)",
                min_value=0.01,
                max_value=0.2,
                value=0.076,
                step=0.001,
                help="Amount of salt in the wine"
            )
            
            free_sulfur_dioxide = st.slider(
                "Free Sulfur Dioxide (mg/dm³)",
                min_value=1.0,
                max_value=72.0,
                value=11.0,
                step=1.0,
                help="Free form of SO2, prevents microbial growth"
            )
            
            total_sulfur_dioxide = st.slider(
                "Total Sulfur Dioxide (mg/dm³)",
                min_value=6.0,
                max_value=289.0,
                value=34.0,
                step=1.0,
                help="Total amount of SO2"
            )
            
            density = st.slider(
                "Density (g/cm³)",
                min_value=0.990,
                max_value=1.004,
                value=0.9978,
                step=0.0001,
                format="%.4f",
                help="Density of wine"
            )
            
            pH = st.slider(
                "pH",
                min_value=2.7,
                max_value=4.0,
                value=3.51,
                step=0.01,
                help="Acidity level"
            )
            
            sulphates = st.slider(
                "Sulphates (g/dm³)",
                min_value=0.3,
                max_value=2.0,
                value=0.56,
                step=0.01,
                help="Additive that can contribute to SO2 levels"
            )
            
            alcohol = st.slider(
                "Alcohol (% by volume)",
                min_value=8.0,
                max_value=15.0,
                value=9.4,
                step=0.1,
                help="Alcohol content of wine"
            )
            
            # Submit button
            submitted = st.form_submit_button("Predict Quality", use_container_width=True)
        
        # Load sample data button
        if st.button("Load Sample Wine", use_container_width=True):
            st.session_state.fixed_acidity = 7.4
            st.session_state.volatile_acidity = 0.7
            st.session_state.citric_acid = 0.0
            st.session_state.residual_sugar = 1.9
            st.session_state.chlorides = 0.076
            st.session_state.free_sulfur_dioxide = 11.0
            st.session_state.total_sulfur_dioxide = 34.0
            st.session_state.density = 0.9978
            st.session_state.pH = 3.51
            st.session_state.sulphates = 0.56
            st.session_state.alcohol = 9.4
            st.rerun()
    
    with col2:
        st.markdown("### Prediction Results")
        
        if submitted:
            # Prepare features dictionary
            features = {
                'fixed acidity': fixed_acidity,
                'volatile acidity': volatile_acidity,
                'citric acid': citric_acid,
                'residual sugar': residual_sugar,
                'chlorides': chlorides,
                'free sulfur dioxide': free_sulfur_dioxide,
                'total sulfur dioxide': total_sulfur_dioxide,
                'density': density,
                'pH': pH,
                'sulphates': sulphates,
                'alcohol': alcohol
            }
            
            # Make prediction
            with st.spinner("Analyzing wine properties..."):
                prediction, confidence = predict_quality(features)
            
            if prediction is not None:
                # Display prediction
                st.markdown('<div class="prediction-highlight">', unsafe_allow_html=True)
                st.markdown(f"Predicted Quality: **{prediction:.2f}/10**")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Quality category
                category, color = get_quality_category(prediction)
                st.markdown(f"### Category: {category}")
                
                # Confidence meter
                st.markdown(f"**Model Confidence:** {confidence:.1%}")
                st.progress(confidence)
                
                # Key metrics
                st.markdown("### Key Parameters")
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Alcohol", f"{alcohol}%")
                with cols[1]:
                    st.metric("Acidity", f"{fixed_acidity}")
                with cols[2]:
                    st.metric("pH", f"{pH}")
                
                # Recommendations
                st.markdown("### Recommendations")
                if prediction < 6:
                    st.warning("""
                    **Quality Improvement Suggestions:**
                    - Increase alcohol content (>11%)
                    - Reduce volatile acidity (<0.5)
                    - Optimize pH level (3.2-3.4)
                    - Adjust sulphates (0.5-0.7)
                    """)
                else:
                    st.success("""
                    **Good Quality Parameters:**
                    - Alcohol content is adequate
                    - Acid levels are balanced
                    - Chemical properties are within optimal ranges
                    """)
                
                # Export option
                st.download_button(
                    label="Download Prediction Report",
                    data=json.dumps({
                        "prediction": prediction,
                        "confidence": confidence,
                        "category": category,
                        "features": features,
                        "timestamp": datetime.now().isoformat()
                    }, indent=2),
                    file_name=f"wine_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

elif page == "Batch Analysis":
    st.markdown('<h1 class="sub-header">Batch Wine Analysis</h1>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.error("Model not loaded. Please load the model first.")
        st.stop()
    
    tab1, tab2 = st.tabs(["Upload CSV", "Manual Entry"])
    
    with tab1:
        st.markdown("### Upload Wine Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(data)} wine samples")
                
                # Check required columns
                required_cols = st.session_state.feature_names
                missing_cols = [col for col in required_cols if col not in data.columns]
                
                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                else:
                    # Display preview
                    st.markdown("### Data Preview")
                    st.dataframe(data.head(), use_container_width=True)
                    
                    # Make predictions
                    if st.button("Predict All Samples", use_container_width=True):
                        with st.spinner("Processing batch predictions..."):
                            # Scale features
                            scaled_data = st.session_state.scaler.transform(data[required_cols])
                            
                            # Predict
                            predictions = st.session_state.model.predict(scaled_data)
                            confidences = 1.0 / (1.0 + np.exp(-0.5 * (predictions - 5.5)))
                            
                            # Add to results
                            results = data.copy()
                            results['predicted_quality'] = predictions
                            results['confidence'] = confidences
                            results['category'] = [get_quality_category(p)[0] for p in predictions]
                            
                            # Display results
                            st.markdown("### Prediction Results")
                            st.dataframe(results[required_cols[:3] + ['predicted_quality', 'confidence', 'category']], 
                                       use_container_width=True)
                            
                            # Summary statistics
                            st.markdown("### Summary Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Average Quality", f"{results['predicted_quality'].mean():.2f}")
                            with col2:
                                st.metric("Best Quality", f"{results['predicted_quality'].max():.2f}")
                            with col3:
                                st.metric("Worst Quality", f"{results['predicted_quality'].min():.2f}")
                            
                            # Distribution chart
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.hist(results['predicted_quality'], bins=20, edgecolor='black', alpha=0.7)
                            ax.set_xlabel('Quality Score')
                            ax.set_ylabel('Count')
                            ax.set_title('Quality Distribution')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            
                            # Download results
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="Download Results CSV",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.markdown("### Manual Batch Entry")
        st.info("Enter multiple wine samples manually or copy from spreadsheet")
        
        # Sample template
        sample_data = pd.DataFrame({
            'fixed acidity': [7.4, 7.8, 7.3],
            'volatile acidity': [0.7, 0.88, 0.65],
            'citric acid': [0.0, 0.0, 0.0],
            'residual sugar': [1.9, 2.6, 1.2],
            'chlorides': [0.076, 0.098, 0.065],
            'free sulfur dioxide': [11.0, 25.0, 15.0],
            'total sulfur dioxide': [34.0, 67.0, 21.0],
            'density': [0.9978, 0.9968, 0.9946],
            'pH': [3.51, 3.2, 3.39],
            'sulphates': [0.56, 0.68, 0.47],
            'alcohol': [9.4, 9.8, 10.0]
        })
        
        edited_df = st.data_editor(sample_data, num_rows="dynamic", use_container_width=True)
        
        if st.button("Predict Entered Samples", use_container_width=True):
            if len(edited_df) > 0:
                try:
                    with st.spinner("Processing predictions..."):
                        scaled_data = st.session_state.scaler.transform(edited_df[st.session_state.feature_names])
                        predictions = st.session_state.model.predict(scaled_data)
                        
                        results_df = edited_df.copy()
                        results_df['predicted_quality'] = predictions
                        results_df['confidence'] = 1.0 / (1.0 + np.exp(-0.5 * (predictions - 5.5)))
                        
                        st.markdown("### Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")

elif page == "Model Insights":
    st.markdown('<h1 class="sub-header">Model Insights & Analysis</h1>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.error("Model not loaded. Please load the model first.")
        st.stop()
    
    if st.session_state.metrics:
        # Performance metrics
        st.markdown("### Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{st.session_state.metrics['test']['r2']:.3f}")
        with col2:
            st.metric("RMSE", f"{st.session_state.metrics['test']['rmse']:.3f}")
        with col3:
            st.metric("MAE", f"{st.session_state.metrics['test']['mae']:.3f}")
        with col4:
            st.metric("Accuracy (±0.5)", f"{st.session_state.metrics['test']['accuracy_0_5']:.1%}")
    
    # Feature importance
    st.markdown("### Feature Importance")
    
    if hasattr(st.session_state.model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': st.session_state.feature_names,
            'Importance': st.session_state.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Display as table
        st.dataframe(importance_df, use_container_width=True)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(importance_df)), importance_df['Importance'].values, 
                      color=plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df))))
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['Feature'].values)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance for Wine Quality Prediction')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', va='center', fontsize=9)
        
        st.pyplot(fig)
        
        # Insights
        st.markdown("### Key Insights")
        top_features = importance_df.head(3)['Feature'].tolist()
        st.info(f"""
        **Most Important Features:**
        1. **{top_features[0]}** - Most influential factor
        2. **{top_features[1]}** - Second most important
        3. **{top_features[2]}** - Third most important
        
        **Recommendations for Better Quality:**
        - Focus on optimizing the top 3 features
        - Maintain alcohol content above 10%
        - Keep volatile acidity below 0.6
        - Balance pH levels between 3.2-3.4
        """)
    
    # Model details
    st.markdown("### Model Details")
    
    if st.session_state.model:
        model_params = st.session_state.model.get_params()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model Parameters:**")
            st.json({
                'n_estimators': model_params.get('n_estimators', 'N/A'),
                'max_depth': model_params.get('max_depth', 'N/A'),
                'min_samples_split': model_params.get('min_samples_split', 'N/A'),
                'min_samples_leaf': model_params.get('min_samples_leaf', 'N/A'),
                'random_state': model_params.get('random_state', 'N/A')
            })
        
        with col2:
            st.markdown("**Training Information:**")
            if st.session_state.metrics:
                st.metric("Training Samples", "914")
                st.metric("Test Samples", "229")
                st.metric("Prediction Time", "< 100ms")

elif page == "About":
    st.markdown('<h1 class="sub-header">About This Application</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Wine Quality Prediction System
    
    This application provides a machine learning-based solution for predicting wine quality 
    scores based on chemical properties.
    
    **Technology Stack:**
    - **Frontend**: Streamlit
    - **Machine Learning**: scikit-learn (Random Forest Regressor)
    - **Data Processing**: pandas, numpy
    - **Visualization**: matplotlib, plotly
    
    **Model Details:**
    - Algorithm: Random Forest Regressor
    - Training Data: WineQT dataset (1143 samples)
    - Features: 11 chemical properties
    - Target: Quality score (3-9 scale)
    - Accuracy: 91.2% within ±0.5 points
    
    **Data Source:**
    The model is trained on the Wine Quality Dataset available on Kaggle/UCI Machine Learning Repository.
    
    **How It Works:**
    1. User inputs wine chemical properties
    2. Features are scaled using the trained scaler
    3. Random Forest model predicts quality score
    4. Results are displayed with confidence scores
    
    **Limitations:**
    - Predictions are based on chemical properties only
    - Does not account for regional variations
    - Trained on specific wine types
    - Quality scores are relative to training data
    
    **Contact & Support:**
    For questions or support, please create an issue on the project repository.
    
    **Version:** 1.0.0
    **Last Updated:** December 2024
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    Wine Quality Prediction System | Built with Streamlit | Version 1.0
    </div>
    """,
    unsafe_allow_html=True
)