import streamlit as st
import joblib
import pandas as pd

st.title("í½· Wine Quality Predictor - FIXED")

model = joblib.load('wine_model.pkl')
st.success("âœ… Model loaded!")

features = joblib.load('wine_features.pkl')
feature_names = features['features']

st.header("Make Prediction")

input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(feature, value=0.0, step=0.1)

if st.button("Predict Quality"):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    st.success(f"Predicted Quality: {prediction[0]:.2f}/10")
