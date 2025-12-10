import joblib
import pandas as pd
print('Testing model...')
model = joblib.load('wine_quality_model.pkl')
features = joblib.load('model_features.pkl')
print(f'Features: {len(features["features"])}')
