# setup.py
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def setup_model():
    print("🔧 Setting up Wine Quality Predictor...")
    
    # Check for existing model
    if os.path.exists('wine_quality_model.pkl'):
        print("✓ Model already exists")
        return True
    
    print("No model found. Creating one...")
    
    try:
        # Try to use real data first
        print("Looking for wine quality dataset...")
        # You can download sample data automatically:
        # URL: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
    except:
        print("Using built-in sample data...")
    
    # Create meaningful sample data (not completely random)
    np.random.seed(42)
    n_samples = 500
    
    # Realistic wine features
    data = {
        'fixed acidity': np.random.uniform(4, 15, n_samples),
        'volatile acidity': np.random.uniform(0.1, 1.5, n_samples),
        'citric acid': np.random.uniform(0, 1, n_samples),
        'residual sugar': np.random.uniform(0.5, 15, n_samples),
        'chlorides': np.random.uniform(0.01, 0.2, n_samples),
        'free sulfur dioxide': np.random.uniform(1, 60, n_samples),
        'total sulfur dioxide': np.random.uniform(10, 200, n_samples),
        'density': np.random.uniform(0.98, 1.005, n_samples),
        'pH': np.random.uniform(2.8, 4, n_samples),
        'sulphates': np.random.uniform(0.3, 2, n_samples),
        'alcohol': np.random.uniform(8, 15, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic quality scores based on features
    # Higher alcohol and lower volatile acidity = better quality
    df['quality'] = (
        3 +  # Base score
        (df['alcohol'] - 8) / 2 +  # Alcohol contributes 0-3.5 points
        (1 - df['volatile acidity']) * 2 +  # Low acidity good
        df['citric acid'] * 1.5  # More citric acid good
    ).clip(0, 10).round().astype(int)
    
    # Train model
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8)
    model.fit(X_train, y_train)
    
    # Save
    joblib.dump(model, 'wine_quality_model.pkl')
    
    # Save feature info
    feature_info = {
        'features': X.columns.tolist(),
        'feature_importances': model.feature_importances_.tolist(),
        'data_stats': df.describe().to_dict()
    }
    joblib.dump(feature_info, 'model_features.pkl')
    
    print(" Model trained and saved!")
    print(f"   - Samples: {n_samples}")
    print(f"   - Features: {len(X.columns)}")
    print(f"   - Quality range: {y.min()} to {y.max()}")
    
    return True

if __name__ == "__main__":
    import numpy as np
    setup_model()