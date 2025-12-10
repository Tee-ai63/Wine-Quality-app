import joblib
import json
import os

print("Debugging model loading...")

print("\nAvailable files:")
files_to_check = ['wine_model.pkl', 'model_features.pkl', 'wine_features.pkl', 'wine_model_metrics.json']
for file in files_to_check:
    if os.path.exists(file):
        print(f"OK {file} exists")
        if file.endswith('.pkl'):
            try:
                data = joblib.load(file)
                print(f"   Type: {type(data)}")
                if isinstance(data, dict):
                    print(f"   Keys: {list(data.keys())}")
                elif hasattr(data, '__len__'):
                    print(f"   Length: {len(data)}")
            except Exception as e:
                print(f"   Error loading: {e}")
    else:
        print(f"NO {file} does not exist")

print("\nTesting model load...")
try:
    model_data = joblib.load('wine_model.pkl')
    print(f"Model loaded successfully!")
    print(f"Type of loaded data: {type(model_data)}")
    
    if isinstance(model_data, dict):
        print("It's a dictionary with keys:", list(model_data.keys()))
        for key in model_data:
            print(f"  {key}: {type(model_data[key])}")
    else:
        print("Not a dictionary, checking attributes...")
        if hasattr(model_data, 'predict'):
            print("OK Has predict() method - likely a model object")
        if hasattr(model_data, 'features'):
            print(f"Features attribute: {model_data.features}")
            
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
