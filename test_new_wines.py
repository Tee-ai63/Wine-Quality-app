# test_new_wines.py
import pandas as pd
import subprocess

# Create new test wines
new_wines = pd.DataFrame([
    # Try higher alcohol
    {'fixed acidity': 7.0, 'volatile acidity': 0.4, 'citric acid': 0.3, 'residual sugar': 2.0,
     'chlorides': 0.05, 'free sulfur dioxide': 20.0, 'total sulfur dioxide': 50.0,
     'density': 0.995, 'pH': 3.4, 'sulphates': 0.6, 'alcohol': 12.0},
    
    # Try lower acidity
    {'fixed acidity': 6.5, 'volatile acidity': 0.3, 'citric acid': 0.4, 'residual sugar': 1.5,
     'chlorides': 0.04, 'free sulfur dioxide': 15.0, 'total sulfur dioxide': 40.0,
     'density': 0.994, 'pH': 3.3, 'sulphates': 0.7, 'alcohol': 11.5},
    
    # Try balanced profile
    {'fixed acidity': 7.2, 'volatile acidity': 0.5, 'citric acid': 0.2, 'residual sugar': 2.5,
     'chlorides': 0.06, 'free sulfur dioxide': 25.0, 'total sulfur dioxide': 60.0,
     'density': 0.996, 'pH': 3.5, 'sulphates': 0.65, 'alcohol': 11.0}
])

# Save to CSV
new_wines.to_csv('new_wines_test.csv', index=False)

# Make predictions
print("Testing new wine combinations...")
subprocess.run([
    "python", "wine_quality_app.py", "predict",
    "--model", "wine_model.pkl",
    "--input", "new_wines_test.csv",
    "--output", "new_wines_predictions.csv"
])

# Show results
results = pd.read_csv('new_wines_predictions.csv')
print(f"\nResults for new wine combinations:")
for idx, row in results.iterrows():
    print(f"\nWine #{idx+1}:")
    print(f"  Alcohol: {row['alcohol']}%")
    print(f"  Predicted Quality: {row['predicted_quality']:.2f}")
    print(f"  Improvement tips:")
    if row['predicted_quality'] < 6:
        print(f"    - Increase alcohol to >11%")
        print(f"    - Reduce volatile acidity to <0.5")