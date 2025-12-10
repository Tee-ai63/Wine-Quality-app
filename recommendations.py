# recommendations.py
import pandas as pd

# Load predictions
predictions = pd.read_csv('predictions.csv')

# Sort by quality
sorted_wines = predictions.sort_values('predicted_quality', ascending=False)

print("="*60)
print("WINE RECOMMENDATIONS")
print("="*60)

print(f"\n TOP RECOMMENDATION:")
best_wine = sorted_wines.iloc[0]
print(f"  Wine #{sorted_wines.index[0] + 1}")
print(f"  Quality Score: {best_wine['predicted_quality']:.2f}")
print(f"  Confidence: {best_wine['prediction_confidence']:.2f}")
print(f"  Key Characteristics:")
print(f"    - Alcohol: {best_wine['alcohol']}%")
print(f"    - Acidity: {best_wine['fixed acidity']}")
print(f"    - pH: {best_wine['pH']}")

print(f"\n ALL WINES RANKED:")
for idx, (_, wine) in enumerate(sorted_wines.iterrows(), 1):
    rating = "" * int(wine['predicted_quality'] - 3)  # Simple star rating
    print(f"\n#{idx}: {rating}")
    print(f"   Quality: {wine['predicted_quality']:.2f}")
    print(f"   Alcohol: {wine['alcohol']}% | Acidity: {wine['fixed acidity']}")

print(f"\n INSIGHTS:")
print(f"- Higher alcohol content tends to improve quality")
print(f"- Target alcohol: >10% for better quality wines")
print(f"- Lower volatile acidity (<0.6) improves quality")

print("="*60)

# Save recommendations
sorted_wines.to_csv('wine_recommendations.csv', index=False)
print(f" Recommendations saved to 'wine_recommendations.csv'")