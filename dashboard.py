# dashboard.py
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
predictions = pd.read_csv('predictions.csv')

print("="*60)
print("WINE QUALITY DASHBOARD")
print("="*60)

# Summary
print(f"\n SUMMARY:")
print(f"Total wines analyzed: {len(predictions)}")
print(f"Average quality: {predictions['predicted_quality'].mean():.2f}")
print(f"Best wine quality: {predictions['predicted_quality'].max():.2f}")
print(f"Worst wine quality: {predictions['predicted_quality'].min():.2f}")

# Classify wines
print(f"\n CLASSIFICATION:")
predictions['category'] = predictions['predicted_quality'].apply(
    lambda x: 'Excellent (8-9)' if x >= 8 else
              'Very Good (7-8)' if x >= 7 else
              'Good (6-7)' if x >= 6 else
              'Average (5-6)' if x >= 5 else
              'Below Average (3-5)'
)

category_counts = predictions['category'].value_counts()
for category, count in category_counts.items():
    print(f"  {category}: {count} wine(s)")

# Show individual results
print(f"\n INDIVIDUAL RESULTS:")
for idx, row in predictions.iterrows():
    print(f"\nWine #{idx+1}:")
    print(f"  Quality: {row['predicted_quality']:.2f} - {row['category'].split(' ')[0]}")
    print(f"  Confidence: {row['prediction_confidence']:.2f}")
    print(f"  Key features: Alcohol={row['alcohol']}%, Acidity={row['fixed acidity']}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart of qualities
colors = ['red' if q < 5 else 'orange' if q < 6 else 'yellow' if q < 7 else 'green' for q in predictions['predicted_quality']]
axes[0].bar(range(len(predictions)), predictions['predicted_quality'], color=colors, edgecolor='black')
axes[0].set_xlabel('Wine Number')
axes[0].set_ylabel('Predicted Quality')
axes[0].set_title('Wine Quality Scores')
axes[0].set_xticks(range(len(predictions)))
axes[0].set_xticklabels([f'Wine {i+1}' for i in range(len(predictions))])
axes[0].axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Average threshold')

# Scatter plot: Alcohol vs Quality
scatter = axes[1].scatter(predictions['alcohol'], predictions['predicted_quality'], 
                         c=predictions['predicted_quality'], cmap='RdYlGn', s=100, edgecolor='black')
axes[1].set_xlabel('Alcohol %')
axes[1].set_ylabel('Quality Score')
axes[1].set_title('Alcohol vs Quality')
axes[1].grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes[1], label='Quality Score')
plt.tight_layout()
plt.savefig('wine_quality_dashboard.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"\n Dashboard saved as 'wine_quality_dashboard.png'")
print("="*60)