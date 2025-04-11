import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('D:/agri_intelligence/data/Crop_recommendation.csv')

# Basic info
print("🔹 First 5 rows of the dataset:")
print(df.head())

print("\n🔹 Statistical Summary:")
print(df.describe())

print("\n🔹 Missing Values:")
print(df.isnull().sum())

# Drop the non-numeric 'label' column before calculating correlation
correlation_matrix = df.drop('label', axis=1).corr(method='pearson')

# Heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='YlGnBu',
    fmt='.2f',
    annot_kws={"size": 10},
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": .75}
)
plt.title("🌾 Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()

# Save the figure
plt.savefig("correlation_heatmap.png", dpi=300)
print("\n✅ Heatmap saved successfully as 'correlation_heatmap.png'.")

# Show plot
plt.show()
