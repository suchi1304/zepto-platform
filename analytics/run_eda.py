import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs("analytics", exist_ok=True)
os.makedirs("analytics/charts", exist_ok=True)

# 1. Load Titanic Dataset exactly once & Save offline fallback
print("=== TASK 1: Loading Dataset & Offline Fallback ===")
try:
    df = sns.load_dataset('titanic')
except Exception as e:
    print(f"Network load failed, checking local: {e}")
    df = pd.read_csv("analytics/titanic.csv")

# Save committed offline fallback
csv_path = "analytics/titanic.csv"
df.to_csv(csv_path, index=False)
print(f"Dataset cached and saved to: {csv_path}")
print(f"Dataset Shape: {df.shape}")
print("\nDataset Info:")
df.info()

# 2. Missing-Value Percentages & Strategy
print("\n=== TASK 2: Missing-Value Percentages & Threshold Strategy ===")
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_cols = missing_pct[missing_pct > 0]
for col, pct in missing_cols.items():
    print(f"Column '{col}': {pct:.2f}% missing")
    if pct < 5.0:
        print(f"  -> Rule: <5% missing. Strategy: Drop affected rows.")
    elif 5.0 <= pct <= 30.0:
        print(f"  -> Rule: 5%-30% missing. Strategy: Impute with median/mode.")
    else:
        print(f"  -> Rule: >30% missing. Strategy: Drop column (or encode missing category) due to high sparsity.")

# Apply cleaning per rule
df_clean = df.copy()
# Embarked (<5% missing) -> drop rows
df_clean = df_clean.dropna(subset=['embarked', 'embark_town'])
# Age (~19.8% missing) -> median impute
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
# Deck (>70% missing) -> drop column due to extreme missing rate
df_clean = df_clean.drop(columns=['deck'])

print(f"\nShape after threshold cleaning: {df_clean.shape}")

# 3. Univariate Analysis: IQR Outliers & Skewness
print("\n=== TASK 3: IQR Outliers & Skewness ===")
for col in ['age', 'fare']:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
    print(f"Column '{col}': IQR={iqr:.2f}, Lower={lower_bound:.2f}, Upper={upper_bound:.2f}, Outlier count={len(outliers)}")

fare_mean = df_clean['fare'].mean()
fare_median = df_clean['fare'].median()
fare_mode = df_clean['fare'].mode()[0]
print(f"\nFare Metrics: Mean={fare_mean:.2f}, Median={fare_median:.2f}, Mode={fare_mode:.2f}")
if fare_mean > fare_median > fare_mode:
    print("Conclusion: Fare is RIGHT-SKEWED because Mean > Median > Mode.")

# 4. Bivariate Analysis: Survival Rates & 6-Column Correlation Heatmap
print("\n=== TASK 4: Bivariate Survival Rates & 6-Column Correlation ===")
print("Survival rate by sex:")
print(df_clean.groupby('sex')['survived'].mean())

print("\nSurvival rate by pclass:")
print(df_clean.groupby('pclass')['survived'].mean())

print("\nSurvival rate by sex + pclass:")
print(df_clean.groupby(['sex', 'pclass'])['survived'].mean())

# Restricted 6 columns: survived, pclass, age, sibsp, parch, fare (adult_male, alone excluded)
corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr_matrix = df_clean[corr_cols].corr()
print("\n6x6 Correlation Matrix:")
print(corr_matrix)

# Plot and save heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (6 Target Numeric Columns)")
plt.tight_layout()
plt.savefig("analytics/charts/01_correlation_heatmap.png")
plt.close()

# Find top 2 off-diagonal pairs
corr_pairs = corr_matrix.abs().unstack()
corr_pairs = corr_pairs[corr_pairs < 1.0].sort_values(ascending=False)
top_pairs = corr_pairs.drop_duplicates()[:2]
print("\nTwo Strongest Off-Diagonal Correlations:")
for (feat1, feat2), val in top_pairs.items():
    actual_val = corr_matrix.loc[feat1, feat2]
    print(f"  - {feat1} & {feat2}: Correlation = {actual_val:.3f}")

# 5. Multivariate Charts (4 distinct charts saved)
print("\n=== TASK 5: Generating 4 Multivariate Charts ===")
# Chart 1: Survival by Sex and Pclass (Catplot)
sns.catplot(data=df_clean, x='pclass', y='survived', hue='sex', kind='bar')
plt.title("Chart 1: Survival Probability by Class and Sex")
plt.savefig("analytics/charts/chart1_sex_pclass_survival.png")
plt.close()

# Chart 2: Age vs Fare with Survival
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='age', y='fare', hue='survived', alpha=0.7)
plt.title("Chart 2: Age vs Fare Colored by Survival")
plt.savefig("analytics/charts/chart2_age_fare_survival.png")
plt.close()

# Chart 3: Fare Distribution by Pclass and Survived (Boxplot)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x='pclass', y='fare', hue='survived')
plt.title("Chart 3: Fare Spread across Pclass and Survival")
plt.savefig("analytics/charts/chart3_pclass_fare_survival.png")
plt.close()

# Chart 4: Family Size vs Survival by Class
df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch']
plt.figure(figsize=(8, 5))
sns.pointplot(data=df_clean, x='family_size', y='survived', hue='pclass')
plt.title("Chart 4: Survival by Family Size across Pclasses")
plt.savefig("analytics/charts/chart4_family_pclass_survival.png")
plt.close()
print("Saved 4 multivariate charts in analytics/charts/")

# 6. Standardization Sanity Check (Age & Fare)
print("\n=== TASK 6: Standardization Sanity Check ===")
for col in ['age', 'fare']:
    mean_val = df_clean[col].mean()
    std_val = df_clean[col].std()
    z_col = (df_clean[col] - mean_val) / std_val
    print(f"{col}: Raw Mean={mean_val:.2f}, Std={std_val:.2f} --> Standardized Mean={z_col.mean():.4f}, Std={z_col.std():.4f}")

print("\n--- Part A Complete! ---")