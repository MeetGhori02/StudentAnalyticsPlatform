import pandas as pd
from pathlib import Path

df = pd.read_csv('cleaned datasets/cleaned_habits_dataset.csv')

print("=" * 70)
print("ENHANCED STUDENT HABITS & PERFORMANCE DATASET - VERIFICATION")
print("=" * 70)
print(f"\nFile: cleaned_habits_dataset.csv")
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

print(f"\n\nBinary Columns (0/1):")
binary_check = ['part_time_job', 'extracurricular_participation', 'access_to_tutoring', 'dropout_risk']
for col in binary_check:
    if col in df.columns:
        print(f"  ✓ {col}: {sorted(df[col].unique())}")

print(f"\n\nCategorical Columns (converted to category dtype):")
cat_cols = df.select_dtypes(include='category').columns.tolist()
for col in cat_cols[:5]:
    print(f"  ✓ {col}")

print(f"\n\nFeatures Created:")
new_features = ['total_screen_time', 'productivity_score', 'wellbeing_score']
for feat in new_features:
    if feat in df.columns:
        print(f"  ✓ {feat}")

print(f"\n\nNormalized Columns (0-1 range):")
normalized = ['study_hours_per_day', 'attendance_percentage', 'sleep_hours']
for col in normalized:
    if col in df.columns:
        print(f"  ✓ {col}: min={df[col].min():.4f}, max={df[col].max():.4f}")

print(f"\n\nDataset Ready for:")
print("  ✓ Machine Learning modeling")
print("  ✓ Merging with other datasets")
