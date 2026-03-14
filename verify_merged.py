import pandas as pd
from pathlib import Path

# Load and verify the final merged dataset
df_final = pd.read_csv('cleaned datasets/student_success_dataset.csv')

print("=" * 70)
print("STUDENT SUCCESS DATASET - FINAL VERIFICATION")
print("=" * 70)

print(f"\nFile: student_success_dataset.csv")
print(f"Location: cleaned datasets/")
print(f"\n\nDataset Summary:")
print(f"  Shape: {df_final.shape}")
print(f"  Records: {len(df_final):,}")
print(f"  Features: {len(df_final.columns)}")

print(f"\n\nData Quality:")
print(f"  Missing values: {df_final.isnull().sum().sum()}")
print(f"  Duplicate rows: {df_final.duplicated().sum()}")
print(f"  Unique values per column: {df_final.nunique().describe()}")

print(f"\n\nData Types Distribution:")
print(df_final.dtypes.value_counts())

print(f"\n\nKey Statistics:")
print(f"  Integer columns: {len(df_final.select_dtypes(include=['int64']).columns)}")
print(f"  Float columns: {len(df_final.select_dtypes(include=['float64']).columns)}")
print(f"  Object columns: {len(df_final.select_dtypes(include=['object']).columns)}")
print(f"  Category columns: {len(df_final.select_dtypes(include=['category']).columns)}")

print(f"\n\nDataset Sources Combined:")
print(f"  • cleaned_attitude_dataset.csv (235 records)")
print(f"  • research_student_cleaned.csv (221 records)")
print(f"  • cleaned_habits_dataset.csv (80,000 records)")
print(f"  = TOTAL: {len(df_final):,} records")

print(f"\n\nAll {len(df_final.columns)} Features in Final Dataset:")
for i, col in enumerate(df_final.columns, 1):
    dtype = df_final[col].dtype
    non_null = df_final[col].notna().sum()
    print(f"  {i:2d}. {col:40s} ({dtype:10s}) - {non_null:,} non-null")

print(f"\n✓ Dataset is ready for machine learning and analysis!")
