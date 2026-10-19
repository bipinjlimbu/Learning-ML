from insurance_csv import df
import pandas as pd

df_cleaned = df.copy()  # Create a copy of the original DataFrame for cleaning and preprocessing

print("\nDataFrame Head:\n")
print(df_cleaned.head())

print("\nDataFrame Shape:\n")
print(df_cleaned.shape)

print("\nDataFrame Drop Duplicates:\n")
df_cleaned.drop_duplicates(inplace=True)  # Drop duplicate rows if any
print(df_cleaned.shape)

print("\nDataFrame Null Values:\n")
print(df_cleaned.isnull().sum())

print("\nDataFrame Data Types:\n")
print(df_cleaned.dtypes)

print("\nChanging Sex to Numeric:\n")
# Convert 'sex' column to numeric (0 for male, 1 for female)
df_cleaned['sex'] = df_cleaned['sex'].map({'male': 0, 'female': 1})
print(df_cleaned.head())

print("\nChanging Smoker to Numeric:\n")
# Convert 'smoker' column to numeric (0 for no, 1 for yes)
df_cleaned['smoker'] = df_cleaned['smoker'].map({'no': 0, 'yes': 1})
print(df_cleaned.head())

print("\nRenaming Sex and Smoker Columns:\n")
df_cleaned.rename(columns={'sex': 'is_female', 'smoker': 'is_smoker'}, inplace=True)
print(df_cleaned.head())

print("\nDividing Region into Dummy Variables:\n")
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)
print(df_cleaned.head())

print("\nDataFrame Cleaned:\n")
df_cleaned = df_cleaned.astype(int)  # Convert all columns to integer type for consistency
print(df_cleaned.head())