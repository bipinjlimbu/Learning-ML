from data_cleaning_preprocessing import df_cleaned
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, chi2_contingency
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the distribution of BMI
sns.histplot(df_cleaned['bmi'])
plt.title('Distribution of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()

print("\nDataFrame after Feature Engineering:\n")
df_cleaned['bmi_category'] = pd.cut(df_cleaned['bmi'], bins=[0, 18.5, 24.9, 29.9, float('inf')], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
print(df_cleaned)

print("\nDataFrame after One-Hot Encoding:\n")
df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)
print(df_cleaned.head())

print("\nDataFrame after Converting to Integer Type:\n")
df_cleaned = df_cleaned.astype(int)
print(df_cleaned.head())

print("\nDataFrame after Standard Scaling:\n")
scaler = StandardScaler()
cols = ['age', 'bmi', 'children']
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
print(df_cleaned.head())

# Select the features to be used in the model
selected_features = [
    'age', 'bmi', 'children', 'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

print("\nPearson Correlation Coefficients:\n")
correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}
correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Pearson Correlation'])
correlation_df.sort_values(by='Pearson Correlation', ascending=False)
print(correlation_df)

# Chi-Square Test for Categorical Features
cat_features = [
    'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

alpha = 0.05

df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)
chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Accept Null (Drop Feature)'
    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

print("\nChi-Square Test Results:\n")
chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value')
print(chi2_df)

print("\nFinal DataFrame with Selected Features:\n")
final_df = df_cleaned[['age', 'is_female', 'bmi', 'children', 'is_smoker', 'charges','region_southeast','bmi_category_Obese']]
print(final_df.head())