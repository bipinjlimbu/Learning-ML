from insurance_csv import df
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

#EDA (Exploratory Data Analysis)

print("\nDataFrame Shape:\n")
print(df.shape)

print("\nDataFrame Head:\n")
print(df.head())

print("\nDataFrame Info:\n")
print(df.info())

print("\nDataFrame Description:\n")
print(df.describe())

print("\nDataFrame Columns:\n")
print(df.columns)

print("\nDataFrame Null Values:\n")
print(df.isnull().sum())

print("\nDataFrame Unique Values:\n")
print(df.nunique())

# Visualizing the distribution of numeric columns in the dataset using histograms and KDE plots.
numeric_columns = ['age', 'bmi', 'children','charges']
for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col],kde = True,bins = 20)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
    
# Countplot for Children column
plt.figure(figsize=(6,4))
sns.countplot(x='children', data=df)
plt.title('Count of Children')
plt.xlabel('Number of Children')
plt.ylabel('Count')
plt.show()

# Countplot for Sex column
plt.figure(figsize=(6,4))
sns.countplot(x='sex', data=df)
plt.title('Count of Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Countplot for Smoker column
plt.figure(figsize=(6,4))
sns.countplot(x='smoker', data=df)
plt.title('Count of Smoker')
plt.xlabel('Smoker')
plt.ylabel('Count')
plt.show()

# Boxplot for numeric columns to visualize the distribution and identify potential outliers.
for col in numeric_columns:
    plt.figure(figsize= (6,4))
    sns.boxplot(x = df[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()
    
# Correlation heatmap to visualize the relationships between numeric columns in the dataset.
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.title('Correlation Heatmap')
plt.show()