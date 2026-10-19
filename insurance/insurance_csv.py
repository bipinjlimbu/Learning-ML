import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore") # Ignore warnings for cleaner output

df = pd.read_csv('insurance/insurance.csv') # Load the insurance dataset into a pandas DataFrame

print("DataFrame:\n")
print(df)