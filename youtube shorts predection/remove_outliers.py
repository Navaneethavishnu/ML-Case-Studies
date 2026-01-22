import pandas as pd
import os

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load Data
input_file = 'youtube_shorts_performance_dataset.csv'
output_file = 'youtube_shorts_cleaned.csv'

try:
    df = pd.read_csv(input_file)
    original_count = len(df)
except FileNotFoundError:
    print(f"Error: {input_file} not found.")
    exit(1)

# Columns to check for outliers
cols_to_check = ['views', 'likes', 'comments', 'shares']

# Calculate IQR and filter
Q1 = df[cols_to_check].quantile(0.25)
Q3 = df[cols_to_check].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Create the mask: Keep rows where ALL values are within bounds
# Alternatively, we can remove if ANY is an outlier. Usually, we want to keep "valid" data.
# Let's use: remove row if ANY column is an outlier.
condition = ~((df[cols_to_check] < lower_bound) | (df[cols_to_check] > upper_bound)).any(axis=1)

df_clean = df[condition]
cleaned_count = len(df_clean)
removed_count = original_count - cleaned_count

# Save
df_clean.to_csv(output_file, index=False)

print(f"Original rows: {original_count}")
print(f"Cleaned rows:  {cleaned_count}")
print(f"Removed rows:  {removed_count}")
print(f"Outliers removed based on columns: {cols_to_check}")
print(f"Cleaned dataset saved to: {output_file}")
