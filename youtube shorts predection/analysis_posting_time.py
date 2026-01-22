import pandas as pd
import matplotlib.pyplot as plt
import os

# Set working directory to the script's location to find the csv
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load Data
try:
    df = pd.read_csv('youtube_shorts_performance_dataset.csv')
except FileNotFoundError:
    print("CSV file not found.")
    exit(1)

# Feature Engineering
df['engagement_rate'] = (df['likes'] + df['comments'] + df['shares']) / df['views']

# Group by upload_hour
hourly_stats = df.groupby('upload_hour')['engagement_rate'].mean()

# Plot
plt.figure(figsize=(12, 6))
hourly_stats.plot(kind='bar', color='skyblue')
plt.title('Average Engagement Rate by Upload Hour')
plt.xlabel('Upload Hour')
plt.ylabel('Average Engagement Rate')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('upload_hour_engagement.png')
print("Plot saved to upload_hour_engagement.png")

# Identify Peaks
peaks = hourly_stats.sort_values(ascending=False).head(3)
print("\nTop 3 Optimal Posting Times:")
for hour, rate in peaks.items():
    print(f"Hour {hour}: Engagement Rate {rate:.4f}")
