import pandas as pd

# Load the CSV
csv_path = "wlasl100_landmarks.csv"
parquet_path = "wlasl100_landmarks.parquet"

df = pd.read_csv(csv_path)

# Save as Parquet
df.to_parquet(parquet_path, index=False)
print(f"✅ Converted and saved as Parquet: {parquet_path}")

df2= pd.read_parquet(parquet_path)
print(df2.head())