import pandas as pd
import numpy as np
import time
import os

# Generate a large dummy DataFrame
rows = 100_000
cols = 50
df = pd.DataFrame(np.random.rand(rows, cols), columns=[f"col_{i}" for i in range(cols)])

csv_path = "test_data.csv"
parquet_path = "test_data.parquet"

# -----------------------
# 📤 Write time comparison
# -----------------------

start = time.time()
df.to_csv(csv_path, index=False)
csv_write_time = time.time() - start
print(f"📄 CSV write time: {csv_write_time:.2f} sec")

start = time.time()
df.to_parquet(parquet_path, index=False)
parquet_write_time = time.time() - start
print(f"📦 Parquet write time: {parquet_write_time:.2f} sec")

# -----------------------
# 📥 Read time comparison
# -----------------------

start = time.time()
df_csv = pd.read_csv(csv_path)
csv_read_time = time.time() - start
print(f"📄 CSV read time: {csv_read_time:.2f} sec")

start = time.time()
df_parquet = pd.read_parquet(parquet_path)
parquet_read_time = time.time() - start
print(f"📦 Parquet read time: {parquet_read_time:.2f} sec")

# Optional cleanup
os.remove(csv_path)
os.remove(parquet_path)
