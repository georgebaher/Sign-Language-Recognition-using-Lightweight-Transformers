import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
parquet_path = os.getenv('WLASL100_HAND_POSE_LANDMARKS_PATH')
df = pd.read_parquet(parquet_path)
print(df.head())
