import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
landmarks_df = pd.read_parquet(os.getenv("WLASL100_HAND_ANGLES_PATH"))
print(landmarks_df)