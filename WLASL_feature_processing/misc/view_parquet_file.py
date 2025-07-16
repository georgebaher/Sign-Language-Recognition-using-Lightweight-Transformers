import pandas as pd
import os
# from dotenv import load_dotenv
# load_dotenv()
# parquet_path = os.getenv('WLASL100_FACIAL_BLENDSHAPES_PATH')
df = pd.read_parquet(r"C:\Users\boulosge\Desktop\Acht\ViTPose\ViTPose\test_videos_folder_annotated\POSE_LANDMARKS.parquet")
print(df.head())
