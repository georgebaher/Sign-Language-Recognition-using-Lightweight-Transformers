import os
import pandas as pd

# Load the Excel file
df = pd.read_excel("WLASL100.xlsx", header=None)  # One-column file, no header

# Split each row by ';' and extract the 4th field (video_id)
video_ids = df[0].apply(lambda x: str(x).split(';')[3].zfill(5))  # Pad to 5 digits

# List files in raw_videos/
raw_video_files = set(os.listdir('videos'))

# Check for missing video files
missing_ids = []

for vid in video_ids:
    found = any(
        f"{vid}.{ext}" in raw_video_files
        for ext in ['mp4', 'mkv', 'webm', 'avi']
    )
    if not found:
        missing_ids.append(vid)

# Save missing video IDs
with open('missing_100.txt', 'w') as f:
    f.write('\n'.join(missing_ids))

print(f"DONE>>> found {len(missing_ids)} missing video(s) ")
