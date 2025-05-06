from collections import Counter
from .utils.extract_hand_landmarks import extract_hands_landmarks
from .models.sign_model import SignModel
import pandas as pd
from .utils.dataset import *
from .utils.dtw import dtw_distances

# Step 1: Load glosses & split
n = 5
glosses = get_first_n_glosses_wlasl(n)
reference_vids, test_vids = split_reference_test_glosses(glosses)

# Step 2: Convert references into a DataFrame of SignModels
records = []
for gloss_name, video_path in reference_vids:
    left_hand_list, right_hand_list = extract_hands_landmarks(video_path)
    sign_model = SignModel(left_hand_list, right_hand_list)
    records.append({
        "name": gloss_name,
        "sign_model": sign_model,
        "distance": 0.0
    })
ref_df = pd.DataFrame(records)

# Step 3: Evaluate on all test samples
correct = 0
total = len(test_vids)

for idx, (true_label, video_path) in enumerate(test_vids):
    left_hand_list, right_hand_list = extract_hands_landmarks(video_path)
    test_sign = SignModel(left_hand_list, right_hand_list)

    # Compute DTW distances to all references
    results_df = dtw_distances(test_sign, ref_df)

    # Get top 10% matches
    top_n = max(1, int(0.1 * len(results_df)))
    top_matches = results_df.head(top_n)

    # Predict label using majority vote
    predicted_label = Counter(top_matches["name"]).most_common(1)[0][0]

    # Track accuracy
    if predicted_label == true_label:
        correct += 1

    print(f"[{idx + 1}/{total}] True: {true_label} | Predicted: {predicted_label}")

# Final accuracy
accuracy = correct / total if total > 0 else 0
print(f"\n✅ Overall Accuracy: {accuracy:.2%}")
