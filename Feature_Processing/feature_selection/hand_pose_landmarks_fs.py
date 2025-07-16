import pandas as pd
import json
import os
from dotenv import load_dotenv
load_dotenv()
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score

# Load summarized feature data
df = pd.read_parquet(os.getenv('WLASL100_HAND_POSE_LANDMARKS_SUMMARY_PATH'))

# Load metadata
with open(os.getenv("WLASL_METADATA_PATH")) as f:
    metadata = json.load(f)

glosses = metadata[:100]
video_splits = {}
for gloss in glosses:
    for instance in gloss["instances"]:
        video_splits[instance["video_id"]] = instance["split"]

df["split"] = df["video_id"].map(video_splits)

# Prepare features and labels
X = df.drop(columns=["video_id", "gloss", "split"])
y = df["gloss"]
splits = df["split"]

X_trainval = X[splits != "test"].reset_index(drop=True)
y_trainval = y[splits != "test"].reset_index(drop=True)
X_test = X[splits == "test"].reset_index(drop=True)
y_test = y[splits == "test"].reset_index(drop=True)

# Save original feature names
original_feature_names = X.columns

# Build final pipeline using best hyperparameters
pipeline = Pipeline([
    ("imputer", SimpleImputer(missing_values=-2, strategy="mean")),
    ("scaler", StandardScaler()),
    ("univariate", SelectKBest(score_func=f_classif, k=200)),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_features='sqrt',
        max_depth=None,
        random_state=42
    ))
])

# Train on full training + validation set
pipeline.fit(X_trainval, y_trainval)

# Evaluate on test set
y_pred = pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"🧪 Final Test Accuracy: {test_acc:.4f}")

# Get selected feature names and importance
selected_feature_names = pipeline.named_steps["univariate"].get_feature_names_out()
decoded_feature_names = [original_feature_names[int(name[1:])] for name in selected_feature_names]
importances = pipeline.named_steps["classifier"].feature_importances_

# Sort and print top 10 features
feature_importance_pairs = sorted(
    zip(decoded_feature_names, importances),
    key=lambda x: x[1],
    reverse=True
)

# Aggregate importances by base name
feature_importance_total = defaultdict(float)

for name, importance in feature_importance_pairs:
    base = name.rsplit("_", 1)[0]  # Remove metric (e.g., "_mean")
    base = base.rsplit("_", 1)[0]  # Remove x and y
    feature_importance_total[base] += importance

# Sort feature bases by total importance
sorted_feature_importance = sorted(feature_importance_total.items(), key=lambda x: x[1], reverse=True)

top_feature_bases = [feature for feature, _ in sorted_feature_importance]
print(len(top_feature_bases))
with open("features/hand_pose_landmarks_features.py", "w") as f:
    f.write("TOP_LANDMARKS_BASES = [\n")
    for feature in top_feature_bases:
        f.write(f'    "{feature}",\n')
    f.write("]\n")
