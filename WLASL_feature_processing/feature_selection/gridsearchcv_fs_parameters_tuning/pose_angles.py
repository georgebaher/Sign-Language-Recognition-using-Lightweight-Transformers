import pandas as pd
import json
import os
from dotenv import load_dotenv
import numpy as np
load_dotenv()
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------
# 1. Load summarized feature data
# -----------------------------------------------------------------------------
df = pd.read_parquet(os.getenv('WLASL100_POSE_ANGLES_SUMMARY_PATH'))

# -----------------------------------------------------------------------------
# 2. Load metadata to get train/val/test splits
# -----------------------------------------------------------------------------
with open(os.getenv("WLASL_METADATA_PATH")) as f:
    metadata = json.load(f)

glosses = metadata[:100]

video_splits = {}
for gloss in glosses:
    for instance in gloss["instances"]:
        video_splits[instance["video_id"]] = instance["split"]

df["split"] = df["video_id"].map(video_splits)


# -----------------------------------------------------------------------------
# 3. Prepare top_features and labels
# -----------------------------------------------------------------------------
X = df.drop(columns=["video_id", "gloss", "split"])
y = df["gloss"]
splits = df["split"]

X_trainval = X[splits != "test"].reset_index(drop=True)
y_trainval = y[splits != "test"].reset_index(drop=True)
split_trainval = splits[splits != "test"].reset_index(drop=True)

# Create PredefinedSplit: -1 = train, 0 = validation
ps = PredefinedSplit(test_fold=[0 if s == "val" else -1 for s in split_trainval])

X_test = X[splits == "test"]
y_test = y[splits == "test"]

# -----------------------------------------------------------------------------
# 4. Build Pipeline: Imputer → Scaler → Feature Selection → Classifier
# -----------------------------------------------------------------------------
pipeline = Pipeline([
    ("imputer", SimpleImputer(missing_values=-2, strategy="mean")),  # Fill -2 with column mean
    ("scaler", StandardScaler()),                                     # Standardize top_features
    ("univariate", SelectKBest(score_func=f_classif)),                # Select top k top_features
    ("classifier", RandomForestClassifier(random_state=42))           # Classifier
])

# -----------------------------------------------------------------------------
# 5. Define parameter grid for GridSearchCV
# -----------------------------------------------------------------------------
param_grid = {
    "univariate__k": [50, 100, 150, 200, 250, 300, 350, 400,
                      450, 500, 550, 600, 650, 700, 750, 800,
                      850, 900, 950, 1000, 'all'],
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None],  # , 10, 20],
    "classifier__max_features": ['sqrt', 'log2']
}

# -----------------------------------------------------------------------------
# 6. Run grid search using predefined train/val split
# -----------------------------------------------------------------------------
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    cv=ps,
    verbose=2,
    n_jobs=-1
)

grid.fit(X_trainval, y_trainval)

# -----------------------------------------------------------------------------
# 7. Show best parameters and validation performance
# -----------------------------------------------------------------------------
print("✅ Best hyperparameters:")
print(grid.best_params_)
print(f"✅ Validation accuracy: {grid.best_score_:.4f}")

# -----------------------------------------------------------------------------
# 8. Evaluate on test set
# -----------------------------------------------------------------------------
y_pred = grid.best_estimator_.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"🧪 Test accuracy: {test_acc:.4f}")
