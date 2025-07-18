# A fully generalizable script to find optimal hyperparameters and perform feature
# selection for different feature types (pose, face, hand) and input types (landmarks, angles).

import pandas as pd
import json
import os
import time
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser

def convert_numpy_types(obj):
    """
    Recursively iterates through a dictionary and converts any NumPy numeric types
    to their native Python counterparts, which are JSON serializable.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(element) for element in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def main():
    """Main function to run the hyperparameter tuning and feature selection pipeline."""
    start_time = time.time()

    parser = ArgumentParser(description='Tune hyperparameters and select features for a given summary file.')
    parser.add_argument('--feature_type', type=str, required=True, choices=['pose', 'face', 'hand'],
                        help="The type of features being processed (e.g., 'pose', 'face', 'hand').")
    parser.add_argument('--input_type', type=str, required=True, choices=['landmarks', 'angles', 'blendshapes'],
                        help="The nature of the input features ('landmarks' or 'angles').")
    parser.add_argument('--summary_path', type=str, required=True,
                        help="Path to the input Parquet file containing summarized features.")
    parser.add_argument('--metadata_path', type=str, required=True,
                        help="Path to the metadata JSON file with train/val/test splits.")
    parser.add_argument('--output_dir', type=str, default="top_features",
                        help="Directory where the output files will be saved.")

    args = parser.parse_args()

    feature_type = args.feature_type
    input_type = args.input_type
    print(f"===== Running Analysis for: {feature_type.upper()}_{input_type.upper()} =====")

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data and Metadata
    print("\n--- Step 1: Loading Data and Metadata ---")
    if not os.path.exists(args.summary_path):
        raise FileNotFoundError(f"The specified summary file was not found: {args.summary_path}")
    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"The specified metadata file was not found: {args.metadata_path}")

    df = pd.read_parquet(args.summary_path)
    with open(args.metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded {len(df)} samples from {os.path.basename(args.summary_path)}")

    # 2. Prepare Features, Labels, and Splits
    print("--- Step 2: Preparing Features and Splits ---")
    glosses = metadata[:100]
    video_splits = {inst["video_id"]: inst["split"] for gloss in glosses for inst in gloss["instances"]}
    df["split"] = df["video_id"].map(video_splits)
    df.dropna(subset=['split'], inplace=True)

    X = df.drop(columns=["video_id", "gloss", "split", "person_id", "frame"], errors='ignore')
    y = df["gloss"]
    splits = df["split"]
    original_feature_names = X.columns
    total_features = len(original_feature_names)
    print(f"Total number of features: {total_features}")

    X_trainval = X[splits != "test"].reset_index(drop=True)
    y_trainval = y[splits != "test"].reset_index(drop=True)
    split_trainval = splits[splits != "test"].reset_index(drop=True)
    X_test = X[splits == "test"].reset_index(drop=True)
    y_test = y[splits == "test"].reset_index(drop=True)

    ps = PredefinedSplit(test_fold=[0 if s == "val" else -1 for s in split_trainval])
    print("Data preparation complete.")

    # 3. Build Pipeline and Define Hyperparameter Grid
    pipeline = Pipeline([
        ("imputer", SimpleImputer(missing_values=-2, strategy="mean")),
        ("scaler", StandardScaler()),
        ("univariate", SelectKBest(score_func=f_classif)),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    k_values = list(np.arange(50, total_features, 50))
    if total_features not in k_values:
        k_values.append(total_features if total_features > 0 else 50)  # Use 'all' equivalent

    param_grid = {
        "univariate__k": k_values,
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__max_features": ['sqrt', 'log2']
    }
    print(f"\nUsing k values for feature selection: {k_values}")

    # 4. Run Grid Search
    print("--- Step 3: Running GridSearchCV ---")
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=ps, verbose=2, n_jobs=-1)
    grid.fit(X_trainval, y_trainval)

    # 5. Display and Save Best Hyperparameters
    print("\n--- Step 4: Displaying and Saving Results ---")
    best_params_serializable = convert_numpy_types(grid.best_params_)
    print("Best hyperparameters found:")
    print(best_params_serializable)  # Print the cleaned version
    print(f"Best validation accuracy: {grid.best_score_:.4f}")
    best_params_serializable["best_val_accuracy"] = round(grid.best_score_, 4)

    # 6. Evaluate and Perform Feature Selection
    print("\n--- Step 5: Evaluating and Selecting Features ---")
    final_model = grid.best_estimator_
    y_pred = final_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    best_params_serializable["test_accuracy"] = round(test_acc, 4)

    # save the best hyperparameters and accuracies
    best_params_path = os.path.join(args.output_dir, f"best_{args.feature_type}_{args.input_type}_hyperparams.json")
    with open(best_params_path, "w") as f:
        # Save the serializable dictionary
        json.dump(best_params_serializable, f, indent=4)
    print(f"Best hyperparameters saved to: {best_params_path}")

    selected_indices = final_model.named_steps["univariate"].get_support(indices=True)
    selected_feature_names = original_feature_names[selected_indices]
    importances = final_model.named_steps["classifier"].feature_importances_

    feature_importance_total = defaultdict(float)
    for name, importance in zip(selected_feature_names, importances):
        # --- Conditional logic for parsing feature names ---
        if input_type == 'landmarks':
            # For landmarks like 'LH#0_x_mean', get 'LH#0'
            base_name = name.split('_')[0]
        elif input_type == 'angles':
            # For angles like 'P_Angle{...}_mean', get 'P_Angle{...}'
            base_name = name.rsplit('_', 1)[0]
        elif input_type == 'blendshapes':
            base_name = name.split('_')[0]
        else:
            base_name = name  # Fallback
        feature_importance_total[base_name] += importance

    sorted_feature_importance = sorted(feature_importance_total.items(), key=lambda x: x[1], reverse=True)
    top_feature_bases = [feature for feature, _ in sorted_feature_importance]
    print(f"Found and ranked {len(top_feature_bases)} unique feature bases.")

    # Dynamic output filename
    feature_file_path = os.path.join(args.output_dir, f"{feature_type}_{input_type}_features.py")
    with open(feature_file_path, "w") as f:
        f.write(f"# This file was automatically generated for '{feature_type} {input_type}'\n")
        f.write("TOP_FEATURES = [\n")  # Using a more generic name
        for feature in top_feature_bases:
            f.write(f'    "{feature}",\n')
        f.write("]\n")
    print(f"Top feature bases saved to: {feature_file_path}")

    end_time = time.time()
    print(f"\n--- Script Finished in {end_time - start_time:.2f} seconds ---")


if __name__ == '__main__':
    main()