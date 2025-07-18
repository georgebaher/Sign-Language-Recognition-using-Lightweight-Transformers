# create_summary_features.py (Version 2.0 - Assumes 'gloss' column exists)
#
# A generalizable script to create a feature summary from a time-series Parquet file.
# It calculates specified statistical metrics for each feature over all frames of a video.

import pandas as pd
import numpy as np
import os
from argparse import ArgumentParser


def main():
    """Main function to run the feature summarization pipeline."""
    parser = ArgumentParser(description='Summarize time-series data from a Parquet file.')

    # --- MODIFIED: Metadata path is now optional ---
    parser.add_argument('--input_path', type=str, required=True,
                        help="Path to the input Parquet file (e.g., POSE_LANDMARKS.parquet).")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to save the output summary Parquet file.")
    parser.add_argument('--metrics', nargs='+', default=['mean', 'std', 'min', 'max', 'var'],
                        help="List of statistical metrics to compute.")

    args = parser.parse_args()

    # 1. Load Data
    print(f"--- Step 1: Loading data from {os.path.basename(args.input_path)} ---")
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    df = pd.read_parquet(args.input_path)

    # --- THIS IS THE KEY CHECK ---
    # Ensure the required 'gloss' column is present in the input file.
    if 'gloss' not in df.columns or 'video_id' not in df.columns:
        raise ValueError("Input Parquet file must contain 'video_id' and 'gloss' columns.")

    print("Data loaded successfully.")

    # 2. Group data by video and compute summary statistics
    print(f"--- Step 2: Summarizing features with metrics: {', '.join(args.metrics)} ---")

    feature_cols = [col for col in df.columns if col not in ['video_id', 'gloss', 'frame', 'person_id']]
    df[feature_cols] = df[feature_cols].replace(-2, np.nan)

    agg_funcs = {col: args.metrics for col in feature_cols}

    summary_df = df.groupby(['video_id', 'gloss']).agg(agg_funcs).reset_index()

    summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
    summary_df.fillna(-2, inplace=True)  # Replace any remaining NaNs

    # 3. Save the final summary DataFrame
    print("--- Step 3: Saving summarized data ---")
    summary_df.to_parquet(args.output_path, index=False)
    print(f"Summarized data saved to {args.output_path}")


if __name__ == "__main__":
    main()