# A generalizable command-line utility to drop columns from a Parquet file
# that fall within a specified numerical range for a given landmark prefix.
# e.g., drop all columns for landmarks P13 through P23.

import pandas as pd
import os
import re
from argparse import ArgumentParser


def main():
    """Main function to run the column dropping process."""
    parser = ArgumentParser(
        description="A script to drop landmark columns within a given index range from a Parquet file."
    )

    # --- Command-Line Arguments ---
    parser.add_argument(
        "-i", "--input_path",
        type=str,
        required=True,
        help="Path to the input Parquet file that needs to be processed."
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        help="Path to save the new, modified Parquet file. If not provided, the input file will be overwritten."
    )
    parser.add_argument(
        "-p", "--prefix",
        type=str,
        required=True,
        help="The prefix of the landmark columns to target (e.g., 'p' for AVASAG, 'P#' for MediaPipe)."
    )
    parser.add_argument(
        "-s", "--start_index",
        type=int,
        required=True,
        help="The starting index of the landmark range to drop (inclusive)."
    )
    parser.add_argument(
        "-e", "--end_index",
        type=int,
        required=True,
        help="The ending index of the landmark range to drop (inclusive)."
    )

    args = parser.parse_args()

    # --- 1. Validate Input Path ---
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found at '{args.input_path}'")
        return

    # --- 2. Load the Parquet File ---
    print(f"Loading data from '{os.path.basename(args.input_path)}'...")
    try:
        df = pd.read_parquet(args.input_path)
    except Exception as e:
        print(f"Error: Could not read the Parquet file. {e}")
        return

    # --- 3. Identify Columns to Drop using Regular Expressions ---
    original_columns = df.columns
    columns_to_drop = []

    # This pattern looks for columns that start with the prefix, followed by a number.
    # It captures the number so we can check if it's in the desired range.
    # Example: For prefix 'p', it will match 'p13_x' and extract '13'.
    # Example: For prefix 'P#', it will match 'P#25_y' and extract '25'.
    pattern = re.compile(f'^{re.escape(args.prefix)}(\d+)')

    for col in original_columns:
        match = pattern.match(col)
        if match:
            landmark_index = int(match.group(1))
            if args.start_index <= landmark_index <= args.end_index:
                columns_to_drop.append(col)

    if not columns_to_drop:
        print(
            f"No columns found with prefix '{args.prefix}' in the range [{args.start_index}-{args.end_index}]. No changes made.")
        return

    print(f"Found {len(columns_to_drop)} columns to drop:")
    # Print a sample of columns to be dropped for user verification
    for col in columns_to_drop[:5]:
        print(f"  - {col}")
    if len(columns_to_drop) > 5:
        print(f"  ... and {len(columns_to_drop) - 5} more.")

    df_modified = df.drop(columns=columns_to_drop)

    # --- 4. Determine Output Path and Save ---
    output_path = args.output_path if args.output_path else args.input_path

    print(f"\nSaving modified DataFrame with {len(df_modified.columns)} columns to '{output_path}'...")
    try:
        df_modified.to_parquet(output_path, index=False)
        print("Done.")
    except Exception as e:
        print(f"Error: Could not save the new Parquet file. {e}")


if __name__ == "__main__":
    main()