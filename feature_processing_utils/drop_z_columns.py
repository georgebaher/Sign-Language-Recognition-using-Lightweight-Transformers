# A command-line utility to read a Parquet file, drop all columns whose
# names end with '_z', and save the result to a new file or overwrite
# the original.

import pandas as pd
import os
from argparse import ArgumentParser


def main():
    """Main function to run the column dropping process."""
    parser = ArgumentParser(
        description="A script to drop columns ending with '_z' from a Parquet file."
    )

    # --- Command-Line Arguments ---
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input Parquet file that needs to be processed."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to save the new, modified Parquet file. If this is not provided, the original input file will be overwritten."
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

    # --- 3. Identify and Drop Columns ---
    original_columns = df.columns
    columns_to_drop = [col for col in original_columns if col.endswith('_z')]

    if not columns_to_drop:
        print("No columns ending with '_z' found in the file. No changes made.")
        return

    print(f"Found {len(columns_to_drop)} columns to drop:")
    for col in columns_to_drop:
        print(f"  - {col}")

    df_modified = df.drop(columns=columns_to_drop)

    # --- 4. Determine Output Path and Save ---
    # If an output path is provided, use it. Otherwise, use the input path to overwrite.
    output_path = args.output if args.output else args.input_path

    print(f"\nSaving modified DataFrame with {len(df_modified.columns)} columns to '{output_path}'...")
    try:
        df_modified.to_parquet(output_path, index=False)
        print("Done.")
    except Exception as e:
        print(f"Error: Could not save the new Parquet file. {e}")


if __name__ == "__main__":
    main()