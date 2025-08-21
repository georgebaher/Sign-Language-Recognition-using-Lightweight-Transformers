# check_parquet.py
#
# A simple command-line script to quickly display the dimensions
# (rows, columns) and file size of a Parquet file.

import pandas as pd
import os
from argparse import ArgumentParser


def format_size(size_bytes):
    """Converts a size in bytes to a human-readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.2f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"


def main():
    """Main function to run the script."""
    parser = ArgumentParser(
        description="Check the dimensions (rows, columns) and size of a Parquet file."
    )

    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the Parquet file you want to inspect."
    )

    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File not found at '{args.filepath}'")
        return

    try:
        # --- NEW: Get file size and format it ---
        file_size_bytes = os.path.getsize(args.filepath)
        file_size_formatted = format_size(file_size_bytes)

        print(f"Reading '{os.path.basename(args.filepath)}'...")
        df = pd.read_parquet(args.filepath)
        rows, cols = df.shape

        print("\n--- File Info ---")
        print(f"Rows:    {rows}")
        print(f"Columns: {cols}")
        print(f"Size:    {file_size_formatted}")  # <-- NEW
        print("-----------------")

    except Exception as e:
        print(f"\nError: Could not read the file. It may be corrupted or not a valid Parquet file.")
        print(f"Details: {e}")


if __name__ == "__main__":
    main()