import json
import matplotlib.pyplot as plt
import os
import argparse
from collections import defaultdict


def plot_dataset_distribution(json_path: str, n_glosses: int = None):
    """
    Reads a metadata JSON file, calculates the train/val/test split distribution for a
    specified number of glosses, and saves a stacked bar chart visualizing the results.

    Args:
        json_path (str): The full path to the input metadata JSON file.
        n_glosses (int, optional): The number of glosses to process from the top of the file.
                                   If None, all glosses are processed.
    """
    # --- 1. Validate Input Path ---
    if not os.path.exists(json_path):
        print(f"Error: The file was not found at the specified path: {json_path}")
        return

    print(f"--- Processing metadata from: {os.path.basename(json_path)} ---")

    # --- 2. Load and Process the JSON Data ---
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # --- NEW: Slice the metadata if n_glosses is specified ---
    if n_glosses is not None and n_glosses > 0:
        print(f"Processing only the first {n_glosses} glosses.")
        metadata = metadata[:n_glosses]
    elif n_glosses is not None:
        print(f"Warning: --n_glosses must be a positive integer. Processing all glosses instead.")

    gloss_stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    for gloss_entry in metadata:
        gloss_name = gloss_entry.get("gloss")
        if not gloss_name:
            continue

        for instance in gloss_entry.get("instances", []):
            split = instance.get("split")
            if split in gloss_stats[gloss_name]:
                gloss_stats[gloss_name][split] += 1

    if not gloss_stats:
        print("Error: No valid glosses or instances found in the JSON file for the given selection.")
        return

    print(f"Found {len(gloss_stats)} glosses to plot.")

    # --- 3. Prepare Data for Plotting ---
    sorted_glosses = sorted(
        gloss_stats.keys(),
        key=lambda g: sum(gloss_stats[g].values()),
        reverse=True
    )

    train_counts = [gloss_stats[g]["train"] for g in sorted_glosses]
    val_counts = [gloss_stats[g]["val"] for g in sorted_glosses]
    test_counts = [gloss_stats[g]["test"] for g in sorted_glosses]
    val_bottom = [train + val for train, val in zip(train_counts, val_counts)]

    # --- 4. Generate the Stacked Bar Chart ---
    x = range(len(sorted_glosses))
    bar_width = 0.8
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.bar(x, train_counts, label="Train", width=bar_width, color='#4c72b0')
    ax.bar(x, val_counts, bottom=train_counts, label="Validation", width=bar_width, color='#55a868')
    ax.bar(x, test_counts, bottom=val_bottom, label="Test", width=bar_width, color='#c44e52')

    # --- 5. Format the Plot with a Dynamic Title ---
    base_filename = os.path.basename(json_path)
    title_text = f"Instance Distribution for Top {len(sorted_glosses)} Glosses from '{base_filename}'" \
        if n_glosses else f"Instance Distribution per Gloss for '{base_filename}'"

    ax.set_xlabel("Gloss", fontsize=14)
    ax.set_ylabel("Number of Instances", fontsize=14)
    ax.set_title(title_text, fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_glosses, rotation=90, fontsize=10)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)
    ax.margins(x=0.01)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    fig.tight_layout()

    # --- 6. Save the Plot with a Dynamic Filename ---
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_dir = os.path.dirname(json_path)

    suffix = f"_top_{n_glosses}_distribution.png" if n_glosses else "_full_distribution.png"
    plot_path = os.path.join(output_dir, f"{base_name}{suffix}")

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    print(f"\nSuccessfully generated and saved plot to: {plot_path}")


def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Generate a stacked bar chart showing the train/validation/test distribution from a metadata JSON file."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the input metadata JSON file."
    )
    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--n_glosses",
        type=int,
        default=None,
        help="Number of glosses to process from the top of the file. If not specified, all glosses will be processed."
    )
    args = parser.parse_args()
    plot_dataset_distribution(args.json_path, args.n_glosses)


if __name__ == "__main__":
    main()