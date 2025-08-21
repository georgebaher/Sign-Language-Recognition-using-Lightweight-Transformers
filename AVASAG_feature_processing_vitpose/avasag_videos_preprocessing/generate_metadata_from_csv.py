import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import argparse
from dotenv import load_dotenv

load_dotenv()


def generate_metadata_and_stats(csv_path, output_json_path, stats_txt_path):
    # read CSV with semicolon separator
    df = pd.read_csv(csv_path, sep=";")

    # group by gloss_name
    metadata = {}
    gloss_stats = {}

    for _, row in df.iterrows():
        gloss_name = row["gloss_name"]
        gloss_idx = int(row["gloss_idx"])
        video_id = int(row["video_id"])
        split = row["split"]

        final_video_id = f"{video_id:04d}_{gloss_idx:04d}"
        instance = {"video_id": final_video_id, "split": split}

        if gloss_name not in metadata:
            metadata[gloss_name] = {"gloss": gloss_name, "instances": []}
            gloss_stats[gloss_name] = {"train": 0, "val": 0, "test": 0}

        metadata[gloss_name]["instances"].append(instance)
        gloss_stats[gloss_name][split] += 1

    # save metadata JSON
    metadata_list = list(metadata.values())
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    # compute and save stats
    total_glosses = len(metadata_list)
    total_instances = sum(len(g["instances"]) for g in metadata_list)

    stats_lines = [
        f"Total number of glosses: {total_glosses}",
        f"Total number of instances: {total_instances}\n"
    ]
    for gloss, counts in gloss_stats.items():
        total = sum(counts.values())
        stats_lines.append(
            f"Gloss '{gloss}': {total} instances "
            f"(train: {counts['train']}, val: {counts['val']}, test: {counts['test']})"
        )
    stats_text = "\n".join(stats_lines)
    print(stats_text)

    os.makedirs(os.path.dirname(stats_txt_path), exist_ok=True)
    with open(stats_txt_path, "w", encoding="utf-8") as f:
        f.write(stats_text)

    # plot bar chart
    sorted_glosses = sorted(gloss_stats, key=lambda g: sum(gloss_stats[g].values()), reverse=True)
    train_counts = [gloss_stats[g]["train"] for g in sorted_glosses]
    val_counts = [gloss_stats[g]["val"] for g in sorted_glosses]
    test_counts = [gloss_stats[g]["test"] for g in sorted_glosses]

    x = range(len(sorted_glosses))
    bar_width = 0.8
    plt.figure(figsize=(16, 8))
    plt.bar(x, train_counts, label="train", width=bar_width)
    plt.bar(x, val_counts, bottom=train_counts, label="val", width=bar_width)
    plt.bar(x, test_counts, bottom=[train_counts[i] + val_counts[i] for i in x], label="test", width=bar_width)
    plt.xlabel("Gloss")
    plt.ylabel("Number of instances")
    plt.title("Instances per gloss (stacked by split)")
    plt.xticks(x, sorted_glosses, rotation=90)
    plt.legend()
    plt.tight_layout()

    plot_path = stats_txt_path.replace(".txt", ".png")
    plt.savefig(plot_path, dpi=300)

    print(f"\nMetadata JSON saved to {output_json_path}")
    print(f"Stats saved to {stats_txt_path}")
    print(f"Bar chart saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata JSON and statistics from AVASAG CSV.")
    parser.add_argument("--csv", required=True, help="Path to input CSV file (e.g., AVASAG_100_v0.0.csv)")
    parser.add_argument("--json", required=True, help="Path to output metadata JSON file")
    parser.add_argument("--stats", required=True, help="Path to output stats text file (also used for bar chart)")

    args = parser.parse_args()

    generate_metadata_and_stats(args.csv, args.json, args.stats)
