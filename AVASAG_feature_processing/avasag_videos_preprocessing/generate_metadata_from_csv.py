import pandas as pd
import json
import matplotlib.pyplot as plt

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

        instance = {
            "video_id": final_video_id,
            "split": split
        }

        if gloss_name not in metadata:
            metadata[gloss_name] = {
                "gloss": gloss_name,
                "instances": []
            }
            gloss_stats[gloss_name] = {"train": 0, "val": 0, "test": 0}

        metadata[gloss_name]["instances"].append(instance)
        gloss_stats[gloss_name][split] += 1

    # convert to list
    metadata_list = list(metadata.values())

    # save JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    # print & save stats
    total_glosses = len(metadata_list)
    total_instances = sum(len(gloss["instances"]) for gloss in metadata_list)

    stats_lines = []
    stats_lines.append(f"Total number of glosses: {total_glosses}")
    stats_lines.append(f"Total number of instances: {total_instances}\n")

    for gloss, counts in gloss_stats.items():
        gloss_total = sum(counts.values())
        stats_lines.append(
            f"Gloss '{gloss}': {gloss_total} instances "
            f"(train: {counts['train']}, val: {counts['val']}, test: {counts['test']})"
        )

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    with open(stats_txt_path, "w", encoding="utf-8") as f:
        f.write(stats_text)

    # prepare for bar chart
    gloss_totals = {gloss: sum(counts.values()) for gloss, counts in gloss_stats.items()}
    # sort by decreasing total
    sorted_glosses = sorted(gloss_totals.keys(), key=lambda g: gloss_totals[g], reverse=True)

    train_counts = [gloss_stats[g]["train"] for g in sorted_glosses]
    val_counts = [gloss_stats[g]["val"] for g in sorted_glosses]
    test_counts = [gloss_stats[g]["test"] for g in sorted_glosses]

    bar_width = 0.8  # leave some spacing
    x = range(len(sorted_glosses))

    plt.figure(figsize=(16, 8))
    plt.bar(x, train_counts, label="train", width=bar_width)
    plt.bar(x, val_counts, bottom=train_counts, label="val", width=bar_width)
    bottom_train_val = [train_counts[i] + val_counts[i] for i in range(len(train_counts))]
    plt.bar(x, test_counts, bottom=bottom_train_val, label="test", width=bar_width)

    plt.xlabel("Gloss")
    plt.ylabel("Number of instances")
    plt.title("Instances per gloss (stacked by split)")
    plt.xticks(x, sorted_glosses, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(stats_txt_path.replace(".txt", ".png"), dpi=300)
    # plt.show()

    print(f"\n? Metadata JSON saved to {output_json_path}")
    print(f"? Stats saved to {stats_txt_path}")
    print(f"? Bar chart saved to {stats_txt_path.replace('.txt', '.png')}")

if __name__ == "__main__":
    csv_path = r"C:\Users\boulosge\Downloads\AVASAG_100_v0.0.csv"
    output_json_path = r"C:\Users\boulosge\Desktop\Acht\AVASAG100\avasag_metadata.json"
    stats_txt_path = r"C:\Users\boulosge\Desktop\Acht\AVASAG100\avasag_stats.txt"
    generate_metadata_and_stats(csv_path, output_json_path, stats_txt_path)
