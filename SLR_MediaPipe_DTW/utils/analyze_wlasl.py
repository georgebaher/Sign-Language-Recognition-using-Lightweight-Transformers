import json
from collections import defaultdict
import os

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def get_train_val_test_split_per_gloss(json_path, n_glosses, plot=False):
    with open(json_path, 'r') as f:
        all_glosses = json.load(f)

    selected_glosses = all_glosses[:n_glosses]

    gloss_split_counts = defaultdict(lambda: {'train': 0, 'test': 0, 'val': 0})
    gloss_video_ids = defaultdict(lambda: {'train': [], 'test': [], 'val': []})

    for gloss_entry in selected_glosses:
        gloss_name = gloss_entry['gloss']
        for instance in gloss_entry['instances']:
            split = instance.get('split', 'train')  # default to 'train'
            if split in gloss_split_counts[gloss_name]:
                gloss_split_counts[gloss_name][split] += 1
                gloss_video_ids[gloss_name][split].append(instance['video_id'])

    if plot:
        plot_stats(gloss_split_counts)

    return gloss_video_ids


def plot_stats(gloss_split_counts):
    glosses = list(gloss_split_counts.keys())
    train_counts = [gloss_split_counts[g]['train'] for g in glosses]
    test_counts = [gloss_split_counts[g]['test'] for g in glosses]
    val_counts = [gloss_split_counts[g]['val'] for g in glosses]

    x = range(len(glosses))
    plt.figure(figsize=(18, 9))
    plt.bar(x, train_counts, label='Train', color='skyblue')
    plt.bar(x, test_counts, bottom=train_counts, label='Test', color='salmon')
    plt.bar(x, val_counts, bottom=np.array(train_counts) + np.array(test_counts), label='Val', color='lightgreen')

    plt.xticks(x, glosses, rotation=90)
    plt.ylabel('Number of Instances')
    plt.title(f'Train/Test/Val Instance Counts for First {len(glosses)} Glosses')
    plt.legend()
    plt.tight_layout()

    # Annotate bars
    for i, (train, test, val) in enumerate(zip(train_counts, test_counts, val_counts)):
        if train > 0:
            plt.text(i, train / 2, str(train), ha='center', va='center', fontsize=8, color='black')
        if test > 0:
            plt.text(i, train + test / 2, str(test), ha='center', va='center', fontsize=8, color='black')
        if val > 0:
            plt.text(i, train + test + val / 2, str(val), ha='center', va='center', fontsize=8, color='black')

    plt.show()

    # ➕ Bell curve histograms
    def plot_bell_curve(data, label, color):
        mu, std = np.mean(data), np.std(data)
        xmin, xmax = min(data), max(data)
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)

        plt.figure(figsize=(8, 5))
        plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal')
        plt.hist(data, bins=20, density=True, alpha=0.5, color=color, edgecolor='black', label='Histogram')
        plt.title(f"{label} Sample Count Distribution\nμ = {mu:.2f}, σ = {std:.2f}")
        plt.xlabel("Number of Samples per Gloss")
        plt.ylabel("Probability Density")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_bell_curve(train_counts, "Train", "skyblue")
    plot_bell_curve(test_counts, "Test", "salmon")
    plot_bell_curve(val_counts, "Val", "lightgreen")


if __name__ == '__main__':
    # TESTING...
    load_dotenv()
    metadata_path = os.getenv("WLASL_METADATA_PATH")

    if not metadata_path or not os.path.exists(metadata_path):
        print("❌ Invalid or missing wlasl_csv_exports path. Set WLASL_METADATA_PATH in your .env file.")
        exit(1)

    result = get_train_val_test_split_per_gloss(metadata_path, 100, plot=True)

