import json
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import numpy as np
from dotenv import load_dotenv
from scipy.stats import norm


def analyze_wlasl_metadata(json_path, n_glosses, plot=False):
    # Load JSON
    with open(json_path, 'r') as f:
        all_glosses = json.load(f)

    selected_glosses = all_glosses[:n_glosses]

    gloss_split_counts = defaultdict(lambda: {'train': 0, 'test': 0})
    gloss_video_ids = defaultdict(lambda: {'train': [], 'test': []})

    for gloss_entry in selected_glosses:
        gloss_name = gloss_entry['gloss']
        for instance in gloss_entry['instances']:
            split = instance['split']
            video_id = instance['video_id']
            if split in ['train', 'test']:
                gloss_split_counts[gloss_name][split] += 1
                gloss_video_ids[gloss_name][split].append(video_id)

    if plot:
        plot_stats(gloss_split_counts)

    return gloss_video_ids

def plot_stats(gloss_split_counts):
    # Plot per-gloss train/test bars
    glosses = list(gloss_split_counts.keys())
    train_counts = [gloss_split_counts[g]['train'] for g in glosses]
    test_counts = [gloss_split_counts[g]['test'] for g in glosses]

    x = range(len(glosses))
    plt.figure(figsize=(18, 9))
    plt.bar(x, train_counts, label='Train')
    plt.bar(x, test_counts, bottom=train_counts, label='Test')
    plt.xticks(x, glosses, rotation=90)
    plt.ylabel('Number of Instances')
    plt.title(f'Train/Test Instance Counts for First {len(glosses)} Glosses')
    plt.legend()
    plt.tight_layout()

    for i, (train, test) in enumerate(zip(train_counts, test_counts)):
        if train > 0:
            plt.text(i, train / 2, str(train), ha='center', va='center', fontsize=8, color='white')
        if test > 0:
            plt.text(i, train + test / 2, str(test), ha='center', va='center', fontsize=8, color='white')

    plt.show()

    # ➕ Bell curve histograms
    def plot_bell_curve(data, label, color):
        mu, std = np.mean(data), np.std(data)
        xmin, xmax = min(data), max(data)
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)

        plt.figure(figsize=(8, 5))
        plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal')
        plt.title(f"{label} Sample Count Distribution\nμ = {mu:.2f}, σ = {std:.2f}")
        plt.xlabel("Number of Samples per Gloss")
        plt.ylabel("Probability Density")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_bell_curve(train_counts, "Train", "skyblue")
    plot_bell_curve(test_counts, "Test", "salmon")

if __name__ == '__main__':
    load_dotenv()
    metadata_path = os.getenv("WLASL_METADATA_PATH")

    if not metadata_path or not os.path.exists(metadata_path):
        print("❌ Invalid or missing metadata path. Set WLASL_METADATA_PATH in your .env file.")
        exit(1)

    result = analyze_wlasl_metadata(metadata_path, 100, plot=True)
