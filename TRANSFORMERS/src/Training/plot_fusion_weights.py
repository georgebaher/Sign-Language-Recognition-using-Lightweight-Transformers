import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def plot_learned_weights(output_dir="plots"):
    """
    Generates and saves a stacked bar chart of the learned fusion weights.
    """
    # --- Data from your LaTeX table ---
    datasets = ['WLASL100', 'AVASAG100']
    weights = {
        'Hand': np.array([0.97, 0.88]),
        'Pose': np.array([0.015, 0.01]),
        'Face': np.array([0.015, 0.11]),
    }

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    bottom = np.zeros(len(datasets))
    colors = ['#4c72b0', '#55a868', '#c44e52']

    for (modality, weight_values), color in zip(weights.items(), colors):
        p = ax.bar(datasets, weight_values, label=modality, bottom=bottom, width=0.5, color=color)

        # Add text annotations inside the bars
        for i, rect in enumerate(p):
            h = rect.get_height()
            if h > 0.03:  # Only add label if the bar is tall enough
                ax.text(rect.get_x() + rect.get_width() / 2., bottom[i] + h / 2.,
                        f'{h:.2%}', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

        bottom += weight_values

    # --- Formatting ---
    ax.set_title('Learned Modality Weights by the Late Fusion Model', fontsize=16, pad=20)
    ax.set_ylabel('Proportion of Total Weight', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(title='Modality', loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.tick_params(axis='x', labelsize=12)
    ax.set_yticks([])  # Remove y-axis ticks as labels are on bars

    fig.tight_layout()

    # --- Saving ---
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "learned_fusion_weights.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Learned weights plot saved to: {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot learned fusion weights.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save the plot.")
    args = parser.parse_args()
    plot_learned_weights(args.output_dir)