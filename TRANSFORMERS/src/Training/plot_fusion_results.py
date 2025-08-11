import matplotlib.pyplot as plt
import numpy as np
import math
import os
import argparse


def calculate_approximate_ci(score_pct, n):
    """
    Calculates an approximate 95% confidence interval margin of error for a
    percentage-based score (e.g., F1-score, Accuracy).

    This uses the standard error of a proportion as a robust approximation.

    Args:
        score_pct (float): The observed score as a percentage (e.g., 60.62).
        n (int): The total number of test samples.

    Returns:
        float: The margin of error as a percentage.
    """
    if n == 0 or score_pct < 0 or score_pct > 100:
        return 0

    p = score_pct / 100.0
    z = 1.96  # z-score for 95% confidence

    # Standard error of a proportion: sqrt(p * (1-p) / n)
    se = math.sqrt(p * (1 - p) / n)
    margin_of_error = z * se

    # Return as a percentage
    return margin_of_error * 100


def plot_fusion_comparison(output_dir="plots"):
    """
    Generates and saves a grouped bar chart comparing early and late fusion results,
    grouped by dataset.
    """
    # --- Data from your LaTeX table, reorganized for the new grouping ---
    datasets = ['WLASL100', 'AVASAG100']
    n_samples = {'WLASL100': 258, 'AVASAG100': 890}

    results = {
        'M-F1 (%)': {
            'Early Fusion': [41.13, 71.42],
            'Late Fusion': [60.51, 68.45]
        },
        'W-F1 (%)': {
            'Early Fusion': [41.40, 77.77],
            'Late Fusion': [60.62, 75.12]
        },
        'Accuracy (%)': {
            'Early Fusion': [43.41, 78.43],
            'Late Fusion': [63.57, 75.84]
        },
        'Parameters (M)': {
            'Early Fusion': [5.74, 4.93],
            'Late Fusion': [5.32, 4.68]
        }
    }

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    x = np.arange(len(datasets))
    width = 0.35
    strategies = ['Early Fusion', 'Late Fusion']
    colors = ['#55a868', '#4c72b0']

    for i, metric in enumerate(results.keys()):
        ax = axes[i]

        # Calculate confidence intervals for all performance metrics
        if metric != 'Parameters (M)':
            early_fusion_err = [calculate_approximate_ci(score, n_samples[ds]) for ds, score in
                                zip(datasets, results[metric]['Early Fusion'])]
            late_fusion_err = [calculate_approximate_ci(score, n_samples[ds]) for ds, score in
                               zip(datasets, results[metric]['Late Fusion'])]
        else:
            early_fusion_err, late_fusion_err = None, None

        # Plot bars for each strategy
        rects1 = ax.bar(x - width / 2, results[metric]['Early Fusion'], width, label='Early Fusion', color=colors[0],
                        yerr=early_fusion_err, capsize=5)
        rects2 = ax.bar(x + width / 2, results[metric]['Late Fusion'], width, label='Late Fusion', color=colors[1],
                        yerr=late_fusion_err, capsize=5)

        # Add text labels on bars
        ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=10)
        ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=10)

        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'Comparison by {metric}', fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=12)
        ax.legend()
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

        # Set appropriate y-limits
        if metric != 'Parameters (M)':
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(0, max(results[metric]['Early Fusion'] + results[metric]['Late Fusion']) * 1.1)

    fig.suptitle('Early Fusion vs. Late Fusion Performance Comparison by Dataset', fontsize=20, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Saving ---
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "fusion_strategy_comparison_by_dataset.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Fusion comparison plot saved to: {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot fusion strategy comparison results, grouped by dataset.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save the plot.")
    args = parser.parse_args()
    plot_fusion_comparison(args.output_dir)