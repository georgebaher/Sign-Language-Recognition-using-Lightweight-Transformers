import matplotlib.pyplot as plt
import numpy as np

# Data
features = [
    "HAND_LANDMARKS", "POSE_LANDMARKS", "HAND_POSE_LANDMARKS",
    "HAND_ANGLES", "POSE_ANGLES", "HAND_POSE_ANGLES"
]

baseline_acc = [64.73, 24.42, 50.00, 62.40, 24.03, 61.24]
spoter_acc = [66.67, 27.52, 55.04, 62.02, 27.13, 62.79]

baseline_ci_low = [58.71, 19.58, 43.95, 56.34, 19.22, 55.16]
baseline_ci_high = [70.30, 30.04, 56.05, 68.08, 29.63, 66.97]

spoter_ci_low = [60.69, 22.43, 48.94, 55.95, 22.08, 56.74]
spoter_ci_high = [72.13, 33.29, 60.99, 67.71, 32.89, 68.45]

# Compute error bars
baseline_err = [  # error = upper - mean
    [acc - low for acc, low in zip(baseline_acc, baseline_ci_low)],
    [high - acc for acc, high in zip(baseline_acc, baseline_ci_high)]
]

spoter_err = [
    [acc - low for acc, low in zip(spoter_acc, spoter_ci_low)],
    [high - acc for acc, high in zip(spoter_acc, spoter_ci_high)]
]

# Plot
x = np.arange(len(features))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

# Bars with error bars
baseline = ax.bar(x - width/2, baseline_acc, width, yerr=baseline_err, label='Baseline', capsize=5)
spoter = ax.bar(x + width/2, spoter_acc, width, yerr=spoter_err, label='SPOTER', capsize=5)

# Labels and formatting
ax.set_ylabel('Top Accuracy (%)')
ax.set_title('Model Comparison with Confidence Intervals')
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=20)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
