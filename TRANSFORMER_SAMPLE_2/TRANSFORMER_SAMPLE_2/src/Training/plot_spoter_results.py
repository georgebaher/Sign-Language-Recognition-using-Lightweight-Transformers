import matplotlib.pyplot as plt
import numpy as np

# Data
features = [
    "HAND_LANDMARKS", "POSE_LANDMARKS", "HAND_POSE_LANDMARKS",
    "HAND_ANGLES", "POSE_ANGLES", "HAND_POSE_ANGLES"
]

spoter_acc = [66.67, 27.52, 55.04, 62.02, 27.13, 62.79]
spoter_ci_low = [60.69, 22.43, 48.94, 55.95, 22.08, 56.74]
spoter_ci_high = [72.13, 33.29, 60.99, 67.71, 32.89, 68.45]

# Compute error bars
spoter_err = [
    [acc - low for acc, low in zip(spoter_acc, spoter_ci_low)],
    [high - acc for acc, high in zip(spoter_acc, spoter_ci_high)]
]

# Plot
x = np.arange(len(features))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
spoter = ax.bar(x + width/2, spoter_acc, width, yerr=spoter_err, label='SPOTER', capsize=5)

# Add accuracy values on top of bars
for rect, acc in zip(spoter, spoter_acc):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 1.5, f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Labels and formatting
ax.set_ylabel('Top Accuracy (%)')
ax.set_title('Model Comparison with Different Input Features')
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=20)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("spoter_results.png", dpi=300)
