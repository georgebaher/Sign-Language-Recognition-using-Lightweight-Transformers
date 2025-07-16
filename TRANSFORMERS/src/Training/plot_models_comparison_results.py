import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

# Paste in your results:
models = [
    "baseline_transformer",
    "spoter",
    "encoder",
    "lstm"
]
acc_means = [46.90, 55.04, 53.88, 16.28]
acc_errors = [6.04, 6.02, 6.04, 4.53]

m_f1_means = [44.01, 53.24, 51.58, 13.34]
m_f1_errors = [6.01, 6.04, 6.05, 4.19]

w_f1_means = [44.73, 53.35, 51.55, 14.28]
w_f1_errors = [6.02, 6.04, 6.05, 4.30]

params = [8_074_788, 8_142_380, 3_887_572, 162_756]
times = [629.6, 560.8, 340.8, 2728.9]

# Plot 1: Top accuracy with confidence interval
plt.figure()
bars = plt.bar(models, acc_means, yerr=acc_errors, capsize=5)
plt.ylabel("Top Accuracy (%)")
plt.title("Model Top Accuracy with 95% CI")
plt.grid(axis="y", linestyle="--", alpha=0.7)
for bar, val in zip(bars, acc_means):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}%",
             ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("out-img/__models_comparison/_charts/models_top_acc.png")

# Plot 2: Macro F1 with confidence interval
plt.figure()
bars = plt.bar(models, m_f1_means, yerr=m_f1_errors, capsize=5)
plt.ylabel("Macro F1 (%)")
plt.title("Model Macro F1 with 95% CI")
plt.grid(axis="y", linestyle="--", alpha=0.7)
for bar, val in zip(bars, m_f1_means):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}%",
             ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("out-img/__models_comparison/_charts/models_macro_f1.png")

# Plot 3: Weighted F1 with confidence interval
plt.figure()
bars = plt.bar(models, w_f1_means, yerr=w_f1_errors, capsize=5)
plt.ylabel("Weighted F1 (%)")
plt.title("Model Weighted F1 with 95% CI")
plt.grid(axis="y", linestyle="--", alpha=0.7)
for bar, val in zip(bars, w_f1_means):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}%",
             ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("out-img/__models_comparison/_charts/models_weighted_f1.png")

# Plot 4: Parameters
plt.figure()
param_millions = [p/1e6 for p in params]
bars = plt.bar(models, param_millions)
plt.ylabel("Parameters (Millions)")
plt.title("Model Parameter Count")
plt.grid(axis="y", linestyle="--", alpha=0.7)
for bar, val in zip(bars, param_millions):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{val:.2f}M",
             ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("out-img/__models_comparison/_charts/models_params.png")

# Plot 5: Elapsed time
plt.figure()
bars = plt.bar(models, times)
plt.ylabel("Elapsed Time (s)")
plt.title("Model Training Time")
plt.grid(axis="y", linestyle="--", alpha=0.7)
for bar, val in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f"{val:.0f}s",
             ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("out-img/__models_comparison/_charts/models_times.png")

print("✅ Plots saved to out-img/")
plt.show()
