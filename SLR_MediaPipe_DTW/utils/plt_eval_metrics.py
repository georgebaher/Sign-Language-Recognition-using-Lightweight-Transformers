import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_dtw_evaluation_results(df: pd.DataFrame):
    sns.set(style="whitegrid", font_scale=1.2)

    # --- Accuracy Plot ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="n_glosses", y="accuracy", hue="feature_option", marker="o")

    # Annotate max points
    for option in df['feature_option'].unique():
        sub_df = df[df['feature_option'] == option]
        max_row = sub_df.loc[sub_df['accuracy'].idxmax()]
        plt.text(max_row['n_glosses'], max_row['accuracy'] + 1,
                 f"{max_row['accuracy']:.1f}%", ha='center', fontsize=10)

    plt.title("DTW Accuracy vs. Number of Glosses")
    plt.xlabel("Number of Glosses")
    plt.ylabel("Accuracy (%)")
    plt.legend(title="Feature Option")
    plt.tight_layout()
    plt.show()

    # --- Time Plot ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="n_glosses", y="avg_dtw_time_sec", hue="feature_option", marker="o")

    # Annotate highest times
    for option in df['feature_option'].unique():
        sub_df = df[df['feature_option'] == option]
        max_row = sub_df.loc[sub_df['avg_dtw_time_sec'].idxmax()]
        plt.text(max_row['n_glosses'], max_row['avg_dtw_time_sec'] + 0.002,
                 f"{max_row['avg_dtw_time_sec']:.3f}s", ha='center', fontsize=10)

    plt.title("Average DTW Time per Test vs. Number of Glosses")
    plt.xlabel("Number of Glosses")
    plt.ylabel("Average DTW Time (s)")
    plt.legend(title="Feature Option")
    plt.tight_layout()
    plt.show()
