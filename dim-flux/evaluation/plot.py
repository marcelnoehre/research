import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load your data
df = pd.read_csv('lattice_metrics_results.csv')

# Define your metric columns
metrics = ['AR', 'Asp', 'CA', 'EC', 'EL', 'EO', 'KSM', 'NP', 'NEO', 'NR', 'NU']

# Set global aesthetic
sns.set_theme(style="whitegrid")

def plot_lattice_analysis(df, metrics):
    # --- FIGURE 1: Boxplots (Distributions) ---
    num_metrics = len(metrics)
    cols = 3
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        sns.boxplot(data=df, x='algo', y=metric, ax=axes[i], palette='viridis', fliersize=2)
        axes[i].set_title(f'Metric: {metric}', fontweight='bold')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        axes[i].set_xlabel('')
        
    # Clean up empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()

    # --- FIGURE 2: Heatmap (Average Comparison) ---
    plt.figure(figsize=(12, 6))
    summary = df.groupby('algo')[metrics].mean()
    
    # Normalize 0-1 so metrics with different scales (e.g., 0.1 vs 100) can be compared
    normalized_summary = (summary - summary.min()) / (summary.max() - summary.min())
    
    sns.heatmap(normalized_summary, annot=summary, fmt=".2f", cmap="YlGnBu")
    plt.title('Average Performance Heatmap (Darker = Higher Value)', fontsize=15)
    plt.ylabel('Drawing Method')
    plt.tight_layout()
    plt.show()

# Run visualization
plot_lattice_analysis(df, metrics)

# Print raw averages for verification
print("Mean Scores per Algorithm:")
print(df.groupby('algo')[metrics].mean().to_string())