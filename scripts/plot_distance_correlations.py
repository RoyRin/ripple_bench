#!/usr/bin/env python3
"""
Plot distance metrics versus actual utility drop to visualize correlations.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from ripple_bench.utils import read_dict

# Configure matplotlib styling
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 15,  # Set font size to 11pt
    'axes.labelsize': 15,  # -> axis labels
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'text.usetex': False,
    'pgf.rcfonts': False,
})


def plot_metric_vs_utility_drop(topic_stats, output_dir, model_name):
    """Create plots showing each distance metric vs actual utility drop."""
    output_dir = Path(output_dir)

    # Define distance metrics
    rag_distances = topic_stats['distance'].values
    metrics = {
        'RAG Distance': rag_distances,
        'Log(1 + Distance)': np.log1p(rag_distances),
        '1 / (1 + Distance)': 1 / (1 + rag_distances),
        'exp(-Distance/10)': np.exp(-rag_distances / 10),
        'Is WMDP Topic': (rag_distances == 0).astype(float)
    }

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric_name, metric_values) in enumerate(metrics.items()):
        ax = axes[idx]

        # Calculate correlation
        valid_mask = ~np.isnan(metric_values) & ~np.isnan(
            topic_stats['accuracy_delta'])
        corr, p_val = stats.pearsonr(metric_values[valid_mask],
                                     topic_stats['accuracy_delta'][valid_mask])

        # Create scatter plot with density coloring
        if metric_name == 'Is WMDP Topic':
            # For binary metric, use violin plot
            binary_data = [
                topic_stats[metric_values == 0]['accuracy_delta'].values,
                topic_stats[metric_values == 1]['accuracy_delta'].values
            ]
            parts = ax.violinplot(binary_data,
                                  positions=[0, 1],
                                  showmeans=True,
                                  showmedians=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Non-WMDP', 'WMDP'])
            ax.set_xlabel(metric_name, fontsize=12)
        else:
            # For continuous metrics, use hexbin for better visualization of dense data
            hexbin = ax.hexbin(metric_values[valid_mask],
                               topic_stats['accuracy_delta'][valid_mask],
                               gridsize=30,
                               cmap='YlOrRd',
                               mincnt=1)

            # Add trend line
            z = np.polyfit(metric_values[valid_mask],
                           topic_stats['accuracy_delta'][valid_mask], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(metric_values[valid_mask].min(),
                                  metric_values[valid_mask].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

            ax.set_xlabel(metric_name, fontsize=12)

            # Add colorbar for density
            cbar = plt.colorbar(hexbin, ax=ax)
            cbar.set_label('Count', fontsize=10)

        ax.set_ylabel('Accuracy Delta (Base - Unlearned)', fontsize=12)
        ax.set_title(f'{metric_name}\nr = {corr:.3f}, p = {p_val:.2e}',
                     fontsize=14,
                     fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[5])

    # Add main title
    fig.suptitle(
        f'Distance Metrics vs Unlearning Effect - {model_name}\n(Base Accuracy > 40%)',
        fontsize=16,
        fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.85])

    # Save plots
    output_path = output_dir / f'distance_metrics_vs_utility_{model_name.lower()}.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved correlation plots to {output_path}")
    plt.close()

    # Create a summary correlation bar plot
    plt.figure(figsize=(10, 6))

    # Calculate all correlations
    correlations = []
    for metric_name, metric_values in metrics.items():
        valid_mask = ~np.isnan(metric_values) & ~np.isnan(
            topic_stats['accuracy_delta'])
        pearson_r, _ = stats.pearsonr(
            metric_values[valid_mask],
            topic_stats['accuracy_delta'][valid_mask])
        spearman_r, _ = stats.spearmanr(
            metric_values[valid_mask],
            topic_stats['accuracy_delta'][valid_mask])
        correlations.append({
            'Metric': metric_name,
            'Pearson': pearson_r,
            'Spearman': spearman_r
        })

    corr_df = pd.DataFrame(correlations)

    # Create grouped bar plot
    x = np.arange(len(corr_df))
    width = 0.35

    plt.bar(x - width / 2,
            corr_df['Pearson'],
            width,
            label='Pearson',
            alpha=0.8)
    plt.bar(x + width / 2,
            corr_df['Spearman'],
            width,
            label='Spearman',
            alpha=0.8)

    plt.xlabel('Distance Metric', fontsize=14)
    plt.ylabel('Correlation with Accuracy Delta', fontsize=14)
    plt.title(f'Distance Metric Correlations - {model_name}',
              fontsize=16,
              fontweight='bold')
    plt.xticks(x, corr_df['Metric'], rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.85])

    # Save correlation summary
    summary_path = output_dir / f'correlation_summary_{model_name.lower()}.pdf'
    plt.savefig(summary_path, bbox_inches='tight')
    plt.savefig(summary_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved correlation summary to {summary_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot distance metrics vs utility drop")
    parser.add_argument(
        "topic_analysis_csv",
        help="Path to topic analysis CSV from analyze_distance_functions.py")
    parser.add_argument("--output-dir",
                        default="distance_correlation_plots",
                        help="Output directory")
    parser.add_argument("--model-name",
                        default="Model",
                        help="Model name for titles")

    args = parser.parse_args()

    # Load topic analysis
    topic_stats = pd.read_csv(args.topic_analysis_csv, index_col=0)

    # Create plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_vs_utility_drop(topic_stats, output_dir, args.model_name)


if __name__ == "__main__":
    main()
