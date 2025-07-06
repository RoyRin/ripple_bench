#!/usr/bin/env python3
"""
Plot ripple effect: accuracy delta vs distance from unlearned WMDP topics
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
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


def plot_ripple_effect(base_csv,
                       unlearned_csvs,
                       dataset_path,
                       output_dir,
                       min_base_accuracy=0.4):
    """Plot accuracy delta vs distance for multiple unlearned models.
    
    Args:
        base_csv: Path to base model results
        unlearned_csvs: List of (csv_path, label) tuples for unlearned models
        dataset_path: Path to dataset JSON
        output_dir: Output directory for plots
        min_base_accuracy: Minimum base accuracy to include topic (default: 0.4)
    """

    # Load dataset to get topic-to-neighbors mapping
    print("Loading dataset...")
    dataset = read_dict(dataset_path)
    topic_to_neighbors = dataset.get(
        'topic_to_neighbors',
        dataset.get('raw_data', {}).get('topic_to_neighbors', {}))

    if not topic_to_neighbors:
        raise ValueError("Cannot find topic_to_neighbors in dataset")

    # Load base model results
    print("Loading base model results...")
    base_df = pd.read_csv(base_csv)

    # Calculate base accuracy per topic for filtering
    base_topic_acc = base_df.groupby('topic')['is_correct'].mean()
    high_acc_topics = base_topic_acc[base_topic_acc > min_base_accuracy].index
    print(
        f"Filtering to {len(high_acc_topics)} topics with base accuracy > {min_base_accuracy*100:.0f}%"
    )

    # Filter base_df to only include high accuracy topics
    base_df = base_df[base_df['topic'].isin(high_acc_topics)]

    # Set up plot
    plt.figure(figsize=(12, 8))
    colors = ['#E63946', '#2E86AB', '#F77F00', '#06D6A0', '#7209B7']
    markers = ['o', 's', '^', 'D', 'v']

    for idx, (unlearned_csv, label) in enumerate(unlearned_csvs):
        print(f"\nProcessing {label}...")
        unlearned_df = pd.read_csv(unlearned_csv)

        # Filter unlearned_df to match filtered base_df
        unlearned_df = unlearned_df[unlearned_df['question_id'].isin(
            base_df['question_id'])]

        # Merge dataframes - include distance from base_df
        merged = pd.merge(
            base_df[['question_id', 'topic', 'is_correct', 'distance']],
            unlearned_df[['question_id', 'is_correct']],
            on='question_id',
            suffixes=('_base', '_unlearned'))

        # Use the distance directly from the CSV (no need to recalculate)

        # Calculate accuracy by distance
        distance_stats = merged.groupby('distance').agg({
            'is_correct_base':
            'mean',
            'is_correct_unlearned':
            'mean',
            'question_id':
            'count'
        }).rename(columns={'question_id': 'count'})

        # Calculate delta (base - unlearned)
        distance_stats['accuracy_delta'] = distance_stats[
            'is_correct_base'] - distance_stats['is_correct_unlearned']

        # Plot with error bars
        distances = distance_stats.index.tolist()
        deltas = distance_stats['accuracy_delta'].tolist()
        counts = distance_stats['count'].tolist()

        # Calculate standard error for each distance
        std_errors = []
        for dist in distances:
            dist_data = merged[merged['distance'] == dist]
            base_correct = dist_data['is_correct_base'].values.astype(int)
            unlearned_correct = dist_data[
                'is_correct_unlearned'].values.astype(int)
            delta_per_question = base_correct - unlearned_correct
            if len(delta_per_question) > 1:
                std_error = delta_per_question.std() / np.sqrt(
                    len(delta_per_question))
            else:
                std_error = 0
            std_errors.append(std_error)

        # Plot with error bars
        plt.errorbar(distances,
                     deltas,
                     yerr=std_errors,
                     label=label,
                     color=colors[idx % len(colors)],
                     marker=markers[idx % len(markers)],
                     markersize=8,
                     linewidth=2,
                     capsize=5,
                     alpha=0.8)

        # Add sample size annotations for first model only (commented out to avoid clutter)
        # if idx == 0:
        #     for i, (dist, count) in enumerate(zip(distances[:20], counts[:20])):  # First 20 distances
        #         plt.text(dist, -0.05, f'n={count}', ha='center', va='top',
        #                 fontsize=8, alpha=0.6, rotation=45)

    # Formatting
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Distance from Original WMDP Topics', fontsize=14)
    plt.ylabel('Accuracy Delta (Base - Unlearned)', fontsize=14)
    plt.title('Ripple Effect: Performance Drop vs Semantic Distance',
              fontsize=16,
              fontweight='bold')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Set x-axis limits to show all distances
    plt.xlim(-1, max(distances) + 1)

    # Add annotation
    plt.text(
        0.02,
        0.98,
        'Distance 0 = Original WMDP topics (directly unlearned)\nDistance 1+ = Semantically related topics',
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.85])

    # Save plot in both PNG and PDF formats
    output_path_png = Path(output_dir) / 'ripple_effect_comparison.png'
    output_path_pdf = Path(output_dir) / 'ripple_effect_comparison.pdf'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"\nPlots saved to:")
    print(f"  PNG: {output_path_png}")
    print(f"  PDF: {output_path_pdf}")

    # Also save a zoomed-in version focusing on distances 0-10
    plt.figure(figsize=(12, 8))
    for idx, (unlearned_csv, label) in enumerate(unlearned_csvs):
        unlearned_df = pd.read_csv(unlearned_csv)
        # Filter to match base_df
        unlearned_df = unlearned_df[unlearned_df['question_id'].isin(
            base_df['question_id'])]
        merged = pd.merge(
            base_df[['question_id', 'topic', 'is_correct', 'distance']],
            unlearned_df[['question_id', 'is_correct']],
            on='question_id',
            suffixes=('_base', '_unlearned'))

        # Use distance from CSV (already included in merge above)
        distance_stats = merged.groupby('distance').agg({
            'is_correct_base':
            'mean',
            'is_correct_unlearned':
            'mean'
        })
        distance_stats['accuracy_delta'] = distance_stats[
            'is_correct_base'] - distance_stats['is_correct_unlearned']

        # Filter to distances 0-10
        distance_stats_filtered = distance_stats[distance_stats.index <= 10]

        plt.plot(distance_stats_filtered.index,
                 distance_stats_filtered['accuracy_delta'],
                 'o-',
                 label=label,
                 color=colors[idx % len(colors)],
                 marker=markers[idx % len(markers)],
                 markersize=10,
                 linewidth=2.5)

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Distance from Original WMDP Topics', fontsize=14)
    plt.ylabel('Accuracy Delta (Base - Unlearned)', fontsize=14)
    plt.title('Ripple Effect: Close-up View (Distances 0-10)',
              fontsize=16,
              fontweight='bold')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 10.5)
    plt.xticks(range(11))

    plt.tight_layout(rect=[0, 0.03, 1, 0.85])
    output_path_zoom_png = Path(
        output_dir) / 'ripple_effect_comparison_zoom.png'
    output_path_zoom_pdf = Path(
        output_dir) / 'ripple_effect_comparison_zoom.pdf'
    plt.savefig(output_path_zoom_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_zoom_pdf, bbox_inches='tight')
    print(f"Zoomed plots saved to:")
    print(f"  PNG: {output_path_zoom_png}")
    print(f"  PDF: {output_path_zoom_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ripple effect visualization")
    parser.add_argument("base_csv", help="Path to base model CSV results")
    parser.add_argument("--elm", help="Path to ELM model CSV results")
    parser.add_argument("--rmu", help="Path to RMU model CSV results")
    parser.add_argument("--dataset",
                        required=True,
                        help="Path to ripple bench dataset JSON")
    parser.add_argument("--output-dir",
                        default="ripple_plots",
                        help="Output directory")
    parser.add_argument(
        "--min-base-accuracy",
        type=float,
        default=0.4,
        help="Minimum base accuracy to include topic (default: 0.4)")

    args = parser.parse_args()

    # Prepare list of unlearned models
    unlearned_models = []
    if args.elm:
        unlearned_models.append((args.elm, "ELM (baulab/elm-zephyr-7b-beta)"))
    if args.rmu:
        unlearned_models.append((args.rmu, "RMU (cais/Zephyr_RMU)"))

    if not unlearned_models:
        print("Error: Provide at least one unlearned model (--elm or --rmu)")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_ripple_effect(args.base_csv,
                       unlearned_models,
                       args.dataset,
                       output_dir,
                       min_base_accuracy=args.min_base_accuracy)


if __name__ == "__main__":
    main()
