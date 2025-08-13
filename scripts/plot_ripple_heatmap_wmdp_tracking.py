#!/usr/bin/env python3
"""
Plot ripple effect heatmap: WMDP topics as rows, distances as columns, tracking neighbors properly
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from ripple_bench.utils import read_dict
from collections import defaultdict

# Configure matplotlib styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'text.usetex': False,
})


def plot_wmdp_tracking_heatmap(base_csv,
                               unlearned_csvs,
                               dataset_path,
                               output_dir,
                               min_base_accuracy=0.4,
                               max_distance=100):
    """Plot heatmap with WMDP topics as rows and distances as columns, properly tracking neighbors.
    
    Args:
        base_csv: Path to base model CSV results
        unlearned_csvs: List of (csv_path, label) tuples for unlearned models
        dataset_path: Path to dataset JSON
        output_dir: Output directory for plots
        min_base_accuracy: Minimum base accuracy to include topic (default: 0.4)
        max_distance: Maximum distance to include (default: 100)
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get date string for filenames
    date_str = datetime.now().strftime("%Y%m%d")

    # Load dataset to get topic-to-neighbors mapping
    print("Loading dataset to get topic relationships...")
    dataset = read_dict(dataset_path)

    # Get the topic to neighbors mapping
    if 'raw_data' in dataset and 'topic_to_neighbors' in dataset['raw_data']:
        topic_to_neighbors = dataset['raw_data']['topic_to_neighbors']
    else:
        print("Error: Cannot find topic_to_neighbors mapping in dataset")
        return

    print(
        f"Found {len(topic_to_neighbors)} WMDP topics with neighbor mappings")

    # Create reverse mapping: neighbor -> (original_topic, distance)
    neighbor_to_original = {}
    for wmdp_topic, neighbors in topic_to_neighbors.items():
        for dist_minus_1, neighbor in enumerate(neighbors):
            neighbor_to_original[neighbor] = (wmdp_topic, dist_minus_1 + 1)

    # Load base model results
    print("Loading base model results...")
    base_df = pd.read_csv(base_csv)

    # Process each unlearned model
    for unlearned_csv, label in unlearned_csvs:
        print(f"\nProcessing {label}...")
        unlearned_df = pd.read_csv(unlearned_csv)

        # Merge base and unlearned results
        merged = pd.merge(
            base_df[['question_id', 'topic', 'distance', 'is_correct']],
            unlearned_df[['question_id', 'is_correct']],
            on='question_id',
            suffixes=('_base', '_unlearned'))

        # Calculate accuracy delta
        merged['accuracy_delta'] = merged['is_correct_base'].astype(
            int) - merged['is_correct_unlearned'].astype(int)

        # Build matrix: WMDP topics x distances
        # First, identify WMDP topics (those at distance 0)
        wmdp_topics = sorted(merged[merged['distance'] == 0]['topic'].unique())

        # Filter by base accuracy
        base_acc_by_topic = merged[merged['distance'] == 0].groupby(
            'topic')['is_correct_base'].mean()
        high_acc_wmdp = [
            t for t in wmdp_topics
            if base_acc_by_topic.get(t, 0) >= min_base_accuracy
        ]

        print(
            f"Filtered to {len(high_acc_wmdp)} WMDP topics with base accuracy >= {min_base_accuracy*100:.0f}%"
        )

        # Create matrix
        matrix_data = []
        topic_labels = []

        for wmdp_topic in high_acc_wmdp:
            # Row for this WMDP topic across all distances
            row_data = np.full(max_distance + 1, np.nan)

            # Distance 0: the WMDP topic itself
            d0_data = merged[(merged['topic'] == wmdp_topic)
                             & (merged['distance'] == 0)]
            if len(d0_data) > 0:
                row_data[0] = d0_data['accuracy_delta'].mean()

            # Distances 1+: the neighbor topics
            if wmdp_topic in topic_to_neighbors:
                neighbors = topic_to_neighbors[wmdp_topic]
                for dist_minus_1, neighbor_topic in enumerate(
                        neighbors[:max_distance]):
                    # Find data for this neighbor
                    neighbor_data = merged[
                        (merged['topic'] == neighbor_topic)
                        & (merged['distance'] == dist_minus_1 + 1)]
                    if len(neighbor_data) > 0:
                        row_data[dist_minus_1 +
                                 1] = neighbor_data['accuracy_delta'].mean()

            matrix_data.append(row_data)
            topic_labels.append(wmdp_topic[:40])  # Truncate long names

        # Convert to numpy array
        matrix = np.array(matrix_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(20, max(10, len(topic_labels) * 0.15)))

        # Create custom colormap
        cmap = sns.diverging_palette(250, 10, as_cmap=True)

        # Plot heatmap
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(0, max_distance + 1, 10))
        ax.set_xticklabels(range(0, max_distance + 1, 10))
        ax.set_yticks(np.arange(len(topic_labels)))
        ax.set_yticklabels(topic_labels)

        # Labels
        ax.set_xlabel('Distance from Original WMDP Topic', fontsize=12)
        ax.set_ylabel('WMDP Topics', fontsize=12)
        ax.set_title(
            f'Ripple Effect Tracking - {label}\nEach row shows one WMDP topic tested on itself (d=0) and its semantic neighbors (d=1-100)',
            fontsize=14,
            fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy Delta (Base - Unlearned)', fontsize=11)

        # Add vertical line at distance 0
        ax.axvline(x=0.5,
                   color='white',
                   linewidth=1,
                   linestyle='--',
                   alpha=0.7)

        plt.tight_layout()

        # Save with date string
        model_suffix = label.lower().replace(' ', '_')
        output_png = output_dir / f'ripple_heatmap_wmdp_tracking_{model_suffix}_{date_str}.png'
        output_pdf = output_dir / f'ripple_heatmap_wmdp_tracking_{model_suffix}_{date_str}.pdf'

        plt.savefig(output_png, dpi=150, bbox_inches='tight')
        plt.savefig(output_pdf, bbox_inches='tight')
        print(f"Saved tracking heatmap for {label}:")
        print(f"  PNG: {output_png}")
        print(f"  PDF: {output_pdf}")

        plt.close()

        # Also create a focused version (distances 0-20)
        fig, ax = plt.subplots(figsize=(12, max(10, len(topic_labels) * 0.15)))

        # Plot focused version
        im = ax.imshow(matrix[:, :21],
                       cmap=cmap,
                       aspect='auto',
                       vmin=-1,
                       vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(21))
        ax.set_xticklabels(range(21))
        ax.set_yticks(np.arange(len(topic_labels)))
        ax.set_yticklabels(topic_labels)

        # Labels
        ax.set_xlabel('Distance from Original WMDP Topic', fontsize=12)
        ax.set_ylabel('WMDP Topics', fontsize=12)
        ax.set_title(
            f'Ripple Effect Tracking (Focused) - {label}\nDistances 0-20 only',
            fontsize=14,
            fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy Delta (Base - Unlearned)', fontsize=11)

        # Add vertical line at distance 0
        ax.axvline(x=0.5,
                   color='white',
                   linewidth=1,
                   linestyle='--',
                   alpha=0.7)

        plt.tight_layout()

        # Save focused version
        output_png_focused = output_dir / f'ripple_heatmap_wmdp_tracking_{model_suffix}_focused_{date_str}.png'
        output_pdf_focused = output_dir / f'ripple_heatmap_wmdp_tracking_{model_suffix}_focused_{date_str}.pdf'

        plt.savefig(output_png_focused, dpi=150, bbox_inches='tight')
        plt.savefig(output_pdf_focused, bbox_inches='tight')
        print(f"Saved focused tracking heatmap for {label}:")
        print(f"  PNG: {output_png_focused}")
        print(f"  PDF: {output_pdf_focused}")

        plt.close()

        # Print summary statistics
        print(f"\nSummary for {label}:")
        print(f"  WMDP topics included: {len(topic_labels)}")
        print(f"  Mean delta at distance 0: {np.nanmean(matrix[:, 0]):.3f}")
        if matrix.shape[1] > 10:
            print(
                f"  Mean delta at distance 10: {np.nanmean(matrix[:, 10]):.3f}"
            )
        if matrix.shape[1] > 50:
            print(
                f"  Mean delta at distance 50: {np.nanmean(matrix[:, 50]):.3f}"
            )

        # Count coverage
        non_nan_count = np.sum(~np.isnan(matrix))
        total_cells = matrix.size
        print(
            f"  Coverage: {non_nan_count}/{total_cells} cells ({100*non_nan_count/total_cells:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description=
        "Create heatmap tracking WMDP topics through their semantic neighbors")
    parser.add_argument("base_csv", help="Path to base model CSV results")
    parser.add_argument("--elm", help="Path to ELM model CSV results")
    parser.add_argument("--rmu", help="Path to RMU model CSV results")
    parser.add_argument("--dataset",
                        required=True,
                        help="Path to ripple bench dataset JSON")
    parser.add_argument("--output-dir",
                        default="results/ripple_plots",
                        help="Output directory for plots")
    parser.add_argument(
        "--min-base-accuracy",
        type=float,
        default=0.4,
        help="Minimum base accuracy to include topic (default: 0.4)")
    parser.add_argument("--max-distance",
                        type=int,
                        default=100,
                        help="Maximum distance to include (default: 100)")

    args = parser.parse_args()

    # Collect unlearned models
    unlearned_csvs = []
    if args.elm:
        unlearned_csvs.append((args.elm, "ELM"))
    if args.rmu:
        unlearned_csvs.append((args.rmu, "RMU"))

    if not unlearned_csvs:
        print(
            "Error: Must specify at least one unlearned model (--elm or --rmu)"
        )
        return

    # Create plots
    plot_wmdp_tracking_heatmap(args.base_csv, unlearned_csvs, args.dataset,
                               args.output_dir, args.min_base_accuracy,
                               args.max_distance)


if __name__ == "__main__":
    main()
