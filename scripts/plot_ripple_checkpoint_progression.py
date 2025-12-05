#!/usr/bin/env python3
"""
Plot ripple effect progression across checkpoints for a single unlearning method.
Shows how the ripple effect evolves from checkpoint 1 to 8.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

# Configure matplotlib styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'text.usetex': False,
})

# Available LLM-GAT methods
LLM_GAT_METHODS = [
    'llama-3-8b-instruct-graddiff', 'llama-3-8b-instruct-elm',
    'llama-3-8b-instruct-pbj', 'llama-3-8b-instruct-tar',
    'llama-3-8b-instruct-rmu', 'llama-3-8b-instruct-rmu-lat',
    'llama-3-8b-instruct-repnoise', 'llama-3-8b-instruct-rr'
]


def load_results(results_dir, model_name):
    """Load results CSV for a specific model."""
    csv_path = Path(results_dir) / f"{model_name}_ripple_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def calculate_accuracy_by_distance(df, max_distance=100):
    """Calculate accuracy for each distance level."""
    accuracies = {}
    for dist in range(max_distance + 1):
        dist_data = df[df['distance'] == dist]
        if len(dist_data) > 0:
            accuracies[dist] = dist_data['is_correct'].mean()
    return accuracies


def plot_checkpoint_progression(method_name,
                                results_dir,
                                output_dir,
                                max_distance=100,
                                focus_distances=[0, 5, 10, 20, 30]):
    """Plot how ripple effects evolve across checkpoints for a single method.
    
    Args:
        method_name: Name of the method (e.g., 'llama-3-8b-instruct-rmu')
        results_dir: Directory containing result CSV files
        output_dir: Output directory for plots
        max_distance: Maximum distance to analyze
        focus_distances: Specific distances to highlight
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model results
    print("Loading base model...")
    base_df = load_results(results_dir, 'Llama-3-8b-Instruct')
    if base_df is None:
        # Try alternative base model
        base_df = load_results(results_dir, 'zephyr-7b-beta')
        if base_df is None:
            print("Error: No base model results found!")
            return

    base_accuracies = calculate_accuracy_by_distance(base_df, max_distance)

    # Load checkpoint results
    print(f"\nLoading checkpoints for {method_name}...")
    checkpoint_data = {}
    for ckpt in range(1, 9):
        model_name = f"{method_name}-ckpt{ckpt}"
        df = load_results(results_dir, model_name)
        if df is not None:
            checkpoint_data[ckpt] = calculate_accuracy_by_distance(
                df, max_distance)
            print(f"  Loaded checkpoint {ckpt}: {len(df)} questions")

    if not checkpoint_data:
        print(f"Error: No checkpoint data found for {method_name}")
        return

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Accuracy delta over distance for each checkpoint
    ax1 = plt.subplot(2, 2, 1)
    colors = plt.cm.viridis(np.linspace(0.2, 1, 8))

    for ckpt, accuracies in sorted(checkpoint_data.items()):
        distances = sorted(
            set(base_accuracies.keys()) & set(accuracies.keys()))
        deltas = [base_accuracies[d] - accuracies[d] for d in distances]

        ax1.plot(distances,
                 deltas,
                 label=f'Checkpoint {ckpt}',
                 color=colors[ckpt - 1],
                 alpha=0.7 + (ckpt / 8) * 0.3)  # Later checkpoints more opaque

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Distance from WMDP Topics')
    ax1.set_ylabel('Accuracy Delta (Base - Unlearned)')
    ax1.set_title(
        f'Ripple Effect Evolution: {method_name.replace("llama-3-8b-instruct-", "").upper()}'
    )
    ax1.legend(loc='upper right', ncol=2, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_distance)

    # Plot 2: Focused view (distance 0-30)
    ax2 = plt.subplot(2, 2, 2)

    for ckpt, accuracies in sorted(checkpoint_data.items()):
        distances = sorted(
            set(base_accuracies.keys()) & set(accuracies.keys()))
        distances = [d for d in distances if d <= 30]
        deltas = [base_accuracies[d] - accuracies[d] for d in distances]

        ax2.plot(distances,
                 deltas,
                 label=f'Ckpt {ckpt}',
                 color=colors[ckpt - 1],
                 alpha=0.7 + (ckpt / 8) * 0.3,
                 marker='o',
                 markersize=3)

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Distance from WMDP Topics')
    ax2.set_ylabel('Accuracy Delta (Base - Unlearned)')
    ax2.set_title('Focused View (Distance 0-30)')
    ax2.legend(loc='upper right', ncol=2, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)

    # Plot 3: Checkpoint progression at specific distances
    ax3 = plt.subplot(2, 2, 3)

    for dist in focus_distances:
        deltas_at_dist = []
        checkpoints = []

        for ckpt, accuracies in sorted(checkpoint_data.items()):
            if dist in base_accuracies and dist in accuracies:
                delta = base_accuracies[dist] - accuracies[dist]
                deltas_at_dist.append(delta)
                checkpoints.append(ckpt)

        if deltas_at_dist:
            ax3.plot(checkpoints,
                     deltas_at_dist,
                     label=f'Distance {dist}',
                     marker='o',
                     markersize=6)

    ax3.set_xlabel('Checkpoint')
    ax3.set_ylabel('Accuracy Delta (Base - Unlearned)')
    ax3.set_title('Unlearning Progression at Key Distances')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(1, 9))

    # Plot 4: Heatmap of accuracy delta
    ax4 = plt.subplot(2, 2, 4)

    # Create matrix for heatmap
    max_plot_dist = 50
    heatmap_data = np.zeros((8, max_plot_dist + 1))

    for ckpt, accuracies in sorted(checkpoint_data.items()):
        for dist in range(max_plot_dist + 1):
            if dist in base_accuracies and dist in accuracies:
                heatmap_data[ckpt - 1,
                             dist] = base_accuracies[dist] - accuracies[dist]
            else:
                heatmap_data[ckpt - 1, dist] = np.nan

    # Plot heatmap
    im = ax4.imshow(heatmap_data,
                    aspect='auto',
                    cmap='RdBu_r',
                    vmin=-0.5,
                    vmax=0.5,
                    interpolation='nearest')
    ax4.set_xlabel('Distance from WMDP Topics')
    ax4.set_ylabel('Checkpoint')
    ax4.set_title('Accuracy Delta Heatmap')
    ax4.set_yticks(range(8))
    ax4.set_yticklabels(range(1, 9))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Accuracy Delta', rotation=270, labelpad=15)

    # Add method name as main title
    fig.suptitle(
        f'Checkpoint Analysis: {method_name.replace("llama-3-8b-instruct-", "").upper()}',
        fontsize=16,
        fontweight='bold')

    plt.tight_layout()

    # Save plot
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"ripple_checkpoint_progression_{method_name}_{date_str}.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")

    # Also save as PNG
    output_path_png = output_dir / f"ripple_checkpoint_progression_{method_name}_{date_str}.png"
    plt.savefig(output_path_png, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path_png}")

    plt.show()

    # Print summary statistics
    print(f"\nSummary for {method_name}:")
    print("=" * 50)
    for dist in focus_distances:
        print(f"\nDistance {dist}:")
        for ckpt in sorted(checkpoint_data.keys()):
            if dist in checkpoint_data[ckpt] and dist in base_accuracies:
                delta = base_accuracies[dist] - checkpoint_data[ckpt][dist]
                print(f"  Checkpoint {ckpt}: Δ = {delta:.3f}")


def plot_all_methods_progression(results_dir, output_dir, distance=0):
    """Plot checkpoint progression for all methods at a specific distance."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model
    print("Loading base model...")
    base_df = load_results(results_dir, 'Llama-3-8b-Instruct')
    if base_df is None:
        base_df = load_results(results_dir, 'zephyr-7b-beta')
        if base_df is None:
            print("Error: No base model results found!")
            return

    base_accuracies = calculate_accuracy_by_distance(base_df, 100)

    if distance not in base_accuracies:
        print(f"Error: Distance {distance} not found in base model results")
        return

    base_acc_at_dist = base_accuracies[distance]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Colors for different methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(LLM_GAT_METHODS)))

    for i, method in enumerate(LLM_GAT_METHODS):
        deltas = []
        checkpoints = []

        for ckpt in range(1, 9):
            model_name = f"{method}-ckpt{ckpt}"
            df = load_results(results_dir, model_name)

            if df is not None:
                acc = calculate_accuracy_by_distance(df, 100)
                if distance in acc:
                    delta = base_acc_at_dist - acc[distance]
                    deltas.append(delta)
                    checkpoints.append(ckpt)

        if deltas:
            ax.plot(checkpoints,
                    deltas,
                    label=method.replace('llama-3-8b-instruct-', '').upper(),
                    color=colors[i],
                    marker='o',
                    markersize=6,
                    linewidth=2)

    ax.set_xlabel('Checkpoint', fontsize=14)
    ax.set_ylabel(f'Accuracy Delta at Distance {distance}', fontsize=14)
    ax.set_title(f'Unlearning Progression: All Methods at Distance {distance}',
                 fontsize=16)
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 9))

    plt.tight_layout()

    # Save plot
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"ripple_all_methods_progression_dist{distance}_{date_str}.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot checkpoint progression for unlearning methods")
    parser.add_argument(
        "--method",
        help="Specific method to analyze (e.g., llama-3-8b-instruct-rmu)")
    parser.add_argument(
        "--all-methods",
        action='store_true',
        help="Plot progression for all methods at a specific distance")
    parser.add_argument("--distance",
                        type=int,
                        default=0,
                        help="Distance to analyze for all-methods plot")
    parser.add_argument("--results-dir",
                        default="results/parallelized",
                        help="Directory containing result CSV files")
    parser.add_argument("--output-dir",
                        default="results/plots",
                        help="Output directory for plots")
    parser.add_argument("--max-distance",
                        type=int,
                        default=100,
                        help="Maximum distance to plot")

    args = parser.parse_args()

    if args.all_methods:
        plot_all_methods_progression(results_dir=args.results_dir,
                                     output_dir=args.output_dir,
                                     distance=args.distance)
    elif args.method:
        plot_checkpoint_progression(method_name=args.method,
                                    results_dir=args.results_dir,
                                    output_dir=args.output_dir,
                                    max_distance=args.max_distance)
    else:
        # Plot all methods individually
        for method in LLM_GAT_METHODS:
            print(f"\n{'='*60}")
            print(f"Processing {method}")
            print('=' * 60)
            try:
                plot_checkpoint_progression(method_name=method,
                                            results_dir=args.results_dir,
                                            output_dir=args.output_dir,
                                            max_distance=args.max_distance)
            except Exception as e:
                print(f"Error processing {method}: {e}")


if __name__ == "__main__":
    main()
