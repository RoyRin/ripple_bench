#!/usr/bin/env python3
"""
Plot ripple effects for all models (checkpoint-8 or final versions).
Shows accuracy delta (base - unlearned) vs distance for all methods.
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
from scipy import stats

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

# Method colors - consistent across base models
METHOD_COLORS = {
    'elm': '#FF6B6B',  # Red
    'rmu': '#4ECDC4',  # Teal
    'graddiff': '#95E77E',  # Light green
    'pbj': '#FFD93D',  # Yellow
    'tar': '#A8E6CF',  # Mint
    'rmu-lat': '#FF8B94',  # Pink
    'repnoise': '#B4A7D6',  # Lavender
    'rr': '#FFB347'  # Orange
}

# Model groups
MODEL_GROUPS = {
    'Base': {
        'models': ['Llama-3-8b-Instruct', 'zephyr-7b-beta'],
        'color': '#000000'
    },
    'Zephyr Unlearned': {
        'models': ['zephyr-7b-elm', 'zephyr-7b-rmu'],
        'marker': 'o'  # Circle marker for Zephyr-based models
    },
    'LLM-GAT Methods': {
        'models': [
            'llama-3-8b-instruct-graddiff-ckpt8',
            'llama-3-8b-instruct-elm-ckpt8', 'llama-3-8b-instruct-pbj-ckpt8',
            'llama-3-8b-instruct-tar-ckpt8', 'llama-3-8b-instruct-rmu-ckpt8',
            'llama-3-8b-instruct-rmu-lat-ckpt8',
            'llama-3-8b-instruct-repnoise-ckpt8',
            'llama-3-8b-instruct-rr-ckpt8'
        ],
        'marker':
        's'  # Square marker for Llama-based models
    }
}


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


def plot_all_models_comparison(results_dir,
                               output_dir,
                               max_distance=100,
                               smoothing_window=5):
    """Plot comparison of all models showing accuracy delta vs distance.
    
    Args:
        results_dir: Directory containing result CSV files
        output_dir: Output directory for plots
        max_distance: Maximum distance to plot
        smoothing_window: Window size for rolling average smoothing
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base models first
    print("Loading base models...")
    base_results = {}
    for base_model in MODEL_GROUPS['Base']['models']:
        df = load_results(results_dir, base_model)
        if df is not None:
            base_results[base_model] = calculate_accuracy_by_distance(
                df, max_distance)
            print(f"  Loaded {base_model}: {len(df)} questions")

    if not base_results:
        print("Error: No base model results found!")
        return

    # Use Llama as primary base, zephyr as secondary
    primary_base = 'Llama-3-8b-Instruct' if 'Llama-3-8b-Instruct' in base_results else 'zephyr-7b-beta'
    base_accuracies = base_results[primary_base]

    # Get datetime string with time for filenames
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ========== PLOT 1: Full distance range ==========
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    print("\nPlotting full distance range (0-100)...")

    # Plot Zephyr unlearned models
    for model in MODEL_GROUPS['Zephyr Unlearned']['models']:
        df = load_results(results_dir, model)
        if df is not None:
            model_acc = calculate_accuracy_by_distance(df, max_distance)

            # Calculate accuracy delta
            distances = sorted(
                set(base_accuracies.keys()) & set(model_acc.keys()))
            deltas = [base_accuracies[d] - model_acc[d] for d in distances]

            # Get method name and color
            method = model.split('-')[-1]  # 'elm' or 'rmu'
            color = METHOD_COLORS.get(method, '#888888')

            # Apply smoothing
            if len(distances) > smoothing_window:
                deltas_smooth = pd.Series(deltas).rolling(
                    window=smoothing_window, center=True).mean()
                ax1.plot(distances,
                         deltas_smooth,
                         label=f'Zephyr-{method.upper()}',
                         color=color,
                         alpha=0.8,
                         linewidth=2.5,
                         linestyle='--')  # Dashed for Zephyr
            else:
                ax1.plot(distances,
                         deltas,
                         label=f'Zephyr-{method.upper()}',
                         color=color,
                         alpha=0.8,
                         linewidth=2.5,
                         linestyle='--')  # Dashed for Zephyr
            print(f"  Plotted {model}")

    # Plot LLM-GAT models (checkpoint 8)
    for model in MODEL_GROUPS['LLM-GAT Methods']['models']:
        df = load_results(results_dir, model)
        if df is not None:
            model_acc = calculate_accuracy_by_distance(df, max_distance)

            # Calculate accuracy delta
            distances = sorted(
                set(base_accuracies.keys()) & set(model_acc.keys()))
            deltas = [base_accuracies[d] - model_acc[d] for d in distances]

            # Get method name and color
            method_full = model.replace('llama-3-8b-instruct-',
                                        '').replace('-ckpt8', '')
            color = METHOD_COLORS.get(method_full, '#888888')

            # Apply smoothing
            if len(distances) > smoothing_window:
                deltas_smooth = pd.Series(deltas).rolling(
                    window=smoothing_window, center=True).mean()
                ax1.plot(distances,
                         deltas_smooth,
                         label=f'Llama3-{method_full.upper()}',
                         color=color,
                         alpha=0.8,
                         linewidth=2.5,
                         linestyle='-')  # Solid for Llama
            else:
                ax1.plot(distances,
                         deltas,
                         label=f'Llama3-{method_full.upper()}',
                         color=color,
                         alpha=0.8,
                         linewidth=2.5,
                         linestyle='-')  # Solid for Llama
            print(f"  Plotted {model}")

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Distance from WMDP Topics', fontsize=14)
    ax1.set_ylabel('Accuracy Delta (Base - Unlearned)', fontsize=14)
    ax1.set_title('Ripple Effects: All Unlearning Methods (Full Range)',
                  fontsize=16)
    ax1.legend(loc='upper right', ncol=2, fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_distance)

    plt.tight_layout()

    # Save full range plot
    output_path = output_dir / f"ripple_all_models_full_{datetime_str}.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved full range plot to {output_path}")

    output_path_png = output_dir / f"ripple_all_models_full_{datetime_str}.png"
    plt.savefig(output_path_png, dpi=150, bbox_inches='tight')
    print(f"Saved PNG to {output_path_png}")

    plt.show()

    # ========== PLOT 2: Focused view (0-30) ==========
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    print("\nPlotting focused view (0-30)...")

    # Plot Zephyr unlearned models
    for model in MODEL_GROUPS['Zephyr Unlearned']['models']:
        df = load_results(results_dir, model)
        if df is not None:
            # Use the same accuracy calculation as the full plot
            model_acc = calculate_accuracy_by_distance(df, max_distance)
            distances = sorted(
                set(base_accuracies.keys()) & set(model_acc.keys()))
            # Filter to only show distances 0-30
            distances = [d for d in distances if d <= 30]
            deltas = [base_accuracies[d] - model_acc[d] for d in distances]

            # Get method name and color
            method = model.split('-')[-1]
            color = METHOD_COLORS.get(method, '#888888')

            ax2.plot(
                distances,
                deltas,
                label=f'Zephyr-{method.upper()}',
                color=color,
                alpha=0.8,
                marker='o',  # Circle for Zephyr
                markersize=5,
                linewidth=2.5,
                linestyle='--')  # Dashed for Zephyr

    # Plot LLM-GAT models
    for model in MODEL_GROUPS['LLM-GAT Methods']['models']:
        df = load_results(results_dir, model)
        if df is not None:
            # Use the same accuracy calculation as the full plot
            model_acc = calculate_accuracy_by_distance(df, max_distance)
            distances = sorted(
                set(base_accuracies.keys()) & set(model_acc.keys()))
            # Filter to only show distances 0-30
            distances = [d for d in distances if d <= 30]
            deltas = [base_accuracies[d] - model_acc[d] for d in distances]

            # Get method name and color
            method_full = model.replace('llama-3-8b-instruct-',
                                        '').replace('-ckpt8', '')
            color = METHOD_COLORS.get(method_full, '#888888')

            ax2.plot(
                distances,
                deltas,
                label=f'Llama3-{method_full.upper()}',
                color=color,
                alpha=0.8,
                marker='s',  # Square for Llama
                markersize=4,
                linewidth=2.5,
                linestyle='-')  # Solid for Llama

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Distance from WMDP Topics', fontsize=14)
    ax2.set_ylabel('Accuracy Delta (Base - Unlearned)', fontsize=14)
    ax2.set_title('Ripple Effects: Focused View (Distance 0-30)', fontsize=16)
    ax2.legend(loc='upper right', ncol=2, fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)

    plt.tight_layout()

    # Save focused view plot
    output_path = output_dir / f"ripple_all_models_focused_{datetime_str}.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved focused view plot to {output_path}")

    output_path_png = output_dir / f"ripple_all_models_focused_{datetime_str}.png"
    plt.savefig(output_path_png, dpi=150, bbox_inches='tight')
    print(f"Saved PNG to {output_path_png}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot ripple effects for all models")
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
    parser.add_argument("--smoothing",
                        type=int,
                        default=5,
                        help="Window size for smoothing")

    args = parser.parse_args()

    plot_all_models_comparison(results_dir=args.results_dir,
                               output_dir=args.output_dir,
                               max_distance=args.max_distance,
                               smoothing_window=args.smoothing)


if __name__ == "__main__":
    main()
