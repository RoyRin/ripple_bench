import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Data paths - Updated to use 9_12 datasets
wmdp_bio_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/data/wmdp/wmdp-bio.json")
ripple_bench_bio_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/data/ripple_bench_2025-09-12-bio")
bio_results_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/results/all_models__duplicated__BIO_9_12"
)

wmdp_chem_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/data/wmdp/wmdp-chem.json")

ripple_bench_chem_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/data/ripple_bench_2025-09-12-chem")
chem_results_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/results/all_models__duplicated__CHEM_9_12"
)

# Plot configuration
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'text.usetex': False,
    'pgf.rcfonts': False,
})

# Method colors - consistent across all plots
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


def load_ripple_bench_results(
    directory_path: str,
    verbose: bool = True
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, dict]]]:
    """
    Load all ripple bench results from a directory containing CSV and summary JSON files.
    
    Args:
        directory_path: Path to directory containing the ripple bench results
        
    Returns:
        Tuple of (csvs_dict, summary_jsons_dict) where:
        - csvs_dict: {model-name: {checkpoint#: DataFrame}} nested dict mapping model names and checkpoints to CSV data
        - summary_jsons_dict: {model-name: {checkpoint#: dict}} nested dict mapping model names and checkpoints to summary JSON data
    """
    csvs = {}
    summary_jsons = {}

    directory = Path(directory_path)

    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")

    # Count files first
    all_files = list(directory.iterdir())
    csv_files = [
        f for f in all_files if f.name.endswith('_ripple_results.csv')
    ]
    json_files = [
        f for f in all_files if f.name.endswith('_ripple_results.summary.json')
    ]

    if verbose:
        print(
            f"Found {len(csv_files)} CSV files and {len(json_files)} JSON files to load"
        )

    # Process all files in the directory
    csv_count = 0
    json_count = 0
    for file_path in all_files:
        if file_path.is_file():
            filename = file_path.name

            # Extract model name and checkpoint
            if filename.endswith('_ripple_results.csv'):
                csv_count += 1
                if verbose:
                    print(
                        f"  Loading CSV {csv_count}/{len(csv_files)}: {filename}"
                    )
                base_name = filename.replace('_ripple_results.csv', '')

                # Check if this is a base model (starts with capital L) or has checkpoint
                if base_name.startswith('Llama'):
                    # Base model without checkpoint
                    model_name = base_name
                    checkpoint = 'base'
                else:
                    # Extract checkpoint number from patterns like "model-name-ckpt1" or "model-name-method-ckpt1"
                    match = re.match(r'(.+?)-ckpt(\d+)$', base_name)
                    if match:
                        model_name = match.group(1)
                        checkpoint = f'ckpt{match.group(2)}'
                    else:
                        # No checkpoint pattern found, treat as base
                        model_name = base_name
                        checkpoint = 'base'

                # Initialize nested dict if needed
                if model_name not in csvs:
                    csvs[model_name] = {}
                if verbose:
                    print(f"    Reading {model_name}/{checkpoint}...")
                csvs[model_name][checkpoint] = pd.read_csv(file_path)
                if verbose:
                    print(
                        f"    Loaded {len(csvs[model_name][checkpoint])} rows")

            elif filename.endswith('_ripple_results.summary.json'):
                json_count += 1
                if verbose:
                    print(
                        f"  Loading JSON {json_count}/{len(json_files)}: {filename}"
                    )
                base_name = filename.replace('_ripple_results.summary.json',
                                             '')

                # Check if this is a base model (starts with capital L) or has checkpoint
                if base_name.startswith('Llama'):
                    # Base model without checkpoint
                    model_name = base_name
                    checkpoint = 'base'
                else:
                    # Extract checkpoint number from patterns like "model-name-ckpt1" or "model-name-method-ckpt1"
                    match = re.match(r'(.+?)-ckpt(\d+)$', base_name)
                    if match:
                        model_name = match.group(1)
                        checkpoint = f'ckpt{match.group(2)}'
                    else:
                        # No checkpoint pattern found, treat as base
                        model_name = base_name
                        checkpoint = 'base'

                # Initialize nested dict if needed
                if model_name not in summary_jsons:
                    summary_jsons[model_name] = {}
                with open(file_path, 'r') as f:
                    summary_jsons[model_name][checkpoint] = json.load(f)

    if verbose:
        print(
            f"\nSuccessfully loaded {len(csvs)} models with their checkpoints")

    return csvs, summary_jsons


def organize_models(
    csvs: Dict[str, Dict[str, pd.DataFrame]], summary_jsons: Dict[str,
                                                                  Dict[str,
                                                                       dict]]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Dict[str,
                                                             pd.DataFrame]]]]:
    """
    Organize the loaded data into base models and unlearning results.
    
    Returns:
        Tuple of (base_models, unlearning_results) where:
        - base_models: {model_name: DataFrame} for base models
        - unlearning_results: {base_model: {method: {checkpoint: DataFrame}}}
    """
    base_models = {}
    unlearning_results = {}

    for model_name, checkpoints in csvs.items():
        # Check if this is a base model
        if 'base' in checkpoints:
            base_models[model_name] = checkpoints['base']
        else:
            # This is an unlearning method - parse the model name
            # Expected format: "llama-3-8b-instruct-method" or "zephyr-7b-method"
            if 'llama-3-8b-instruct' in model_name:
                base_model = 'llama'
                method = model_name.replace('llama-3-8b-instruct-', '')
            elif 'zephyr' in model_name:
                base_model = 'zephyr'
                method = model_name.replace('zephyr-7b-', '')
            else:
                # Unknown model pattern
                continue

            if base_model not in unlearning_results:
                unlearning_results[base_model] = {}

            unlearning_results[base_model][method] = checkpoints

    return base_models, unlearning_results


def process_df(df: pd.DataFrame,
               bucket_size: int = 10,
               max_distance: Optional[int] = None) -> pd.DataFrame:
    """
    Process a dataframe by filtering distance and adding buckets.

    Args:
        df: Input dataframe
        bucket_size: Size of distance buckets (default: 10)
        max_distance: Optional maximum distance to include (default: None = no limit)
    """
    df = df.copy()
    if max_distance is not None:
        df = df[df["distance"] <= max_distance]
    df["distance_bucket"] = (df["distance"] // bucket_size) * bucket_size
    return df


def load_df(path, bucket_size=10):
    """Legacy function for loading from file path."""
    df = pd.read_csv(path)
    return process_df(df, bucket_size)


def get_dedup_results(df):
    df_dedup = df.groupby("question")[["is_correct", "distance_bucket"]].agg(
        ["max", "min", "sum", "count", "mean"])
    results = df_dedup.groupby(df_dedup["distance_bucket"]["min"]).agg(
        ["mean", "std"])["is_correct"]["mean"]
    results["sem"] = results["std"] / np.sqrt(
        df_dedup["distance_bucket"].groupby("min").size())
    return results


def get_legacy_results(df):
    raw_results = df.groupby("distance_bucket")["is_correct"].agg(
        ["mean", "std"])
    raw_results["sem"] = raw_results["std"] / np.sqrt(
        df.groupby("distance_bucket").size())
    return raw_results


def draw_from_data(base_models: Dict[str, pd.DataFrame],
                   unlearning_results: Dict[str, Dict[str,
                                                      Dict[str,
                                                           pd.DataFrame]]],
                   results_fn,
                   checkpoint_selector='best',
                   save_plots=False,
                   dataset='bio',
                   show_plots=False):
    """
    Draw plots comparing base and unlearned models.
    
    Args:
        base_models: Dictionary of base model DataFrames
        unlearning_results: Nested dict of unlearning results
        results_fn: Function to process results (get_dedup_results or get_legacy_results)
        checkpoint_selector: 'best', 'last', or specific checkpoint like 'ckpt6'
    """
    # Count total number of plots needed
    total_plots = sum(len(methods) for methods in unlearning_results.values())

    # Calculate grid dimensions
    cols = 3
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    plot_idx = 0

    for base_model_name, methods in unlearning_results.items():
        # Get base model DataFrame
        base_df = None
        for bm_name, bm_df in base_models.items():
            if ('Llama' in bm_name and base_model_name == 'llama') or \
               ('zephyr' in bm_name.lower() and base_model_name == 'zephyr'):
                base_df = process_df(bm_df)
                break

        if base_df is None:
            continue

        base_results = results_fn(base_df)

        for method, checkpoints in methods.items():
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]
            plot_idx += 1

            # Select checkpoint
            if checkpoint_selector == 'best':
                # Select checkpoint with best average accuracy
                best_ckpt = None
                best_acc = -1
                for ckpt_name, ckpt_df in checkpoints.items():
                    processed_df = process_df(ckpt_df)
                    acc = processed_df['is_correct'].mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_ckpt = ckpt_name
                selected_checkpoint = best_ckpt
            elif checkpoint_selector == 'last':
                # Get the highest numbered checkpoint
                ckpt_nums = [
                    int(k.replace('ckpt', '')) for k in checkpoints.keys()
                    if k.startswith('ckpt')
                ]
                if ckpt_nums:
                    selected_checkpoint = f'ckpt{max(ckpt_nums)}'
                else:
                    selected_checkpoint = list(checkpoints.keys())[0]
            else:
                selected_checkpoint = checkpoint_selector

            if selected_checkpoint not in checkpoints:
                selected_checkpoint = list(checkpoints.keys())[0]

            # Load unlearning results
            unlearn_df = process_df(checkpoints[selected_checkpoint])
            unlearn_results = results_fn(unlearn_df)

            # Plot base results
            ax.errorbar(
                base_results.index,
                base_results["mean"],
                yerr=base_results["sem"],
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=3,
                label="Base",
            )

            # Plot unlearning results
            ax.errorbar(unlearn_results.index,
                        unlearn_results["mean"],
                        yerr=unlearn_results["sem"],
                        marker="s",
                        linewidth=2,
                        markersize=6,
                        capsize=3,
                        label=f"Unlearn ({selected_checkpoint})",
                        color=METHOD_COLORS.get(method, '#888888'))

            ax.set_xlabel("Semantic Distance")
            ax.set_ylabel("Accuracy")
            ax.set_title(
                f"{base_model_name.title()} - {method.upper()} ({selected_checkpoint})"
            )
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            if base_model_name == "llama":
                ax.set_ylim(0.4, 0.75)

    # Hide unused axes
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save the plot if requested
    if save_plots:
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        PLOT_DIR = Path(
            "plots") / f"{dataset}_{checkpoint_selector}_{date_str}"
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        fn = f"grid_{dataset}_{checkpoint_selector}_{results_fn.__name__}"
        png_path = PLOT_DIR / f"{fn}.png"
        pdf_path = PLOT_DIR / f"{fn}.pdf"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved grid plot to: {png_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()


def draw(results_fn):
    # Initialize the plot with 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    i = 0
    for model in UNLEARNING_RESULTS:
        base_df = load_df(BASE_RESULTS[model])
        base_results = results_fn(base_df)

        for method in UNLEARNING_RESULTS[model]:
            ax = axes[i]
            i += 1

            # Load unlearning results for this method
            unlearn_df_method = load_df(UNLEARNING_RESULTS[model][method])
            unlearn_results_method = results_fn(unlearn_df_method)

            # Plot base results
            ax.errorbar(
                base_results.index,
                base_results["mean"],
                yerr=base_results["sem"],
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=3,
                label="Base",
            )

            # Plot unlearning results for this method
            ax.errorbar(
                unlearn_results_method.index,
                unlearn_results_method["mean"],
                yerr=unlearn_results_method["sem"],
                marker="s",
                linewidth=2,
                markersize=6,
                capsize=3,
                label=f"Unlearn",
            )

            ax.set_xlabel("Semantic Distance")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{model} ({method.title()})")
            if i == 1:
                ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            if model == "llama":
                ax.set_ylim(0.4, 0.95)

    plt.tight_layout()
    plt.show()


def draw_delta(results_fn):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    i = 0
    for model in UNLEARNING_RESULTS:
        base_df = load_df(BASE_RESULTS[model])
        base_results = results_fn(base_df)

        for method in UNLEARNING_RESULTS[model]:
            ax = axes[i]
            i += 1

            # Load unlearning results for this method
            unlearn_df_method = load_df(UNLEARNING_RESULTS[model][method])
            unlearn_results_method = results_fn(unlearn_df_method)

            # Calculate the difference between base and unlearn results
            difference = base_results["mean"] - unlearn_results_method["mean"]

            # Calculate error propagation for the difference
            # For independent measurements: σ(A-B) = sqrt(σ_A² + σ_B²)
            error_propagated = (base_results["sem"]**2 +
                                unlearn_results_method["sem"]**2)**0.5

            # Plot the difference
            ax.errorbar(
                base_results.index,
                difference,
                yerr=error_propagated,
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=3,
                label="Base - Unlearn",
            )

            ax.set_xlabel("Semantic Distance")
            ax.set_ylabel("Accuracy delta")
            ax.set_title(f"{model} ({method.title()})")
            if i == 1:
                ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # if model == "llama":
            #     ax.set_ylim(0.4, 0.75)

    plt.tight_layout()
    plt.show()


def draw_combined_from_data(base_models: Dict[str, pd.DataFrame],
                            unlearning_results: Dict[str,
                                                     Dict[str,
                                                          Dict[str,
                                                               pd.DataFrame]]],
                            results_fn=None,
                            checkpoint_selector='ckpt8',
                            save_plots=True,
                            dataset='bio',
                            show_plots=False):
    """
    Draw all models on a single combined plot.

    Args:
        base_models: Dictionary of base model DataFrames
        unlearning_results: Nested dict of unlearning results
        results_fn: Function to process results (default: get_legacy_results)
        checkpoint_selector: 'best', 'last', or specific checkpoint (default: 'ckpt8')
        save_plots: Whether to save plots to files
        dataset: Dataset name for the plot filename
    """
    if results_fn is None:
        results_fn = get_legacy_results

    # Create date-stamped plot directory
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    PLOT_DIR = Path("plots") / f"{dataset}_{checkpoint_selector}_{date_str}"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if save_plots:
        print(f"Plots will be saved to: {PLOT_DIR}")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Find and plot base Llama model
    base_llama_df = None
    for model_name, df in base_models.items():
        if 'Llama' in model_name:
            base_llama_df = process_df(df)
            break

    if base_llama_df is None:
        print("Warning: No Llama base model found")
        return

    base_results = results_fn(base_llama_df)

    ax.errorbar(
        base_results.index,
        base_results["mean"] * 100,  # Convert to percentage
        yerr=base_results["sem"] * 100,  # Convert to percentage
        marker='o',
        linewidth=3,
        markersize=7,
        capsize=3,
        linestyle='-',
        color='black',
        alpha=0.9,
        label="Llama3 Base",
        zorder=10  # Put base on top
    )

    # Process each unlearned model
    for base_model_name, methods in unlearning_results.items():
        # Set line style based on base model
        if base_model_name == "zephyr":
            linestyle = '--'  # Dashed for Zephyr
            marker = '^'  # Triangle for Zephyr
            prefix = "Zephyr-"
            alpha = 0.7
        else:  # llama
            linestyle = '-'  # Solid for Llama
            marker = 's'  # Square for Llama
            prefix = "Llama3-"
            alpha = 0.8

        for method, checkpoints in methods.items():
            # Select checkpoint
            if checkpoint_selector == 'last':
                ckpt_nums = [
                    int(k.replace('ckpt', '')) for k in checkpoints.keys()
                    if k.startswith('ckpt')
                ]
                if ckpt_nums:
                    selected_checkpoint = f'ckpt{max(ckpt_nums)}'
                else:
                    selected_checkpoint = list(checkpoints.keys())[0]
            elif checkpoint_selector == 'best':
                # Select checkpoint with best average accuracy
                best_ckpt = None
                best_acc = -1
                for ckpt_name, ckpt_df in checkpoints.items():
                    acc = ckpt_df['is_correct'].mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_ckpt = ckpt_name
                selected_checkpoint = best_ckpt
            else:
                selected_checkpoint = checkpoint_selector

            if selected_checkpoint not in checkpoints:
                selected_checkpoint = list(checkpoints.keys())[0]

            # Load unlearning results for this method
            unlearn_df = process_df(checkpoints[selected_checkpoint])
            unlearn_results_method = results_fn(unlearn_df)

            # Get color for this method
            color = METHOD_COLORS.get(method, '#888888')

            # Plot the unlearned model accuracy
            ax.errorbar(
                unlearn_results_method.index,
                unlearn_results_method["mean"] * 100,  # Convert to percentage
                yerr=unlearn_results_method["sem"] *
                100,  # Convert to percentage
                marker=marker,
                linewidth=2,
                markersize=5,
                capsize=2,
                linestyle=linestyle,
                color=color,
                alpha=alpha,
                label=f"{prefix}{method.upper().replace('_', '-')}",
            )

    ax.set_xlabel("Semantic Distance", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_title("Ripple Effects: Base vs Unlearned Models", fontsize=16)
    ax.legend(loc="lower right", ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 95)  # Update y-axis limits to percentage scale

    plt.tight_layout()

    # Save the plot
    if save_plots:
        fn = f"ripple_effects_{dataset}_full_{checkpoint_selector}_{results_fn.__name__}"
        png_path = PLOT_DIR / f"{fn}.png"
        pdf_path = PLOT_DIR / f"{fn}.pdf"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved full plot to: {png_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    # Also create a focused view (0-30)
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    # Plot base Llama model (focused)
    base_results_filtered = base_results[base_results.index <= 30]

    ax2.errorbar(
        base_results_filtered.index,
        base_results_filtered["mean"] * 100,  # Convert to percentage
        yerr=base_results_filtered["sem"] * 100,  # Convert to percentage
        marker='o',
        linewidth=3,
        markersize=8,
        capsize=3,
        linestyle='-',
        color='black',
        alpha=0.9,
        label="Llama3 Base",
        zorder=10  # Put base on top
    )

    # Process each unlearned model (focused view)
    for base_model_name, methods in unlearning_results.items():
        # Set line style based on base model
        if base_model_name == "zephyr":
            linestyle = '--'  # Dashed for Zephyr
            marker = '^'  # Triangle for Zephyr
            prefix = "Zephyr-"
            alpha = 0.7
        else:  # llama
            linestyle = '-'  # Solid for Llama
            marker = 's'  # Square for Llama
            prefix = "Llama3-"
            alpha = 0.8

        for method, checkpoints in methods.items():
            # Select checkpoint
            if checkpoint_selector == 'last':
                ckpt_nums = [
                    int(k.replace('ckpt', '')) for k in checkpoints.keys()
                    if k.startswith('ckpt')
                ]
                if ckpt_nums:
                    selected_checkpoint = f'ckpt{max(ckpt_nums)}'
                else:
                    selected_checkpoint = list(checkpoints.keys())[0]
            else:
                selected_checkpoint = checkpoint_selector

            if selected_checkpoint not in checkpoints:
                continue

            # Load unlearning results for this method
            unlearn_df = process_df(checkpoints[selected_checkpoint])
            unlearn_results_method = results_fn(unlearn_df)

            # Filter to only show 0-30
            unlearn_results_filtered = unlearn_results_method[
                unlearn_results_method.index <= 30]

            # Get color for this method
            color = METHOD_COLORS.get(method, '#888888')

            # Plot the unlearned model accuracy
            ax2.errorbar(
                unlearn_results_filtered.index,
                unlearn_results_filtered["mean"] *
                100,  # Convert to percentage
                yerr=unlearn_results_filtered["sem"] *
                100,  # Convert to percentage
                marker=marker,
                linewidth=2.5,
                markersize=6,
                capsize=3,
                linestyle=linestyle,
                color=color,
                alpha=alpha,
                label=f"{prefix}{method.upper().replace('_', '-')}",
            )

    ax2.set_xlabel("Semantic Distance", fontsize=14)
    ax2.set_ylabel("Accuracy (%)", fontsize=14)
    ax2.set_title("Ripple Effects: Focused View (Distance 0-30)", fontsize=16)
    ax2.legend(loc="lower right", ncol=2, fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 32)
    ax2.set_ylim(40, 95)  # Update y-axis limits to percentage scale

    plt.tight_layout()

    # Save the focused plot
    if save_plots:
        fn_focused = f"ripple_effects_{dataset}_focused_{checkpoint_selector}_{results_fn.__name__}"
        png_path = PLOT_DIR / f"{fn_focused}.png"
        pdf_path = PLOT_DIR / f"{fn_focused}.pdf"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved focused plot to: {png_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()


def draw_delta(results_fn):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    i = 0
    for model in UNLEARNING_RESULTS:
        base_df = load_df(BASE_RESULTS[model])
        base_results = results_fn(base_df)

        for method in UNLEARNING_RESULTS[model]:
            ax = axes[i]
            i += 1

            # Load unlearning results for this method
            unlearn_df_method = load_df(UNLEARNING_RESULTS[model][method])
            unlearn_results_method = results_fn(unlearn_df_method)

            # Calculate the difference between base and unlearn results
            difference = base_results["mean"] - unlearn_results_method["mean"]

            # Calculate error propagation for the difference
            # For independent measurements: σ(A-B) = sqrt(σ_A² + σ_B²)
            error_propagated = (base_results["sem"]**2 +
                                unlearn_results_method["sem"]**2)**0.5

            # Plot the difference
            ax.errorbar(
                base_results.index,
                difference,
                yerr=error_propagated,
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=3,
                label="Base - Unlearn",
            )

            ax.set_xlabel("Semantic Distance")
            ax.set_ylabel("Accuracy delta")
            ax.set_title(f"{model} ({method.title()})")
            if i == 1:
                ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # if model == "llama":
            #     ax.set_ylim(0.4, 0.75)

    plt.tight_layout()
    plt.show()


def draw_combined(results_fn=get_legacy_results):
    """Legacy wrapper for draw_combined_from_data."""

    # This is a legacy function that requires BASE_RESULTS and UNLEARNING_RESULTS
    # to be defined globally. Use draw_combined_from_data instead.
    pass


def plot_checkpoint_progression(
        base_models: Dict[str, pd.DataFrame],
        unlearning_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        method_name: str,
        base_model_name: str = 'llama',
        results_fn=None,
        save_plots=True,
        dataset='bio'):
    """
    Plot accuracy progression across checkpoints for a specific method with baseline.

    Args:
        base_models: Dictionary of base model DataFrames
        unlearning_results: Nested dict of unlearning results
        method_name: Method name to plot (e.g., 'elm', 'rmu', etc.)
        base_model_name: Base model name ('llama' or 'zephyr')
        results_fn: Function to process results
        save_plots: Whether to save plots
        dataset: Dataset name for filename
    """
    if results_fn is None:
        results_fn = get_legacy_results

    # Get base model data
    base_df = None
    for bm_name, bm_df in base_models.items():
        if ('Llama' in bm_name and base_model_name == 'llama') or \
           ('zephyr' in bm_name.lower() and base_model_name == 'zephyr'):
            base_df = process_df(bm_df)
            break

    if base_df is None:
        print(f"Base model {base_model_name} not found")
        return

    # Get method data
    if base_model_name not in unlearning_results or method_name not in unlearning_results[
            base_model_name]:
        print(f"Method {method_name} not found for {base_model_name}")
        return

    checkpoints = unlearning_results[base_model_name][method_name]

    # Sort checkpoints numerically
    sorted_checkpoints = sorted(
        [(k, v) for k, v in checkpoints.items() if k.startswith('ckpt')],
        key=lambda x: int(x[0].replace('ckpt', '')))

    if not sorted_checkpoints:
        print(f"No checkpoints found for {method_name}")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot baseline first (thick black line)
    base_results = results_fn(base_df)
    ax.errorbar(base_results.index,
                base_results["mean"] * 100,
                yerr=base_results["sem"] * 100,
                marker='o',
                linewidth=3,
                markersize=8,
                capsize=3,
                color='black',
                alpha=0.9,
                label=f'{base_model_name.title()} Baseline',
                zorder=10)

    # Plot progression for each checkpoint
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(sorted_checkpoints)))

    for idx, (ckpt_name, ckpt_df) in enumerate(sorted_checkpoints):
        processed_df = process_df(ckpt_df)
        ckpt_results = results_fn(processed_df)

        ax.errorbar(ckpt_results.index,
                    ckpt_results["mean"] * 100,
                    yerr=ckpt_results["sem"] * 100,
                    marker='s',
                    linewidth=2,
                    markersize=5,
                    capsize=2,
                    color=colors[idx],
                    alpha=0.7,
                    label=ckpt_name)

    ax.set_xlabel("Semantic Distance", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_title(
        f"Checkpoint Progression: {base_model_name.title()}-{method_name.upper()}",
        fontsize=16)
    ax.legend(loc="lower right", ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 95)

    plt.tight_layout()

    # Save the plot if requested
    if save_plots:
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        PLOT_DIR = Path(
            "plots") / f"checkpoint_progression_{dataset}_{date_str}"
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        fn = f"progression_{dataset}_{base_model_name}_{method_name}_{results_fn.__name__}"
        png_path = PLOT_DIR / f"{fn}.png"
        pdf_path = PLOT_DIR / f"{fn}.pdf"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"  Saved to: {png_path}")

    plt.show()


def plot_all_checkpoint_progressions(
        base_models: Dict[str, pd.DataFrame],
        unlearning_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        results_fn=None,
        save_plots=True,
        dataset='bio',
        show_plots=False):
    """
    Generate checkpoint progression plots for all methods.

    Args:
        base_models: Dictionary of base model DataFrames
        unlearning_results: Nested dict of unlearning results
        results_fn: Function to process results
        save_plots: Whether to save plots
        dataset: Dataset name for filename
    """
    if results_fn is None:
        results_fn = get_legacy_results

    print(f"\nGenerating checkpoint progression plots for all methods...")

    # Create a single timestamp for all plots
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    PLOT_DIR = Path("plots") / f"checkpoint_progression_{dataset}_{date_str}"

    if save_plots:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving all plots to: {PLOT_DIR}")

    # Iterate through all methods
    for base_model_name, methods in unlearning_results.items():
        print(f"\nProcessing {base_model_name} methods:")

        for method_name in sorted(methods.keys()):
            print(f"  Generating plots for {method_name}...")

            # Track all y-values for dynamic ylim
            all_y_values = []

            # Get base model data
            base_df = None
            for bm_name, bm_df in base_models.items():
                if ('Llama' in bm_name and base_model_name == 'llama') or \
                   ('zephyr' in bm_name.lower() and base_model_name == 'zephyr'):
                    base_df = process_df(bm_df)
                    break

            if base_df is None:
                continue

            # Calculate baseline results
            base_results = results_fn(base_df)
            base_mean = base_results["mean"] * 100
            base_sem = base_results["sem"] * 100
            all_y_values.extend(base_mean - base_sem)
            all_y_values.extend(base_mean + base_sem)

            # Get checkpoints for this method
            checkpoints = methods[method_name]
            sorted_checkpoints = sorted(
                [(k, v)
                 for k, v in checkpoints.items() if k.startswith('ckpt')],
                key=lambda x: int(x[0].replace('ckpt', '')))

            if not sorted_checkpoints:
                continue

            # Calculate checkpoint results for dynamic ylim
            checkpoint_results_list = []
            for ckpt_name, ckpt_df in sorted_checkpoints:
                processed_df = process_df(ckpt_df)
                ckpt_results = results_fn(processed_df)
                checkpoint_results_list.append((ckpt_name, ckpt_results))

                ckpt_mean = ckpt_results["mean"] * 100
                ckpt_sem = ckpt_results["sem"] * 100
                all_y_values.extend(ckpt_mean - ckpt_sem)
                all_y_values.extend(ckpt_mean + ckpt_sem)

            # Calculate dynamic ylim with padding
            y_min = min(all_y_values)
            y_max = max(all_y_values)
            y_range = y_max - y_min
            y_padding = y_range * 0.05  # 5% padding
            dynamic_ylim = (y_min - y_padding, y_max + y_padding)

            # Generate two versions of the plot
            for ylim_type in ['fixed', 'dynamic']:
                # Create plot for this method
                fig, ax = plt.subplots(figsize=(14, 8))

                # Plot baseline
                ax.errorbar(base_results.index,
                            base_mean,
                            yerr=base_sem,
                            marker='o',
                            linewidth=3,
                            markersize=8,
                            capsize=3,
                            color='black',
                            alpha=0.9,
                            label=f'{base_model_name.title()} Baseline',
                            zorder=10)

                # Plot checkpoints
                colors = plt.cm.coolwarm(
                    np.linspace(0, 1, len(sorted_checkpoints)))

                for idx, (ckpt_name,
                          ckpt_results) in enumerate(checkpoint_results_list):
                    ax.errorbar(ckpt_results.index,
                                ckpt_results["mean"] * 100,
                                yerr=ckpt_results["sem"] * 100,
                                marker='s',
                                linewidth=2,
                                markersize=5,
                                capsize=2,
                                color=colors[idx],
                                alpha=0.7,
                                label=ckpt_name)

                ax.set_xlabel("Semantic Distance", fontsize=14)
                ax.set_ylabel("Accuracy (%)", fontsize=14)

                # Set title and ylim based on type
                ax.set_title(
                    f"Checkpoint Progression: {base_model_name.title()}-{method_name.upper()}",
                    fontsize=16)
                if ylim_type == 'fixed':
                    ax.set_ylim(40, 95)
                else:  # dynamic
                    ax.set_ylim(dynamic_ylim)

                ax.legend(loc="lower right", ncol=2, fontsize=10)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()

                # Save the plot
                if save_plots:
                    fn = f"progression_{dataset}_{base_model_name}_{method_name}_{results_fn.__name__}_{ylim_type}"
                    png_path = PLOT_DIR / f"{fn}.png"
                    pdf_path = PLOT_DIR / f"{fn}.pdf"
                    plt.savefig(png_path, dpi=150, bbox_inches='tight')
                    plt.savefig(pdf_path, bbox_inches='tight')
                    print(f"    Saved: {fn}.png and {fn}.pdf")

                if show_plots:
                    plt.show()
                else:
                    plt.close()

    print(f"\nCompleted generating all checkpoint progression plots")


def main(results_dir: str = None,
         dataset: str = 'bio',
         plot_type: str = 'combined',
         checkpoint_selector: str = 'ckpt8',
         results_fn_name: str = 'legacy',
         show_plots: bool = False):
    """
    Main function to load data and generate plots.

    Args:
        results_dir: Directory containing results (if None, uses default paths)
        dataset: 'bio' or 'chem'
        plot_type: 'combined', 'grid', 'delta', or 'progression'
        checkpoint_selector: 'best', 'last', or specific checkpoint
        results_fn_name: 'legacy' or 'dedup'
        show_plots: Whether to display plots interactively (default: False, only save)
    """
    # Select results directory
    if results_dir is None:
        if dataset == 'bio':
            results_dir = str(bio_results_path)
        elif dataset == 'chem':
            results_dir = str(chem_results_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    # Load data
    print(f"Loading data from {results_dir}...")
    csvs, summary_jsons = load_ripple_bench_results(results_dir)

    print(f"Loaded data for {len(csvs)} models")

    # Organize into base models and unlearning results
    base_models, unlearning_results = organize_models(csvs, summary_jsons)

    print(f"Found {len(base_models)} base models:")
    for name in base_models.keys():
        print(f"  - {name}")

    print(f"\nFound unlearning results for:")
    for base_model, methods in unlearning_results.items():
        print(f"  {base_model}: {list(methods.keys())}")

    # Select results function
    if results_fn_name == 'dedup':
        results_fn = get_dedup_results
    else:
        results_fn = get_legacy_results

    # Generate plots
    if plot_type == 'combined':
        print(
            f"\nGenerating combined plot with {checkpoint_selector} checkpoints..."
        )
        draw_combined_from_data(base_models,
                                unlearning_results,
                                results_fn=results_fn,
                                checkpoint_selector=checkpoint_selector,
                                save_plots=True,
                                dataset=dataset,
                                show_plots=show_plots)

    elif plot_type == 'grid':
        print(
            f"\nGenerating grid plot with {checkpoint_selector} checkpoints..."
        )
        draw_from_data(base_models,
                       unlearning_results,
                       results_fn=results_fn,
                       checkpoint_selector=checkpoint_selector,
                       save_plots=True,
                       dataset=dataset,
                       show_plots=show_plots)

    elif plot_type == 'delta':
        print("\nGenerating delta plots...")
        # TODO: Implement draw_delta_from_data
        print("Delta plots not yet implemented for new data format")

    elif plot_type == 'progression':
        print("\nGenerating checkpoint progression plots for all methods...")
        plot_all_checkpoint_progressions(base_models,
                                         unlearning_results,
                                         results_fn=results_fn,
                                         save_plots=True,
                                         dataset=dataset,
                                         show_plots=show_plots)

    return csvs, summary_jsons, base_models, unlearning_results


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate ripple bench plots')
    parser.add_argument('--dataset',
                        type=str,
                        default='bio',
                        choices=['bio', 'chem'],
                        help='Dataset to use')
    parser.add_argument('--plot',
                        type=str,
                        default='combined',
                        choices=['combined', 'grid', 'delta', 'progression'],
                        help='Type of plot to generate')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='ckpt8',
        help='Checkpoint selection: best, last, or specific (e.g., ckpt8)')
    parser.add_argument('--results-fn',
                        type=str,
                        default='legacy',
                        choices=['legacy', 'dedup'],
                        help='Results processing function')
    parser.add_argument('--dir',
                        type=str,
                        default=None,
                        help='Custom results directory')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively (default: only save)')

    args = parser.parse_args()

    csvs, summary_jsons, base_models, unlearning_results = main(
        results_dir=args.dir,
        dataset=args.dataset,
        plot_type=args.plot,
        checkpoint_selector=args.checkpoint,
        results_fn_name=args.results_fn,
        show_plots=args.show)
