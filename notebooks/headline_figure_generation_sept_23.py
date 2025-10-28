import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Set
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

bio_results_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/results/all_models__duplicated__BIO_9_12_filtered_20250924_145547"
)
bio_results_path = Path(
    "/Users/roy/data/ripple_bench/9_25_2025/results/all_models__duplicated__BIO_9_24"
)

wmdp_chem_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/data/wmdp/wmdp-chem.json")

ripple_bench_chem_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/data/ripple_bench_2025-09-12-chem")
chem_results_path = Path(
    "/Users/roy/data/ripple_bench/9_05_2025/results/all_models__duplicated__CHEM_9_17"
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


def load_neighbor_quality_filter(filter_path: Optional[Path] = None,
                                 bio_chem_threshold: int = 13,
                                 biosafety_threshold: int = 7,
                                 any_safety_threshold: int = 15) -> Set[str]:
    """
    Load neighbor quality results and return set of topics that pass filtration.

    A topic passes if it has:
    - More than bio_chem_threshold bio/chem-related neighbors (default: >13/20), OR
    - More than biosafety_threshold biosafety-related neighbors (default: >7/20), OR
    - More than any_safety_threshold general safety-related neighbors (default: >15/20)

    Args:
        filter_path: Path to bio_neighbor_quality_results.json file
        bio_chem_threshold: Minimum bio/chem neighbors required (topics with MORE than this pass)
        biosafety_threshold: Minimum biosafety neighbors required (topics with MORE than this pass)
        any_safety_threshold: Minimum general safety neighbors required (topics with MORE than this pass)

    Returns:
        Set of topic names that pass the filter
    """
    if filter_path is None:
        return set()  # No filtering if no path provided

    if not filter_path.exists():
        print(
            f"Warning: Filter file not found at {filter_path}, proceeding without filtering"
        )
        return set()

    print(f"\nLoading neighbor quality filter from {filter_path}")

    with open(filter_path, 'r') as f:
        quality_data = json.load(f)

    passing_topics = set()

    for topic, data in quality_data.items():
        # Calculate the number of relevant neighbors (opposite of non-relevant counts)
        bio_chem_count = data['total'] - data['non_bio_chem_count']
        biosafety_count = data['total'] - data['non_biosafety_count']
        any_safety_count = data['total'] - data['non_any_safety_count']

        # Check if topic passes any threshold
        if bio_chem_count > bio_chem_threshold or \
           biosafety_count > biosafety_threshold or \
           any_safety_count > any_safety_threshold:
            passing_topics.add(topic)

    print(
        f"Filter results: {len(passing_topics)}/{len(quality_data)} topics pass the thresholds"
    )
    print(
        f"  Thresholds: bio/chem>{bio_chem_threshold}, biosafety>{biosafety_threshold}, safety>{any_safety_threshold}"
    )

    return passing_topics


def apply_topic_filter(df: pd.DataFrame,
                       valid_topics: Set[str]) -> pd.DataFrame:
    """
    Filter a dataframe to only include rows with topics in the valid set.
    This keeps ALL distances for valid topics, not just distance 0.

    Args:
        df: DataFrame with a 'topic' or 'question_topic' column
        valid_topics: Set of valid topic names

    Returns:
        Filtered DataFrame with all distances for valid topics
    """
    if not valid_topics:
        return df  # No filtering if no valid topics

    # Debug: Check what columns and unique values we have
    if 'topic' in df.columns:
        unique_topics = df['topic'].unique()
        print(f"    DataFrame has {len(unique_topics)} unique topics")
        # Find which topics in the dataframe match our filter
        matching_topics = set(unique_topics) & valid_topics
        print(f"    {len(matching_topics)} topics match the filter")

        if len(matching_topics) > 0:
            # Show some examples
            print(f"    Example matching topics: {list(matching_topics)[:3]}")

        original_len = len(df)
        filtered_df = df[df['topic'].isin(valid_topics)]
        filtered_len = len(filtered_df)

        # Check distances in filtered data
        if 'distance' in filtered_df.columns:
            unique_distances = sorted(filtered_df['distance'].unique())
            print(
                f"    Filtered data contains distances: {unique_distances[:10]}..."
            )

    elif 'question_topic' in df.columns:
        unique_topics = df['question_topic'].unique()
        print(
            f"    DataFrame has {len(unique_topics)} unique topics (question_topic)"
        )
        matching_topics = set(unique_topics) & valid_topics
        print(f"    {len(matching_topics)} topics match the filter")

        original_len = len(df)
        filtered_df = df[df['question_topic'].isin(valid_topics)]
        filtered_len = len(filtered_df)

    else:
        print(
            "Warning: No topic column found in dataframe, skipping filtering")
        return df

    if original_len != filtered_len:
        topic_col = 'topic' if 'topic' in df.columns else 'question_topic'
        num_topics_before = len(df[topic_col].unique())
        num_topics_after = len(filtered_df[topic_col].unique())
        print(
            f"    Filtered: {original_len} → {filtered_len} rows ({filtered_len/original_len*100:.1f}% retained)"
        )
        print(
            f"    Topics: {num_topics_before} → {num_topics_after} ({num_topics_after/num_topics_before*100:.1f}% retained)"
        )

    return filtered_df


def load_ripple_bench_results(
    directory_path: str,
    verbose: bool = True,
    checkpoint_filter: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, dict]]]:
    """
    Load all ripple bench results from a directory containing CSV and summary JSON files.

    Args:
        directory_path: Path to directory containing the ripple bench results
        verbose: Whether to print loading progress
        checkpoint_filter: If specified, only load this checkpoint (e.g., 'ckpt8')

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
        f for f in all_files
        if (f.name.endswith('_ripple_results.csv')
            or f.name.endswith('_ripple_results_filtered.csv'))
    ]
    json_files = [
        f for f in all_files if f.name.endswith('_ripple_results.summary.json')
    ]

    if verbose:
        if checkpoint_filter:
            # Count files that match the filter
            filtered_csv = [
                f for f in csv_files
                if checkpoint_filter in f.name or 'Llama' in f.name
            ]
            filtered_json = [
                f for f in json_files
                if checkpoint_filter in f.name or 'Llama' in f.name
            ]
            print(
                f"Found {len(filtered_csv)} CSV files and {len(filtered_json)} JSON files matching filter '{checkpoint_filter}'"
            )
        else:
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
            if filename.endswith('_ripple_results.csv') or filename.endswith(
                    '_ripple_results_filtered.csv'):
                base_name = filename.replace('_ripple_results_filtered.csv',
                                             '').replace(
                                                 '_ripple_results.csv', '')

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

                # Skip if we have a checkpoint filter and this doesn't match
                if checkpoint_filter and checkpoint != checkpoint_filter and checkpoint != 'base':
                    continue

                csv_count += 1
                if verbose:
                    print(
                        f"  Loading CSV {csv_count}/{len([f for f in csv_files if not checkpoint_filter or checkpoint_filter in f.name or 'Llama' in f.name])}: {filename}"
                    )

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

                # Skip if we have a checkpoint filter and this doesn't match
                if checkpoint_filter and checkpoint != checkpoint_filter and checkpoint != 'base':
                    continue

                json_count += 1
                if verbose:
                    print(
                        f"  Loading JSON {json_count}/{len([f for f in json_files if not checkpoint_filter or checkpoint_filter in f.name or 'Llama' in f.name])}: {filename}"
                    )

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
    # Use assigned_question_id for deduplication if available, otherwise use question
    if "assigned_question_id" in df.columns:
        group_col = "assigned_question_id"
    elif "question" in df.columns:
        group_col = "question"
    else:
        # If neither column exists, fall back to question_id
        print(
            "Warning: No 'assigned_question_id' or 'question' column found for deduplication"
        )
        group_col = "question_id"

    df_dedup = df.groupby(group_col)[["is_correct", "distance_bucket"]].agg(
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


def apply_rolling_average(results, window_size=10):
    """
    Apply rolling average to results DataFrame.

    Args:
        results: DataFrame with 'mean', 'std', 'sem' columns
        window_size: Size of the rolling window

    Returns:
        DataFrame with smoothed values
    """
    if window_size <= 1:
        return results

    # Create a copy to avoid modifying original
    smoothed = results.copy()

    # Apply rolling average to mean values
    smoothed['mean'] = results['mean'].rolling(
        window=window_size,
        center=True,  # Center the window
        min_periods=1  # Use available data at edges
    ).mean()

    # Also smooth the std and sem for consistency
    if 'std' in results.columns:
        smoothed['std'] = results['std'].rolling(window=window_size,
                                                 center=True,
                                                 min_periods=1).mean()

    if 'sem' in results.columns:
        smoothed['sem'] = results['sem'].rolling(window=window_size,
                                                 center=True,
                                                 min_periods=1).mean()

    return smoothed


def draw_from_data(base_models: Dict[str, pd.DataFrame],
                   unlearning_results: Dict[str, Dict[str,
                                                      Dict[str,
                                                           pd.DataFrame]]],
                   results_fn,
                   checkpoint_selector='best',
                   save_plots=False,
                   dataset='bio',
                   show_plots=False,
                   show_wmdp_results=False):
    """
    Draw plots comparing base and unlearned models.

    Args:
        base_models: Dictionary of base model DataFrames
        unlearning_results: Nested dict of unlearning results
        results_fn: Function to process results (get_dedup_results or get_legacy_results)
        checkpoint_selector: 'best', 'last', or specific checkpoint like 'ckpt6'
        show_wmdp_results: Whether to show WMDP results as stars at distance 0
    """
    # WMDP results data (as percentages)
    wmdp_data = {
        "Llama3 8B Instruct": 0.70 * 100,
        "Grad Diff": 0.25 * 100,
        "RMU": 0.26 * 100,
        "RMU + LAT": 0.32 * 100,
        "RepNoise": 0.29 * 100,
        "ELM": 0.24 * 100,
        "RR": 0.26 * 100,
        "TAR": 0.28 * 100,
        "PB&J": 0.31 * 100
    }

    # Map method names to WMDP data keys
    wmdp_method_map = {
        "graddiff": "Grad Diff",
        "rmu": "RMU",
        "rmu-lat": "RMU + LAT",
        "repnoise": "RepNoise",
        "elm": "ELM",
        "rr": "RR",
        "tar": "TAR",
        "pbj": "PB&J"
    }
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

            # Add WMDP results as stars at distance 0 if requested
            if show_wmdp_results and base_model_name == "llama":
                # Plot base model WMDP point
                if "Llama3 8B Instruct" in wmdp_data:
                    ax.scatter(0,
                               wmdp_data["Llama3 8B Instruct"],
                               marker='*',
                               s=300,
                               color='black',
                               edgecolors='white',
                               linewidths=1,
                               zorder=15,
                               label="WMDP: Base")

                # Plot unlearning method WMDP point
                wmdp_key = wmdp_method_map.get(method)
                if wmdp_key and wmdp_key in wmdp_data:
                    ax.scatter(0,
                               wmdp_data[wmdp_key],
                               marker='*',
                               s=250,
                               color=METHOD_COLORS.get(method, '#888888'),
                               edgecolors='white',
                               linewidths=1,
                               zorder=15,
                               label=f"WMDP: {method.upper()}")

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

        # Create PNG and PDF subdirectories
        png_dir = PLOT_DIR / "PNG"
        pdf_dir = PLOT_DIR / "PDF"
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        fn = f"grid_{dataset}_{checkpoint_selector}_{results_fn.__name__}"
        png_path = png_dir / f"{fn}.png"
        pdf_path = pdf_dir / f"{fn}.pdf"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved grid plot to: PNG/{fn}.png and PDF/{fn}.pdf")

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
                            show_plots=False,
                            show_wmdp_results=False):
    """
    Draw all models on a single combined plot.

    Args:
        base_models: Dictionary of base model DataFrames
        unlearning_results: Nested dict of unlearning results
        results_fn: Function to process results (default: get_legacy_results)
        checkpoint_selector: 'best', 'last', or specific checkpoint (default: 'ckpt8')
        save_plots: Whether to save plots to files
        dataset: Dataset name for the plot filename
        show_wmdp_results: Whether to show WMDP results as stars at distance 0
    """
    if results_fn is None:
        results_fn = get_legacy_results

    # WMDP results data (as percentages)
    wmdp_data = {
        "Llama3 8B Instruct": 0.70 * 100,
        "Grad Diff": 0.25 * 100,
        "RMU": 0.26 * 100,
        "RMU + LAT": 0.32 * 100,
        "RepNoise": 0.29 * 100,
        "ELM": 0.24 * 100,
        "RR": 0.26 * 100,
        "TAR": 0.28 * 100,
        "PB&J": 0.31 * 100
    }

    # Map method names to WMDP data keys
    wmdp_method_map = {
        "graddiff":
        "Grad Diff",  # Note: filename uses "graddiff" not "grad_diff"
        "rmu": "RMU",
        "rmu-lat": "RMU + LAT",  # Note: might be "rmu-lat" in filename
        "repnoise": "RepNoise",
        "elm": "ELM",
        "rr": "RR",
        "tar": "TAR",
        "pbj": "PB&J"
    }

    # Create date-stamped plot directory
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    PLOT_DIR = Path("plots") / f"{dataset}_{checkpoint_selector}_{date_str}"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if save_plots:
        print(f"Plots will be saved to: {PLOT_DIR}")
        # Create PNG and PDF subdirectories
        png_dir = PLOT_DIR / "PNG"
        pdf_dir = PLOT_DIR / "PDF"
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)

    # Track all y-values for dynamic ylim
    all_y_values = []

    # Find base Llama model first
    base_llama_df = None
    for model_name, df in base_models.items():
        if 'Llama' in model_name:
            base_llama_df = process_df(df)
            break

    if base_llama_df is None:
        print("Warning: No Llama base model found")
        return

    base_results = results_fn(base_llama_df)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

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

    # Plot WMDP results as stars at distance 0 if requested
    if show_wmdp_results and "Llama3 8B Instruct" in wmdp_data:
        ax.scatter(0,
                   wmdp_data["Llama3 8B Instruct"],
                   marker='*',
                   s=300,
                   color='black',
                   edgecolors='white',
                   linewidths=1,
                   zorder=15,
                   label="WMDP: Llama3 Base")

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

            # Add WMDP result as star at distance 0 if available
            if show_wmdp_results and base_model_name == "llama":  # Only show for Llama models
                wmdp_key = wmdp_method_map.get(method)
                if wmdp_key and wmdp_key in wmdp_data:
                    ax.scatter(
                        0,
                        wmdp_data[wmdp_key],
                        marker='*',
                        s=250,
                        color=color,
                        edgecolors='white',
                        linewidths=1,
                        zorder=15,
                        label=
                        f"WMDP: {prefix}{method.upper().replace('_', '-')}")

    ax.set_xlabel("Semantic Distance", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_title("Ripple Effects: Base vs Unlearned Models", fontsize=20)
    ax.legend(loc="lower right", ncol=2, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Generate both fixed and dynamic y-axis versions
    for ylim_type in ['fixed', 'dynamic']:
        if ylim_type == 'fixed':
            ax.set_ylim(40, 95)  # Fixed percentage scale
        else:  # dynamic
            # Calculate dynamic ylim from all plotted data
            y_vals = []
            for line in ax.lines:
                y_vals.extend(line.get_ydata())

            # Also include WMDP star points if shown
            if show_wmdp_results:
                # Add the base model WMDP point
                if "Llama3 8B Instruct" in wmdp_data:
                    y_vals.append(wmdp_data["Llama3 8B Instruct"])

                # Add all unlearning method WMDP points
                for method in wmdp_method_map:
                    wmdp_key = wmdp_method_map[method]
                    if wmdp_key in wmdp_data:
                        y_vals.append(wmdp_data[wmdp_key])

            if y_vals:
                y_min, y_max = min(y_vals), max(y_vals)
                y_range = y_max - y_min
                y_padding = y_range * 0.05  # 5% padding
                ax.set_ylim(y_min - y_padding, y_max + y_padding)

        plt.tight_layout()

        # Save the plot
        if save_plots:
            fn = f"ripple_effects_{dataset}_full_{checkpoint_selector}_{results_fn.__name__}_{ylim_type}_scale"
            png_path = png_dir / f"{fn}.png"
            pdf_path = pdf_dir / f"{fn}.pdf"
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.savefig(pdf_path, bbox_inches='tight')
            print(
                f"Saved full plot ({ylim_type} scale) to: PNG/{fn}.png and PDF/{fn}.pdf"
            )

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
        fontsize=20)
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

        # Create PNG and PDF subdirectories
        png_dir = PLOT_DIR / "PNG"
        pdf_dir = PLOT_DIR / "PDF"
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        fn = f"progression_{dataset}_{base_model_name}_{method_name}_{results_fn.__name__}"
        png_path = png_dir / f"{fn}.png"
        pdf_path = pdf_dir / f"{fn}.pdf"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"  Saved to: PNG/{fn}.png and PDF/{fn}.pdf")

    plt.show()


def plot_delta_from_baseline(
        base_models: Dict[str, pd.DataFrame],
        unlearning_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        checkpoint: str = 'ckpt8',
        results_fn=None,
        save_plots: bool = True,
        dataset: str = 'bio',
        show_plots: bool = False,
        apply_average: bool = False,
        window_size: int = 10):
    """
    Plot the delta (difference) between baseline and checkpoint for each method.
    Positive values indicate performance improvement over baseline.

    Args:
        base_models: Dictionary of base model DataFrames
        unlearning_results: Nested dict of unlearning results
        checkpoint: Which checkpoint to use (default: 'ckpt8')
        results_fn: Function to process results
        save_plots: Whether to save plots
        dataset: Dataset name for filename
        show_plots: Whether to display plots
        apply_average: Whether to apply rolling average
        window_size: Window size for rolling average
    """
    if results_fn is None:
        results_fn = get_legacy_results

    print(f"\nGenerating delta plots (baseline - {checkpoint})...")
    print(f"Base models available: {list(base_models.keys())}")
    print(f"Unlearning results available: {list(unlearning_results.keys())}")

    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Find the base model (Llama or Zephyr)
    base_df = None
    base_model_display = None
    for bm_name, bm_df in base_models.items():
        if 'Llama' in bm_name or 'zephyr' in bm_name.lower():
            base_df = process_df(bm_df)
            base_model_display = bm_name
            break

    if base_df is None:
        print("No base model found")
        return fig

    # Get base results (this is our baseline)
    base_results = results_fn(base_df)
    print(f"Base model {base_model_display} shape: {base_results.shape}")

    # Find unlearning results
    if len(unlearning_results) == 0:
        print("No unlearning results found")
        return fig

    # Get the first (and likely only) set of unlearning results
    um_name, methods = next(iter(unlearning_results.items()))
    print(f"Found unlearning methods under key '{um_name}'")

    # Plot delta for each method
    for method_name, checkpoints in methods.items():
        print(f"\n  Processing method: {method_name}")
        print(f"    Available checkpoints: {list(checkpoints.keys())}")

        # Get the specified checkpoint
        ckpt_key = None
        for key in checkpoints.keys():
            if checkpoint in key:
                ckpt_key = key
                break

        if not ckpt_key:
            print(f"    No {checkpoint} found for {method_name}")
            continue

        # Process checkpoint data
        ckpt_df = process_df(checkpoints[ckpt_key])
        ckpt_results = results_fn(ckpt_df)
        print(f"    Checkpoint {ckpt_key} shape: {ckpt_results.shape}")

        # Ensure both results are aligned by index
        # Use common indices between base model (baseline) and checkpoint
        common_idx = base_results.index.intersection(ckpt_results.index)
        baseline_aligned = base_results.loc[common_idx]
        ckpt_aligned = ckpt_results.loc[common_idx]

        print(f"    Common distances: {len(common_idx)}")
        if len(common_idx) == 0:
            print(
                f"    WARNING: No common indices between base model and {checkpoint}"
            )
            continue

        print(
            f"    Sample baseline values: {baseline_aligned['mean'].iloc[:5].tolist()}"
        )
        print(
            f"    Sample checkpoint values: {ckpt_aligned['mean'].iloc[:5].tolist()}"
        )

        # Calculate delta (baseline - checkpoint)
        # Positive means checkpoint has lower accuracy than baseline (more unlearning)
        delta = (baseline_aligned['mean'] - ckpt_aligned['mean']) * 100
        print(f"    Delta range: {delta.min():.2f} to {delta.max():.2f}")
        print(f"    Sample delta values: {delta.iloc[:5].tolist()}")

        # Apply rolling average if requested
        if apply_average:
            delta = delta.rolling(window=window_size,
                                  center=True,
                                  min_periods=1).mean()

        # Plot delta
        color = METHOD_COLORS.get(method_name, '#888888')
        print(f"    Plotting with color {color}")
        line = ax.plot(common_idx,
                       delta,
                       label=f'{method_name.upper()}',
                       color=color,
                       linewidth=2.5,
                       marker='o',
                       markersize=3,
                       markevery=5,
                       alpha=0.8)
        print(f"    Line plotted: {line}")

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # Formatting
    ax.set_xlabel('Semantic Distance', fontsize=14)
    ax.set_ylabel('Δ Accuracy (%)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)

    # Debug axes limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    print(f"\n  Axes limits: X={xlim}, Y={ylim}")

    plt.tight_layout()

    # Save plots
    if save_plots:
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        PLOT_DIR = Path("plots") / f"delta_{dataset}_{date_str}"
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        png_dir = PLOT_DIR / "PNG"
        pdf_dir = PLOT_DIR / "PDF"
        png_dir.mkdir(exist_ok=True)
        pdf_dir.mkdir(exist_ok=True)

        fn = f"delta_{checkpoint}_{dataset}"
        if apply_average:
            fn += f"_smooth{window_size}"

        png_path = png_dir / f"{fn}.png"
        pdf_path = pdf_dir / f"{fn}.pdf"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved delta plot to: PNG/{fn}.png and PDF/{fn}.pdf")

    if show_plots:
        plt.show()
    else:
        plt.close()

    return fig


def plot_method_comparison(base_models: Dict[str, pd.DataFrame],
                           unlearning_results: Dict[str,
                                                    Dict[str,
                                                         Dict[str,
                                                              pd.DataFrame]]],
                           methods_to_compare: list = ['rmu', 'elm'],
                           base_model_name: str = 'llama',
                           results_fn=None,
                           save_plots=True,
                           dataset='bio',
                           show_plots=False):
    """
    Plot side-by-side comparison of two methods' checkpoint progression.

    Args:
        base_models: Dictionary of base model DataFrames
        unlearning_results: Nested dict of unlearning results
        methods_to_compare: List of two method names to compare (default: ['rmu', 'elm'])
        base_model_name: Base model name ('llama' or 'zephyr')
        results_fn: Function to process results
        save_plots: Whether to save plots
        dataset: Dataset name for filename
        show_plots: Whether to display plots
    """
    if results_fn is None:
        results_fn = get_legacy_results

    if len(methods_to_compare) != 2:
        print("Error: Exactly two methods must be specified for comparison")
        return

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

    base_results = results_fn(base_df)

    # Check if both methods exist
    for method_name in methods_to_compare:
        if base_model_name not in unlearning_results or method_name not in unlearning_results[
                base_model_name]:
            print(f"Method {method_name} not found for {base_model_name}")
            return

    # Create figure with two subplots side by side with shared y-axis
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    # Track all y-values across both plots for shared scaling
    all_y_values_global = []
    all_y_values_global.extend(
        (base_results["mean"] - base_results["sem"]) * 100)
    all_y_values_global.extend(
        (base_results["mean"] + base_results["sem"]) * 100)

    # Store handles and labels for combined legend
    all_handles = []
    all_labels = []

    # Process each method
    for idx, method_name in enumerate(methods_to_compare):
        ax = axes[idx]

        # Get method checkpoints
        checkpoints = unlearning_results[base_model_name][method_name]

        # Sort checkpoints by number
        sorted_checkpoints = []
        for ckpt_name in checkpoints.keys():
            match = re.match(r'ckpt(\d+)$', ckpt_name)
            if match:
                sorted_checkpoints.append(
                    (int(match.group(1)), ckpt_name, checkpoints[ckpt_name]))
        sorted_checkpoints.sort()

        if not sorted_checkpoints:
            print(f"No valid checkpoints found for {method_name}")
            continue

        # Define colors for checkpoints (blue to red gradient)
        cmap = plt.cm.get_cmap('coolwarm')
        colors = [
            cmap(i / (len(sorted_checkpoints) - 1))
            for i in range(len(sorted_checkpoints))
        ]

        # Plot base model results (only label on first subplot)
        base_line = ax.errorbar(base_results.index,
                                base_results["mean"] * 100,
                                yerr=base_results["sem"] * 100,
                                marker='o',
                                linewidth=2.5,
                                markersize=7,
                                capsize=3,
                                linestyle='-',
                                color='black',
                                alpha=0.9,
                                label="Base Model" if idx == 0 else None,
                                zorder=10)

        if idx == 0:
            all_handles.append(base_line)
            all_labels.append("Base Model")

        # Plot each checkpoint
        for ckpt_idx, (ckpt_num, ckpt_name,
                       ckpt_df) in enumerate(sorted_checkpoints):
            processed_df = process_df(ckpt_df)
            ckpt_results = results_fn(processed_df)

            # Collect y-values for dynamic scaling
            ckpt_mean = ckpt_results["mean"] * 100
            ckpt_sem = ckpt_results["sem"] * 100
            all_y_values_global.extend(ckpt_mean - ckpt_sem)
            all_y_values_global.extend(ckpt_mean + ckpt_sem)

            # Only add to legend from first subplot
            line = ax.errorbar(ckpt_results.index,
                               ckpt_mean,
                               yerr=ckpt_sem,
                               marker='s',
                               linewidth=2,
                               markersize=5,
                               capsize=2,
                               linestyle='--',
                               color=colors[ckpt_idx],
                               alpha=0.7,
                               label=ckpt_name if idx == 0 else None)

            if idx == 0:
                all_handles.append(line)
                all_labels.append(ckpt_name)

        ax.set_xlabel("Semantic Distance", fontsize=14)
        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=14)
        ax.set_title(f"{method_name.upper()}", fontsize=20)
        ax.grid(True, alpha=0.3)

        # Set tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Set shared y-axis limits
    if all_y_values_global:
        y_min = min(all_y_values_global)
        y_max = max(all_y_values_global)
        y_range = y_max - y_min
        y_padding = y_range * 0.05  # 5% padding
        axes[0].set_ylim(y_min - y_padding, y_max + y_padding)

    # Add legend to the right plot (ELM)
    axes[1].legend(all_handles,
                   all_labels,
                   loc='lower right',
                   ncol=2,
                   fontsize=14)

    plt.tight_layout()

    # Save the plot if requested
    if save_plots:
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        PLOT_DIR = Path("plots") / f"method_comparison_{dataset}_{date_str}"
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        # Create PNG and PDF subdirectories
        png_dir = PLOT_DIR / "PNG"
        pdf_dir = PLOT_DIR / "PDF"
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        fn = f"comparison_{base_model_name}_{methods_to_compare[0]}_vs_{methods_to_compare[1]}_{dataset}_{results_fn.__name__}"
        png_path = png_dir / f"{fn}.png"
        pdf_path = pdf_dir / f"{fn}.pdf"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved comparison plot to: PNG/{fn}.png and PDF/{fn}.pdf")

    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_distance_over_checkpoints(
        base_models: Dict[str, pd.DataFrame],
        unlearning_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        method_name: str,
        base_model_name: str = 'llama',
        distances: list = [1, 50, 500],
        results_fn=None,
        save_plots=True,
        dataset='bio',
        show_plots=False):
    """
    Plot accuracy at specific distances over checkpoints for a specific method.

    Args:
        base_models: Dictionary of base model DataFrames
        unlearning_results: Nested dict of unlearning results
        method_name: Method name to plot (e.g., 'elm', 'rmu', etc.)
        base_model_name: Base model name ('llama' or 'zephyr')
        distances: List of distances to plot (default: [0, 50, 500])
        results_fn: Function to process results
        save_plots: Whether to save plots
        dataset: Dataset name for filename
        show_plots: Whether to display plots
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

    # Check if method exists
    if base_model_name not in unlearning_results or method_name not in unlearning_results[
            base_model_name]:
        print(f"Method {method_name} not found for {base_model_name}")
        return

    # Get method checkpoints
    checkpoints = unlearning_results[base_model_name][method_name]

    # Sort checkpoints by number
    sorted_checkpoints = []
    for ckpt_name in checkpoints.keys():
        match = re.match(r'ckpt(\d+)$', ckpt_name)
        if match:
            sorted_checkpoints.append((int(match.group(1)), ckpt_name))
    sorted_checkpoints.sort()

    if not sorted_checkpoints:
        print(f"No valid checkpoints found for {method_name}")
        return

    # Create figure with subplots for each distance
    fig, axes = plt.subplots(1,
                             len(distances),
                             figsize=(6 * len(distances), 5))
    if len(distances) == 1:
        axes = [axes]

    # Get base model results
    base_results = results_fn(base_df)

    # Process each distance
    for idx, distance in enumerate(distances):
        ax = axes[idx]

        # Get accuracy at this distance for base model
        distance_bucket = distance // 10 * 10  # Round to nearest bucket
        base_acc_at_dist = base_results.loc[
            distance_bucket,
            'mean'] * 100 if distance_bucket in base_results.index else None

        # Collect accuracies at this distance across checkpoints
        checkpoint_nums = []
        accuracies = []
        sems = []

        for ckpt_num, ckpt_name in sorted_checkpoints:
            ckpt_df = process_df(checkpoints[ckpt_name])
            ckpt_results = results_fn(ckpt_df)

            if distance_bucket in ckpt_results.index:
                checkpoint_nums.append(ckpt_num)
                accuracies.append(ckpt_results.loc[distance_bucket, 'mean'] *
                                  100)
                sems.append(ckpt_results.loc[distance_bucket, 'sem'] * 100)

        # Plot checkpoint progression
        if checkpoint_nums:
            ax.errorbar(checkpoint_nums,
                        accuracies,
                        yerr=sems,
                        marker='o',
                        linewidth=2,
                        markersize=6,
                        capsize=3,
                        color=METHOD_COLORS.get(method_name, '#888888'),
                        label=f'{method_name.upper()}')

        # Add baseline as horizontal line
        if base_acc_at_dist is not None:
            ax.axhline(y=base_acc_at_dist,
                       color='black',
                       linestyle='--',
                       alpha=0.5,
                       label='Base Model')

        ax.set_xlabel('Checkpoint', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_title(f'Distance = {distance}', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # Set x-axis to show only integer checkpoint numbers
        if checkpoint_nums:
            ax.set_xticks(range(min(checkpoint_nums),
                                max(checkpoint_nums) + 1))

    # Add overall title
    fig.suptitle(
        f'{method_name.upper()} - Accuracy at Specific Distances Over Checkpoints',
        fontsize=20,
        y=1.02)

    plt.tight_layout()

    # Save plots
    if save_plots:
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        PLOT_DIR = Path("plots") / f"distance_progression_{dataset}_{date_str}"
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        # Create PNG and PDF subdirectories
        png_dir = PLOT_DIR / "PNG"
        pdf_dir = PLOT_DIR / "PDF"
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        fn = f"{method_name}_distance_progression_{dataset}"
        png_path = png_dir / f"{fn}.png"
        pdf_path = pdf_dir / f"{fn}.pdf"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(
            f"Saved {method_name} distance progression to: PNG/{fn}.png and PDF/{fn}.pdf"
        )

    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_all_distance_progressions(
        base_models: Dict[str, pd.DataFrame],
        unlearning_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        distances: list = [1, 50, 500],
        results_fn=None,
        save_plots=True,
        dataset='bio',
        show_plots=False):
    """
    Generate distance progression plots with all methods on the same plot.

    Args:
        base_models: Dictionary of base model DataFrames
        unlearning_results: Nested dict of unlearning results
        distances: List of distances to plot (default: [1, 50, 500])
        results_fn: Function to process results
        save_plots: Whether to save plots
        dataset: Dataset name for filename
        show_plots: Whether to display plots
    """
    if results_fn is None:
        results_fn = get_legacy_results

    print(f"\nGenerating combined distance progression plot...")
    print(f"Distances to plot: {distances}")

    # Create a single timestamp for all plots
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    PLOT_DIR = Path("plots") / f"distance_progression_{dataset}_{date_str}"

    if save_plots:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        # Create PNG and PDF subdirectories
        png_dir = PLOT_DIR / "PNG"
        pdf_dir = PLOT_DIR / "PDF"
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving plots to: {PLOT_DIR}")
        print(f"  PNG files in: PNG/")
        print(f"  PDF files in: PDF/")

    # Process each base model group
    for base_model_name, methods in unlearning_results.items():
        print(
            f"\nGenerating combined plot for {base_model_name.upper()} models..."
        )

        # Get base model data
        base_df = None
        for bm_name, bm_df in base_models.items():
            if ('Llama' in bm_name and base_model_name == 'llama') or \
               ('zephyr' in bm_name.lower() and base_model_name == 'zephyr'):
                base_df = process_df(bm_df)
                break

        if base_df is None:
            print(f"  Base model {base_model_name} not found")
            continue

        # Get base model results
        base_results = results_fn(base_df)

        # Create figure with subplots for each distance, sharing y-axis
        fig, axes = plt.subplots(1,
                                 len(distances),
                                 figsize=(7 * len(distances), 6),
                                 sharey=True)
        if len(distances) == 1:
            axes = [axes]

        # Track all y-values for shared axis limits
        all_y_values = []

        # Process each distance
        for idx, distance in enumerate(distances):
            ax = axes[idx]

            # Get accuracy at this distance for base model
            distance_bucket = distance // 10 * 10  # Round to nearest bucket
            base_acc_at_dist = base_results.loc[
                distance_bucket,
                'mean'] * 100 if distance_bucket in base_results.index else None

            # Add baseline as horizontal line
            if base_acc_at_dist is not None:
                # Always add label - matplotlib will handle duplicates
                ax.axhline(y=base_acc_at_dist,
                           color='black',
                           linestyle='--',
                           alpha=0.5,
                           linewidth=2,
                           label='Base Model',
                           zorder=1)
                all_y_values.append(base_acc_at_dist)

            # Plot each method
            for method_name, checkpoints in methods.items():
                # Sort checkpoints by number
                sorted_checkpoints = []
                for ckpt_name in checkpoints.keys():
                    match = re.match(r'ckpt(\d+)$', ckpt_name)
                    if match:
                        sorted_checkpoints.append(
                            (int(match.group(1)), ckpt_name))
                sorted_checkpoints.sort()

                if not sorted_checkpoints:
                    print(f"  No valid checkpoints found for {method_name}")
                    continue

                # Collect accuracies at this distance across checkpoints
                checkpoint_nums = []
                accuracies = []
                sems = []

                for ckpt_num, ckpt_name in sorted_checkpoints:
                    ckpt_df = process_df(checkpoints[ckpt_name])
                    ckpt_results = results_fn(ckpt_df)

                    if distance_bucket in ckpt_results.index:
                        checkpoint_nums.append(ckpt_num)
                        accuracies.append(
                            ckpt_results.loc[distance_bucket, 'mean'] * 100)
                        sems.append(ckpt_results.loc[distance_bucket, 'sem'] *
                                    100)

                # Plot checkpoint progression for this method
                if checkpoint_nums:
                    # Always add label - matplotlib will handle duplicates
                    ax.errorbar(
                        checkpoint_nums,
                        accuracies,
                        yerr=sems,
                        marker='o',
                        linewidth=2,
                        markersize=6,
                        capsize=3,
                        color=METHOD_COLORS.get(method_name, '#888888'),
                        label=f'{method_name.upper().replace("_", "-")}',
                        alpha=0.8,
                        zorder=2)
                    all_y_values.extend(accuracies)

            ax.set_xlabel('Checkpoint', fontsize=16)
            # Only set y-label on first subplot since y-axis is shared
            if idx == 0:
                ax.set_ylabel('Accuracy (%)', fontsize=16)
            ax.set_title(f'Distance = {distance}', fontsize=20)
            ax.grid(True, alpha=0.3)

            # Only add legend to middle subplot
            if idx == len(distances) // 2:
                ax.legend(loc='best', fontsize=14, ncol=1)

            # Set tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=14)

            # Set x-axis to show only integer checkpoint numbers
            ax.set_xticks(range(1, 9))  # Assuming checkpoints 1-8

        # Set shared y-axis limits with some padding
        if all_y_values:
            y_min = min(all_y_values) - 5
            y_max = max(all_y_values) + 5
            axes[0].set_ylim(y_min, y_max)

        # Remove suptitle - no overall title needed

        plt.tight_layout()

        # Save the figure
        if save_plots:
            fn = f"{base_model_name}_all_methods_distance_progression_{dataset}"
            png_path = png_dir / f"{fn}.png"
            pdf_path = pdf_dir / f"{fn}.pdf"
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"  Saved to: PNG/{fn}.png and PDF/{fn}.pdf")

        if show_plots:
            plt.show()
        else:
            plt.close()

    print(f"\nCompleted generating combined distance progression plots!")


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

        # Create PNG and PDF subdirectories
        png_dir = PLOT_DIR / "PNG"
        pdf_dir = PLOT_DIR / "PDF"
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving all plots to: {PLOT_DIR}")
        print(f"  PNG files in: PNG/")
        print(f"  PDF files in: PDF/")
    else:
        # Define these for non-save mode
        png_dir = PLOT_DIR / "PNG"
        pdf_dir = PLOT_DIR / "PDF"

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
                    fontsize=18)
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
                    png_path = png_dir / f"{fn}.png"
                    pdf_path = pdf_dir / f"{fn}.pdf"
                    plt.savefig(png_path, dpi=150, bbox_inches='tight')
                    plt.savefig(pdf_path, bbox_inches='tight')
                    print(f"    Saved: PNG/{fn}.png and PDF/{fn}.pdf")

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
         show_plots: bool = False,
         show_wmdp: bool = False,
         filter_path: Optional[str] = None,
         apply_average: bool = False,
         window_size: int = 10):
    """
    Main function to load data and generate plots.

    Args:
        results_dir: Directory containing results (if None, uses default paths)
        dataset: 'bio' or 'chem'
        plot_type: 'combined', 'grid', 'delta', or 'progression'
        checkpoint_selector: 'best', 'last', or specific checkpoint
        results_fn_name: 'legacy' or 'dedup'
        show_plots: Whether to display plots interactively (default: False, only save)
        show_wmdp: Whether to show WMDP results as stars
        filter_path: Path to bio_neighbor_quality_results.json for topic filtering
    """
    # Select results directory
    if results_dir is None:
        if dataset == 'bio':
            results_dir = str(bio_results_path)
        elif dataset == 'chem':
            results_dir = str(chem_results_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    # Load topic filter if provided
    valid_topics = set()
    if filter_path:
        valid_topics = load_neighbor_quality_filter(Path(filter_path))
        if not valid_topics:
            print(
                "Warning: No valid topics found in filter, proceeding without filtering"
            )

    # Load data
    print(f"Loading data from {results_dir}")
    # Optimize loading based on plot type
    checkpoint_filter_to_use = None

    if plot_type == 'delta':
        # For delta plots, we need baseline and specific checkpoint
        # Load only the checkpoint we need (baseline is always loaded)
        if checkpoint_selector not in ['best', 'last', 'all']:
            checkpoint_filter_to_use = checkpoint_selector
            print(f"Loading baseline and {checkpoint_selector} for delta plot")
        else:
            checkpoint_filter_to_use = 'ckpt8'  # Default to ckpt8 for delta
            print(f"Loading baseline and ckpt8 for delta plot")
    elif plot_type == 'combined' and checkpoint_selector not in [
            'best', 'last'
    ]:
        checkpoint_filter_to_use = checkpoint_selector
        print(f"Filtering to only load checkpoint: {checkpoint_filter_to_use}")

    csvs, summary_jsons = load_ripple_bench_results(
        results_dir, checkpoint_filter=checkpoint_filter_to_use)

    print(f"Loaded data for {len(csvs)} models")

    # Apply topic filter if we have valid topics
    if valid_topics:
        print(f"\nApplying topic filter to {len(csvs)} models...")
        for model_name, checkpoints in csvs.items():
            for checkpoint_name, df in checkpoints.items():
                csvs[model_name][checkpoint_name] = apply_topic_filter(
                    df, valid_topics)

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
        base_results_fn = get_dedup_results
    else:
        base_results_fn = get_legacy_results

    # Wrap with rolling average if requested
    if apply_average:

        def results_fn(df):
            results = base_results_fn(df)
            return apply_rolling_average(results, window_size)

        print(f"Applying rolling average with window size: {window_size}")
    else:
        results_fn = base_results_fn

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
                                show_plots=show_plots,
                                show_wmdp_results=show_wmdp)

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
                       show_plots=show_plots,
                       show_wmdp_results=show_wmdp)

    elif plot_type == 'progression':
        print("\nGenerating checkpoint progression plots for all methods...")
        plot_all_checkpoint_progressions(base_models,
                                         unlearning_results,
                                         results_fn=results_fn,
                                         save_plots=True,
                                         dataset=dataset,
                                         show_plots=show_plots)

    elif plot_type == 'distance':
        print("\nGenerating distance progression plots for all methods...")
        plot_all_distance_progressions(
            base_models,
            unlearning_results,
            distances=[1, 50, 500],  # Can be made configurable
            results_fn=results_fn,
            save_plots=True,
            dataset=dataset,
            show_plots=show_plots)

    elif plot_type == 'comparison':
        print("\nGenerating RMU vs ELM comparison plot...")
        plot_method_comparison(
            base_models,
            unlearning_results,
            methods_to_compare=['rmu', 'elm'],
            base_model_name='llama',  # Can be made configurable
            results_fn=results_fn,
            save_plots=True,
            dataset=dataset,
            show_plots=show_plots)

    elif plot_type == 'delta':
        print("\nGenerating delta plots (checkpoint - baseline)...")
        plot_delta_from_baseline(base_models,
                                 unlearning_results,
                                 checkpoint=checkpoint_selector
                                 if checkpoint_selector != 'best' else 'ckpt8',
                                 results_fn=results_fn,
                                 save_plots=True,
                                 dataset=dataset,
                                 show_plots=show_plots,
                                 apply_average=apply_average,
                                 window_size=window_size)

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
    parser.add_argument(
        '--plot',
        type=str,
        nargs='+',
        default=['combined'],
        choices=[
            'combined', 'grid', 'delta', 'progression', 'distance',
            'comparison', 'all'
        ],
        help='Type(s) of plot to generate. Can specify multiple or "all"')
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
                        '--results-dir',
                        type=str,
                        default=None,
                        dest='dir',
                        help='Custom results directory')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively (default: only save)')
    parser.add_argument('--wmdp',
                        action='store_true',
                        help='Show WMDP results as stars at distance 0')
    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        help='Path to bio_neighbor_quality_results.json for topic filtering')
    parser.add_argument('--average',
                        action='store_true',
                        help='Apply rolling average smoothing to the plots')
    parser.add_argument('--window',
                        type=int,
                        default=10,
                        help='Window size for rolling average (default: 10)')

    args = parser.parse_args()

    # Handle 'all' option and convert to list if single value
    plot_types = args.plot
    if 'all' in plot_types:
        plot_types = [
            'combined', 'progression', 'distance', 'comparison', 'delta'
        ]

    # Load data once (will be cached for subsequent calls)
    first_run = True
    for plot_type in plot_types:
        if len(plot_types) > 1:
            print(f"\n{'='*60}")
            print(f"Generating {plot_type} plot...")
            print('=' * 60)

        csvs, summary_jsons, base_models, unlearning_results = main(
            results_dir=args.dir,
            dataset=args.dataset,
            plot_type=plot_type,
            checkpoint_selector=args.checkpoint,
            results_fn_name=args.results_fn,
            show_plots=args.show,
            show_wmdp=args.wmdp if hasattr(args, 'wmdp') else False,
            filter_path=args.filter,
            apply_average=args.average if hasattr(args, 'average') else False,
            window_size=args.window if hasattr(args, 'window') else 10)
