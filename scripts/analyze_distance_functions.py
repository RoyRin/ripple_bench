#!/usr/bin/env python3
"""
Analyze different distance functions to find what best captures the ripple effect.
Filters out topics where base model accuracy is ≤ 40%.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from ripple_bench.utils import read_dict
import warnings

warnings.filterwarnings('ignore')

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


def load_and_filter_results(base_csv, unlearned_csv, min_base_accuracy=0.4):
    """Load results and filter out topics with low base accuracy."""
    base_df = pd.read_csv(base_csv)
    unlearned_df = pd.read_csv(unlearned_csv)

    # Merge dataframes
    merged = pd.merge(
        base_df[['question_id', 'topic', 'is_correct', 'distance']],
        unlearned_df[['question_id', 'is_correct']],
        on='question_id',
        suffixes=('_base', '_unlearned'))

    # Calculate topic-level statistics
    topic_stats = merged.groupby('topic').agg({
        'is_correct_base': 'mean',
        'is_correct_unlearned': 'mean',
        'distance': 'first',
        'question_id': 'count'
    }).rename(columns={'question_id': 'n_questions'})

    # Filter topics with base accuracy > min_base_accuracy
    topic_stats = topic_stats[topic_stats['is_correct_base'] >
                              min_base_accuracy]

    # Calculate accuracy delta (base - unlearned)
    topic_stats['accuracy_delta'] = topic_stats[
        'is_correct_base'] - topic_stats['is_correct_unlearned']

    print(
        f"Filtered from {len(merged['topic'].unique())} to {len(topic_stats)} topics"
    )
    print(f"Removed topics with base accuracy ≤ {min_base_accuracy*100:.0f}%")

    return topic_stats, merged


def calculate_embedding_distances(topic_stats, dataset_path):
    """Calculate embedding-based distances between topics."""
    dataset = read_dict(dataset_path)

    # Get topic embeddings if available
    topic_embeddings = dataset.get('topic_embeddings', {})

    if not topic_embeddings:
        print("Warning: No topic embeddings found in dataset")
        return None

    # Calculate pairwise cosine distances
    topics = list(topic_stats.index)
    embeddings = []

    for topic in topics:
        if topic in topic_embeddings:
            embeddings.append(topic_embeddings[topic])
        else:
            # Use zero vector if embedding not found
            embeddings.append(
                np.zeros(len(next(iter(topic_embeddings.values())))))

    embeddings = np.array(embeddings)

    # Calculate cosine similarity and convert to distance
    similarities = cosine_similarity(embeddings)
    distances = 1 - similarities

    return distances, topics


def analyze_distance_correlation(topic_stats, distance_name, distance_values):
    """Analyze correlation between distance metric and accuracy delta."""
    # Calculate correlation
    valid_mask = ~np.isnan(distance_values) & ~np.isnan(
        topic_stats['accuracy_delta'])
    if valid_mask.sum() < 10:
        return None

    correlation, p_value = stats.pearsonr(
        distance_values[valid_mask], topic_stats['accuracy_delta'][valid_mask])

    spearman_corr, spearman_p = stats.spearmanr(
        distance_values[valid_mask], topic_stats['accuracy_delta'][valid_mask])

    return {
        'distance_name': distance_name,
        'pearson_r': correlation,
        'pearson_p': p_value,
        'spearman_r': spearman_corr,
        'spearman_p': spearman_p,
        'n_valid': valid_mask.sum()
    }


def plot_distance_analysis(topic_stats, output_dir, model_name):
    """Create visualizations for distance analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Distribution of accuracy deltas by original distance
    plt.figure(figsize=(12, 8))

    # Group by distance and plot
    distance_groups = topic_stats.groupby('distance')['accuracy_delta'].apply(
        list)
    distances = []
    deltas = []

    for dist, delta_list in distance_groups.items():
        distances.extend([dist] * len(delta_list))
        deltas.extend(delta_list)

    # Create violin plot
    plt.violinplot([
        topic_stats[topic_stats['distance'] == d]['accuracy_delta'].values
        for d in sorted(topic_stats['distance'].unique())[:20]
    ],
                   positions=sorted(topic_stats['distance'].unique())[:20],
                   showmeans=True,
                   showmedians=True)

    plt.xlabel('Distance from WMDP Topics', fontsize=14)
    plt.ylabel('Accuracy Delta (Base - Unlearned)', fontsize=14)
    plt.title(
        f'Distribution of Accuracy Changes by Distance\n{model_name} (Base accuracy > 40%)',
        fontsize=16,
        fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.85])
    plt.savefig(output_dir /
                f'accuracy_delta_distribution_{model_name.lower()}.pdf',
                bbox_inches='tight')
    plt.savefig(output_dir /
                f'accuracy_delta_distribution_{model_name.lower()}.png',
                dpi=300,
                bbox_inches='tight')
    plt.close()

    # 2. Scatter plot: Base accuracy vs accuracy delta
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(topic_stats['is_correct_base'],
                          topic_stats['accuracy_delta'],
                          c=topic_stats['distance'],
                          cmap='viridis',
                          alpha=0.6,
                          s=50)

    plt.colorbar(scatter, label='Distance from WMDP')
    plt.xlabel('Base Model Accuracy', fontsize=14)
    plt.ylabel('Accuracy Delta (Base - Unlearned)', fontsize=14)
    plt.title(
        f'Base Accuracy vs Unlearning Effect\n{model_name} (Base accuracy > 40%)',
        fontsize=16,
        fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=0.4,
                color='red',
                linestyle='--',
                alpha=0.5,
                label='40% threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.85])
    plt.savefig(output_dir / f'base_vs_delta_{model_name.lower()}.pdf',
                bbox_inches='tight')
    plt.savefig(output_dir / f'base_vs_delta_{model_name.lower()}.png',
                dpi=300,
                bbox_inches='tight')
    plt.close()

    # 3. Top affected topics
    top_affected = topic_stats.nlargest(20, 'accuracy_delta')

    plt.figure(figsize=(12, 10))
    plt.barh(range(len(top_affected)), top_affected['accuracy_delta'])
    plt.yticks(range(len(top_affected)), top_affected.index)
    plt.xlabel('Accuracy Delta (Base - Unlearned)', fontsize=14)
    plt.title(
        f'Top 20 Most Affected Topics\n{model_name} (Base accuracy > 40%)',
        fontsize=16,
        fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add distance annotations
    for i, (idx, row) in enumerate(top_affected.iterrows()):
        plt.text(row['accuracy_delta'] + 0.01,
                 i,
                 f"d={row['distance']}",
                 va='center',
                 fontsize=10,
                 alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.85])
    plt.savefig(output_dir / f'top_affected_topics_{model_name.lower()}.pdf',
                bbox_inches='tight')
    plt.savefig(output_dir / f'top_affected_topics_{model_name.lower()}.png',
                dpi=300,
                bbox_inches='tight')
    plt.close()


def explore_alternative_distances(topic_stats, output_dir, model_name):
    """Explore different ways to measure distance and their correlation with unlearning effect."""
    correlations = []

    # 1. Original RAG distance
    rag_distances = topic_stats['distance'].values
    corr_result = analyze_distance_correlation(topic_stats, 'RAG Distance',
                                               rag_distances)
    if corr_result:
        correlations.append(corr_result)

    # 2. Log-transformed distance
    log_distances = np.log1p(rag_distances)
    corr_result = analyze_distance_correlation(topic_stats,
                                               'Log(1 + Distance)',
                                               log_distances)
    if corr_result:
        correlations.append(corr_result)

    # 3. Inverse distance
    inverse_distances = 1 / (1 + rag_distances)
    corr_result = analyze_distance_correlation(topic_stats,
                                               '1 / (1 + Distance)',
                                               inverse_distances)
    if corr_result:
        correlations.append(corr_result)

    # 4. Exponential decay
    exp_distances = np.exp(-rag_distances / 10)  # decay rate of 10
    corr_result = analyze_distance_correlation(topic_stats,
                                               'exp(-Distance/10)',
                                               exp_distances)
    if corr_result:
        correlations.append(corr_result)

    # 5. Binary: WMDP topic or not
    is_wmdp = (rag_distances == 0).astype(float)
    corr_result = analyze_distance_correlation(topic_stats, 'Is WMDP Topic',
                                               is_wmdp)
    if corr_result:
        correlations.append(corr_result)

    # Create correlation summary plot
    if correlations:
        corr_df = pd.DataFrame(correlations)

        plt.figure(figsize=(10, 6))
        x = range(len(corr_df))

        plt.bar(x, corr_df['pearson_r'], alpha=0.7, label='Pearson')
        plt.bar(x, corr_df['spearman_r'], alpha=0.7, label='Spearman')

        plt.xticks(x, corr_df['distance_name'], rotation=45, ha='right')
        plt.ylabel('Correlation with Accuracy Delta', fontsize=14)
        plt.title(
            f'Distance Metric Correlations with Unlearning Effect\n{model_name}',
            fontsize=16,
            fontweight='bold')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout(rect=[0, 0.03, 1, 0.85])
        plt.savefig(Path(output_dir) /
                    f'distance_correlations_{model_name.lower()}.pdf',
                    bbox_inches='tight')
        plt.savefig(Path(output_dir) /
                    f'distance_correlations_{model_name.lower()}.png',
                    dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Save correlation results
        corr_df.to_csv(Path(output_dir) /
                       f'distance_correlations_{model_name.lower()}.csv',
                       index=False)

        print(f"\nDistance Metric Correlations for {model_name}:")
        print(corr_df.to_string(index=False))

    return correlations


def main():
    parser = argparse.ArgumentParser(
        description="Analyze distance functions for ripple effect")
    parser.add_argument("base_csv", help="Path to base model CSV results")
    parser.add_argument("unlearned_csv",
                        help="Path to unlearned model CSV results")
    parser.add_argument("--dataset",
                        required=True,
                        help="Path to ripple bench dataset JSON")
    parser.add_argument("--output-dir",
                        default="distance_analysis",
                        help="Output directory")
    parser.add_argument("--model-name",
                        default="Unlearned",
                        help="Name of unlearned model")
    parser.add_argument(
        "--min-base-accuracy",
        type=float,
        default=0.4,
        help="Minimum base accuracy to include topic (default: 0.4)")

    args = parser.parse_args()

    # Load and filter data
    topic_stats, merged_df = load_and_filter_results(args.base_csv,
                                                     args.unlearned_csv,
                                                     args.min_base_accuracy)

    # Create visualizations
    plot_distance_analysis(topic_stats, args.output_dir, args.model_name)

    # Explore alternative distance metrics
    explore_alternative_distances(topic_stats, args.output_dir,
                                  args.model_name)

    # Sort topics by how much they're affected
    topics_by_effect = topic_stats.sort_values('accuracy_delta',
                                               ascending=False)

    # Save detailed results
    output_path = Path(
        args.output_dir) / f'topic_analysis_{args.model_name.lower()}.csv'
    topics_by_effect.to_csv(output_path)
    print(f"\nDetailed topic analysis saved to: {output_path}")

    # Print summary statistics
    print(f"\nSummary Statistics for {args.model_name}:")
    print(f"Number of topics analyzed: {len(topic_stats)}")
    print(f"Mean accuracy delta: {topic_stats['accuracy_delta'].mean():.4f}")
    print(f"Std accuracy delta: {topic_stats['accuracy_delta'].std():.4f}")
    print(
        f"Topics with positive delta (hurt by unlearning): {(topic_stats['accuracy_delta'] > 0).sum()}"
    )
    print(
        f"Topics with negative delta (helped by unlearning): {(topic_stats['accuracy_delta'] < 0).sum()}"
    )


if __name__ == "__main__":
    main()
