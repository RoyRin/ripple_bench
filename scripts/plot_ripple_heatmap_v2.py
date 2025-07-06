#!/usr/bin/env python3
"""
Create heatmap visualization of ripple effects for unlearning.
Version 2: Uses ripple_bench_dataset.json to track topic relationships.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import csv
from collections import defaultdict
import matplotlib.colors as mcolors

# Configure matplotlib styling
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


def load_ripple_bench_structure():
    """Load the ripple bench dataset to get topic relationships."""
    ripple_file = Path(
        "/Users/roy/code/research/unlearning/data_to_concept_unlearning/data/ripple_bench_2000/ripple_bench_dataset.json"
    )

    with open(ripple_file, 'r') as f:
        data = json.load(f)

    topics = data['topics']

    # Build mapping of WMDP topics to their neighbors
    wmdp_neighbors = defaultdict(list)

    for topic_entry in topics:
        if topic_entry.get('distance') == 0:
            # This is a WMDP topic
            topic_name = topic_entry.get('topic')
            wmdp_neighbors[topic_name] = []

    # Now find all neighbors
    for topic_entry in topics:
        original = topic_entry.get('original_topic')
        if original in wmdp_neighbors and topic_entry.get('distance') > 0:
            wmdp_neighbors[original].append({
                'topic':
                topic_entry.get('topic'),
                'distance':
                topic_entry.get('distance')
            })

    return wmdp_neighbors


def load_evaluation_results(results_dir, model_name):
    """Load evaluation results from CSV for a specific model."""
    csv_file = Path(results_dir) / f"{model_name}.csv"

    results = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = row['topic']
            if topic not in results:
                results[topic] = []
            results[topic].append({
                'question_id': row['question_id'],
                'distance': int(row['distance']),
                'prediction': row['model_response'],
                'answer': row['correct_answer'],
                'is_correct': row['is_correct'] == 'True'
            })

    return results


def calculate_topic_accuracy(questions):
    """Calculate accuracy for a list of questions."""
    if not questions:
        return None
    correct = sum(q['is_correct'] for q in questions)
    return correct / len(questions)


def create_ripple_heatmap_v2(base_results,
                             unlearned_results,
                             wmdp_neighbors,
                             output_path,
                             model_name,
                             max_topics=50):
    """Create heatmap visualization using proper topic relationships."""

    # Calculate accuracy for each topic
    base_accuracy = {}
    unlearned_accuracy = {}

    for topic, questions in base_results.items():
        base_accuracy[topic] = calculate_topic_accuracy(questions)

    for topic, questions in unlearned_results.items():
        unlearned_accuracy[topic] = calculate_topic_accuracy(questions)

    # Calculate accuracy drop for WMDP topics and sort by drop
    wmdp_drops = []
    for wmdp_topic in wmdp_neighbors.keys():
        if wmdp_topic in base_accuracy and wmdp_topic in unlearned_accuracy:
            if base_accuracy[wmdp_topic] is not None and unlearned_accuracy[
                    wmdp_topic] is not None:
                drop = base_accuracy[wmdp_topic] - unlearned_accuracy[
                    wmdp_topic]
                wmdp_drops.append((wmdp_topic, drop))

    # Sort by drop and select top N
    wmdp_drops.sort(key=lambda x: x[1], reverse=True)
    selected_topics = [t[0] for t in wmdp_drops[:max_topics]]

    # Create matrix for heatmap
    max_distance = 100
    matrix = np.full((len(selected_topics), max_distance + 1), np.nan)

    for i, wmdp_topic in enumerate(selected_topics):
        # Add the WMDP topic itself (distance 0)
        if wmdp_topic in base_accuracy and wmdp_topic in unlearned_accuracy:
            if base_accuracy[wmdp_topic] is not None and unlearned_accuracy[
                    wmdp_topic] is not None:
                matrix[i, 0] = base_accuracy[wmdp_topic] - unlearned_accuracy[
                    wmdp_topic]

        # Add all its neighbors
        for neighbor in wmdp_neighbors[wmdp_topic]:
            neighbor_topic = neighbor['topic']
            distance = neighbor['distance']

            if distance <= max_distance:
                if neighbor_topic in base_accuracy and neighbor_topic in unlearned_accuracy:
                    if base_accuracy[
                            neighbor_topic] is not None and unlearned_accuracy[
                                neighbor_topic] is not None:
                        accuracy_delta = base_accuracy[
                            neighbor_topic] - unlearned_accuracy[neighbor_topic]
                        matrix[i, distance] = accuracy_delta

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(16, 10))

    # Use a diverging colormap centered at 0
    cmap = plt.cm.RdBu_r  # Red-Blue diverging colormap
    norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)

    # Create heatmap
    im = ax.imshow(matrix,
                   cmap=cmap,
                   norm=norm,
                   aspect='auto',
                   interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy Delta (Base - Unlearned)',
                   rotation=270,
                   labelpad=20)

    # Set ticks
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xticklabels(np.arange(0, 101, 10))

    # Add y-axis labels with topic names (for readability, show every 5th)
    y_ticks = list(range(0, len(selected_topics), 5))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([
        selected_topics[i][:30] +
        '...' if len(selected_topics[i]) > 30 else selected_topics[i]
        for i in y_ticks
    ],
                       fontsize=10)

    # Set labels
    ax.set_xlabel('Semantic Distance from WMDP Topic')
    ax.set_ylabel(f'WMDP Topics (top {max_topics} by accuracy drop)')
    ax.set_title(
        f'Ripple Effect Heatmap - {model_name}\n'
        f'Each row shows how unlearning affects a specific WMDP topic and its semantic neighbors'
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Also create a more focused version (distances 0-20)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create focused heatmap
    im = ax.imshow(matrix[:, :21],
                   cmap=cmap,
                   norm=norm,
                   aspect='auto',
                   interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy Delta (Base - Unlearned)',
                   rotation=270,
                   labelpad=20)

    # Set ticks
    ax.set_xticks(np.arange(0, 21))
    ax.set_xticklabels(np.arange(0, 21))

    # Add y-axis labels
    y_ticks = list(range(0, len(selected_topics), 5))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([
        selected_topics[i][:30] +
        '...' if len(selected_topics[i]) > 30 else selected_topics[i]
        for i in y_ticks
    ],
                       fontsize=10)

    # Set labels
    ax.set_xlabel('Semantic Distance from WMDP Topic')
    ax.set_ylabel(f'WMDP Topics (top {max_topics} by accuracy drop)')
    ax.set_title(f'Ripple Effect Heatmap (Focused View) - {model_name}\n'
                 f'Showing distances 0-20 for clarity')

    # Adjust layout and save
    plt.tight_layout()
    focused_path = str(output_path).replace('.pdf', '_focused.pdf')
    plt.savefig(focused_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created heatmap: {output_path}")
    print(f"Created focused heatmap: {focused_path}")


def main():
    # Load ripple bench structure
    print("Loading ripple bench structure...")
    wmdp_neighbors = load_ripple_bench_structure()
    print(f"Found {len(wmdp_neighbors)} WMDP topics with neighbors")

    # Paths
    results_dir = Path(
        "/Users/roy/code/research/unlearning/data_to_concept_unlearning/results/results"
    )
    output_dir = Path(
        "/Users/roy/code/research/unlearning/data_to_concept_unlearning/results/experiments"
    )
    output_dir.mkdir(exist_ok=True)

    # Load base model results
    print("\nLoading base model results...")
    base_results = load_evaluation_results(results_dir, "zephyr_base")

    # Process ELM
    print("\nProcessing ELM results...")
    elm_results = load_evaluation_results(results_dir, "zephyr_elm")
    create_ripple_heatmap_v2(base_results, elm_results, wmdp_neighbors,
                             output_dir / "ripple_heatmap_elm_v2.pdf", "ELM")

    # Process RMU
    print("\nProcessing RMU results...")
    rmu_results = load_evaluation_results(results_dir, "zephyr_rmu")
    create_ripple_heatmap_v2(base_results, rmu_results, wmdp_neighbors,
                             output_dir / "ripple_heatmap_rmu_v2.pdf", "RMU")

    print("\nDone! Heatmaps created.")


if __name__ == "__main__":
    main()
