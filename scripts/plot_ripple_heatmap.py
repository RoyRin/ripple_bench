#!/usr/bin/env python3
"""
Create heatmap visualization of ripple effects for unlearning.
Each row is a WMDP topic, columns show semantic neighbors at increasing distances.
Color indicates accuracy delta (base - unlearned).
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


def load_evaluation_results(results_dir, model_name):
    """Load evaluation results from CSV for a specific model."""
    csv_file = Path(results_dir) / f"{model_name}.csv"

    results = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question_id = row['question_id']
            results[question_id] = {
                'topic': row['topic'],
                'distance': int(row['distance']),
                'prediction': row['model_response'],
                'answer': row['correct_answer'],
                'question': row['question'],
                'is_correct': row['is_correct'] == 'True'
            }

    return results


def create_ripple_heatmap(base_results,
                          unlearned_results,
                          output_path,
                          model_name,
                          max_topics=50):
    """Create heatmap visualization of ripple effects."""

    # Organize data by topic and distance
    topic_data = defaultdict(lambda: defaultdict(dict))

    # Process all questions
    for question_id in base_results:
        if question_id not in unlearned_results:
            continue

        base_item = base_results[question_id]
        unlearned_item = unlearned_results[question_id]

        topic = base_item['topic']
        distance = base_item['distance']

        # Calculate accuracy delta
        base_correct = base_item['prediction'] == base_item['answer']
        unlearned_correct = unlearned_item['prediction'] == unlearned_item[
            'answer']

        # Store the result
        if 'base_correct' not in topic_data[topic][distance]:
            topic_data[topic][distance]['base_correct'] = 0
            topic_data[topic][distance]['unlearned_correct'] = 0
            topic_data[topic][distance]['count'] = 0

        topic_data[topic][distance]['base_correct'] += int(base_correct)
        topic_data[topic][distance]['unlearned_correct'] += int(
            unlearned_correct)
        topic_data[topic][distance]['count'] += 1

    # Filter to only WMDP topics (those that have distance 0)
    wmdp_topics = [topic for topic in topic_data if 0 in topic_data[topic]]

    # Sort topics by their distance 0 accuracy drop
    topic_drops = []
    for topic in wmdp_topics:
        if 0 in topic_data[topic] and topic_data[topic][0]['count'] > 0:
            base_acc = topic_data[topic][0]['base_correct'] / topic_data[
                topic][0]['count']
            unlearned_acc = topic_data[topic][0][
                'unlearned_correct'] / topic_data[topic][0]['count']
            drop = base_acc - unlearned_acc
            topic_drops.append((topic, drop))

    # Sort by accuracy drop and take top N topics
    topic_drops.sort(key=lambda x: x[1], reverse=True)
    selected_topics = [t[0] for t in topic_drops[:max_topics]]

    # Create matrix for heatmap
    max_distance = 100
    matrix = np.full((len(selected_topics), max_distance + 1), np.nan)

    for i, topic in enumerate(selected_topics):
        for distance in range(max_distance + 1):
            if distance in topic_data[topic] and topic_data[topic][distance][
                    'count'] > 0:
                base_acc = topic_data[topic][distance][
                    'base_correct'] / topic_data[topic][distance]['count']
                unlearned_acc = topic_data[topic][distance][
                    'unlearned_correct'] / topic_data[topic][distance]['count']
                accuracy_delta = base_acc - unlearned_acc
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
    ax.set_yticks([])  # Don't show topic names (too many)

    # Set labels
    ax.set_xlabel('Semantic Distance from WMDP Topic')
    ax.set_ylabel(f'WMDP Topics (top {max_topics} by accuracy drop)')
    ax.set_title(
        f'Ripple Effect Heatmap - {model_name}\n'
        f'Each row shows how unlearning affects a WMDP topic and its semantic neighbors'
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
    ax.set_yticks([])  # Don't show topic names (too many)

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
    # Paths
    results_dir = Path(
        "/Users/roy/code/research/unlearning/data_to_concept_unlearning/results/results"
    )
    output_dir = Path(
        "/Users/roy/code/research/unlearning/data_to_concept_unlearning/results/experiments"
    )
    output_dir.mkdir(exist_ok=True)

    # Load base model results
    print("Loading base model results...")
    base_results = load_evaluation_results(results_dir, "zephyr_base")

    # Process ELM
    print("\nProcessing ELM results...")
    elm_results = load_evaluation_results(results_dir, "zephyr_elm")
    create_ripple_heatmap(base_results, elm_results,
                          output_dir / "ripple_heatmap_elm.pdf", "ELM")

    # Process RMU
    print("\nProcessing RMU results...")
    rmu_results = load_evaluation_results(results_dir, "zephyr_rmu")
    create_ripple_heatmap(base_results, rmu_results,
                          output_dir / "ripple_heatmap_rmu.pdf", "RMU")

    print("\nDone! Heatmaps created.")


if __name__ == "__main__":
    main()
