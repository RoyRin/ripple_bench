#!/usr/bin/env python3
"""
Analyze and visualize results from two model evaluations on Ripple Bench

Usage:
    python analyze_ripple_results.py <base_csv> <comparison_csv> --output-dir <output_dir> [--dataset <dataset.json>]
    
Example:
    python analyze_ripple_results.py zephyr_base_results.csv zephyr_elm_results.csv --output-dir analysis_results --dataset ripple_bench.json
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from ripple_bench.utils import read_dict


def load_results(csv_path: str) -> pd.DataFrame:
    """Load evaluation results from CSV."""
    df = pd.read_csv(csv_path)

    # Load summary if available
    summary_path = Path(csv_path).with_suffix('.summary.json')
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            print(
                f"Loaded {summary['model_name']}: {summary['accuracy']:.2%} accuracy"
            )

    return df


def analyze_differences(base_df: pd.DataFrame,
                        comparison_df: pd.DataFrame) -> dict:
    """Analyze differences between two model evaluations."""

    # Merge on question_id to align results
    merged = pd.merge(
        base_df[[
            'question_id', 'question', 'topic', 'correct_answer',
            'model_response', 'is_correct'
        ]],
        comparison_df[['question_id', 'model_response', 'is_correct']],
        on='question_id',
        suffixes=('_base', '_comp'))

    # Calculate differences
    analysis = {
        'base_accuracy':
        base_df['is_correct'].mean(),
        'comp_accuracy':
        comparison_df['is_correct'].mean(),
        'accuracy_diff':
        base_df['is_correct'].mean() - comparison_df['is_correct'].mean(),
        'base_model':
        base_df['model_name'].iloc[0],
        'comp_model':
        comparison_df['model_name'].iloc[0],
    }

    # Categorize changes
    merged['change_type'] = 'unchanged'
    merged.loc[(merged['is_correct_base'] == True) &
               (merged['is_correct_comp'] == False),
               'change_type'] = 'degraded'
    merged.loc[(merged['is_correct_base'] == False) &
               (merged['is_correct_comp'] == True), 'change_type'] = 'improved'

    analysis['degraded_count'] = (merged['change_type'] == 'degraded').sum()
    analysis['improved_count'] = (merged['change_type'] == 'improved').sum()
    analysis['unchanged_count'] = (merged['change_type'] == 'unchanged').sum()

    # Topic-wise analysis
    topic_analysis = merged.groupby('topic').agg({
        'is_correct_base': 'mean',
        'is_correct_comp': 'mean',
        'question_id': 'count'
    }).rename(columns={'question_id': 'count'})

    topic_analysis['accuracy_diff'] = topic_analysis[
        'is_correct_base'] - topic_analysis['is_correct_comp']
    topic_analysis = topic_analysis.sort_values('accuracy_diff',
                                                ascending=False)

    analysis['topic_analysis'] = topic_analysis
    analysis['merged_df'] = merged

    return analysis


def analyze_ripple_effects(base_df: pd.DataFrame,
                           comparison_df: pd.DataFrame,
                           dataset: dict = None) -> dict:
    """Analyze ripple effects based on topic distances if dataset is provided."""

    if not dataset or 'topic_to_neighbors' not in dataset:
        return None

    print("Analyzing ripple effects based on topic distances...")

    topic_to_neighbors = dataset['topic_to_neighbors']

    # Merge dataframes
    merged = pd.merge(base_df[['question_id', 'topic', 'is_correct']],
                      comparison_df[['question_id', 'is_correct']],
                      on='question_id',
                      suffixes=('_base', '_comp'))

    # Calculate distance for each question
    distances = []
    for _, row in merged.iterrows():
        topic = row['topic']
        distance = None

        # Find distance from original WMDP topics
        for orig_topic, neighbors in topic_to_neighbors.items():
            if topic == orig_topic:
                distance = 0
                break
            elif topic in neighbors:
                distance = neighbors.index(topic) + 1
                break

        # If not found, assign max distance
        if distance is None:
            distance = max(
                len(neighbors)
                for neighbors in topic_to_neighbors.values()) + 1

        distances.append(distance)

    merged['distance'] = distances

    # Aggregate by distance
    ripple_stats = merged.groupby('distance').agg({
        'is_correct_base': ['sum', 'count', 'mean'],
        'is_correct_comp': ['sum', 'count', 'mean']
    })

    ripple_stats.columns = [
        'base_correct', 'base_total', 'base_accuracy', 'comp_correct',
        'comp_total', 'comp_accuracy'
    ]

    ripple_stats['accuracy_diff'] = ripple_stats[
        'base_accuracy'] - ripple_stats['comp_accuracy']
    ripple_stats['num_questions'] = ripple_stats['base_total']

    # Count unique topics per distance
    topics_per_distance = merged.groupby('distance')['topic'].nunique()
    ripple_stats['num_topics'] = topics_per_distance

    return {'ripple_stats': ripple_stats, 'merged_with_distance': merged}


def create_visualizations(analysis: dict,
                          output_dir: Path,
                          ripple_analysis: dict = None):
    """Create visualization plots."""

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 1. Accuracy Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))

    models = [
        analysis['base_model'].split('/')[-1],
        analysis['comp_model'].split('/')[-1]
    ]
    accuracies = [analysis['base_accuracy'], analysis['comp_accuracy']]
    colors = ['#2E86AB', '#E63946']

    bars = ax.bar(models, accuracies, color=colors, width=0.6)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f'{acc:.1%}',
                ha='center',
                va='bottom',
                fontsize=14,
                fontweight='bold')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Comparison on Ripple Bench',
                 fontsize=16,
                 fontweight='bold')
    ax.set_ylim(0, max(accuracies) * 1.15)

    # Add difference annotation
    mid_x = 0.5
    mid_y = max(accuracies) * 0.5
    ax.annotate(f'Δ = {analysis["accuracy_diff"]:.1%}',
                xy=(mid_x, mid_y),
                xycoords='axes fraction',
                ha='center',
                fontsize=14,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor='yellow',
                          alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_dir / f"accuracy_comparison_{timestamp}.png", dpi=150)
    plt.close()

    # 2. Change Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    categories = ['Degraded', 'Unchanged', 'Improved']
    counts = [
        analysis['degraded_count'], analysis['unchanged_count'],
        analysis['improved_count']
    ]
    colors = ['#E63946', '#F1FAEE', '#2E86AB']

    bars = ax1.bar(categories, counts, color=colors, edgecolor='black')
    ax1.set_xlabel('Performance Change', fontsize=12)
    ax1.set_ylabel('Number of Questions', fontsize=12)
    ax1.set_title('Question Performance Changes', fontsize=14)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.,
                 height + 0.5,
                 f'{count}',
                 ha='center',
                 va='bottom',
                 fontsize=11)

    # Pie chart
    ax2.pie(counts,
            labels=categories,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 11})
    ax2.set_title('Performance Change Distribution', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / f"change_distribution_{timestamp}.png", dpi=150)
    plt.close()

    # 3. Topic-wise Performance Difference
    topic_df = analysis['topic_analysis']

    # Filter topics with at least 5 questions
    significant_topics = topic_df[topic_df['count'] >= 5].head(20)

    if len(significant_topics) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(significant_topics))
        diffs = significant_topics['accuracy_diff'].values
        colors = ['#E63946' if d > 0 else '#2E86AB' for d in diffs]

        bars = ax.barh(y_pos, diffs, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([
            t[:40] + '...' if len(t) > 40 else t
            for t in significant_topics.index
        ])
        ax.set_xlabel('Accuracy Difference (Base - Comparison)', fontsize=12)
        ax.set_title('Performance Difference by Topic', fontsize=14)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

        # Add value labels
        for bar, diff in zip(bars, diffs):
            width = bar.get_width()
            label_x = width + (0.02 if width > 0 else -0.02)
            ax.text(label_x,
                    bar.get_y() + bar.get_height() / 2,
                    f'{diff:.2f}',
                    ha='left' if width > 0 else 'right',
                    va='center',
                    fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / f"topic_differences_{timestamp}.png", dpi=150)
        plt.close()

    # 4. Accuracy Scatter Plot by Topic
    if len(topic_df) > 10:
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(
            topic_df['is_correct_base'],
            topic_df['is_correct_comp'],
            s=topic_df['count'] * 10,  # Size by number of questions
            alpha=0.6,
            c=topic_df['accuracy_diff'],
            cmap='RdBu_r',
            edgecolors='black',
            linewidth=0.5)

        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No change')

        ax.set_xlabel(f'{analysis["base_model"].split("/")[-1]} Accuracy',
                      fontsize=12)
        ax.set_ylabel(f'{analysis["comp_model"].split("/")[-1]} Accuracy',
                      fontsize=12)
        ax.set_title('Topic-wise Accuracy Comparison', fontsize=14)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Accuracy Difference', rotation=270, labelpad=20)

        plt.tight_layout()
        plt.savefig(output_dir / f"topic_scatter_{timestamp}.png", dpi=150)
        plt.close()

    # 5. Ripple Effect Visualization (if available)
    if ripple_analysis:
        fig, ax = plt.subplots(figsize=(10, 6))

        ripple_stats = ripple_analysis['ripple_stats']
        distances = ripple_stats.index.tolist()
        base_accs = ripple_stats['base_accuracy'].tolist()
        comp_accs = ripple_stats['comp_accuracy'].tolist()

        # Plot lines
        ax.plot(distances,
                base_accs,
                'o-',
                label=analysis['base_model'].split('/')[-1],
                color='#2E86AB',
                markersize=10,
                linewidth=2)
        ax.plot(distances,
                comp_accs,
                's-',
                label=analysis['comp_model'].split('/')[-1],
                color='#E63946',
                markersize=10,
                linewidth=2)

        # Add shaded region showing difference
        ax.fill_between(distances,
                        base_accs,
                        comp_accs,
                        alpha=0.2,
                        color='gray',
                        label='Performance Gap')

        # Add annotations for number of questions at each distance
        for dist in distances:
            num_q = ripple_stats.loc[dist, 'num_questions']
            ax.text(dist,
                    max(base_accs[distances.index(dist)],
                        comp_accs[distances.index(dist)]) + 0.02,
                    f'n={num_q}',
                    ha='center',
                    fontsize=9,
                    alpha=0.7)

        ax.set_xlabel('Distance from Original WMDP Topics', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Ripple Effect: Performance Drop by Topic Distance',
                     fontsize=14,
                     fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(distances)
        ax.set_ylim(0, max(max(base_accs), max(comp_accs)) * 1.1)

        plt.tight_layout()
        plt.savefig(output_dir / f"ripple_effect_{timestamp}.png", dpi=150)
        plt.close()

    print(f"Visualizations saved to {output_dir}")


def generate_report(analysis: dict,
                    output_dir: Path,
                    ripple_analysis: dict = None):
    """Generate analysis report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Ripple Bench Model Comparison Report
Generated: {timestamp}

## Models Compared
- **Base Model**: {analysis['base_model']}
- **Comparison Model**: {analysis['comp_model']}

## Overall Performance
- **Base Model Accuracy**: {analysis['base_accuracy']:.2%}
- **Comparison Model Accuracy**: {analysis['comp_accuracy']:.2%}
- **Accuracy Difference**: {analysis['accuracy_diff']:.2%}

## Question-Level Changes
- **Degraded**: {analysis['degraded_count']} questions
- **Improved**: {analysis['improved_count']} questions
- **Unchanged**: {analysis['unchanged_count']} questions

## Top Topics with Performance Drop
"""

    # Add top topics with drops
    topic_df = analysis['topic_analysis']
    top_drops = topic_df[topic_df['accuracy_diff'] > 0].head(10)

    if len(top_drops) > 0:
        report += "\n| Topic | Base Acc | Comp Acc | Difference | Questions |\n"
        report += "|-------|----------|----------|------------|----------|\n"

        for topic, row in top_drops.iterrows():
            report += f"| {topic[:50]} | {row['is_correct_base']:.1%} | {row['is_correct_comp']:.1%} | {row['accuracy_diff']:.1%} | {int(row['count'])} |\n"

    # Add sample degraded questions
    merged_df = analysis['merged_df']
    degraded_samples = merged_df[merged_df['change_type'] == 'degraded'].head(
        5)

    if len(degraded_samples) > 0:
        report += "\n## Sample Questions with Degraded Performance\n"
        for idx, row in degraded_samples.iterrows():
            report += f"\n### Question {row['question_id']}\n"
            report += f"- **Topic**: {row['topic']}\n"
            report += f"- **Question**: {row['question'][:200]}...\n" if len(
                row['question']
            ) > 200 else f"- **Question**: {row['question']}\n"
            report += f"- **Correct Answer**: {row['correct_answer']}\n"
            report += f"- **Base Model**: {row['model_response_base']} ({'✓' if row['is_correct_base'] else '✗'})\n"
            report += f"- **Comparison Model**: {row['model_response_comp']} ({'✓' if row['is_correct_comp'] else '✗'})\n"

    # Add ripple effect analysis if available
    if ripple_analysis:
        report += "\n## Ripple Effect Analysis\n\n"
        report += "Performance drop by distance from original WMDP topics:\n\n"
        report += "| Distance | Base Acc | Comp Acc | Drop | Questions | Topics |\n"
        report += "|----------|----------|----------|------|-----------|--------|\n"

        ripple_stats = ripple_analysis['ripple_stats']
        for dist in ripple_stats.index:
            row = ripple_stats.loc[dist]
            report += f"| {dist} | {row['base_accuracy']:.1%} | {row['comp_accuracy']:.1%} | "
            report += f"{row['accuracy_diff']:.1%} | {int(row['num_questions'])} | {int(row['num_topics'])} |\n"

        report += "\n**Distance 0**: Original WMDP topics (directly unlearned)\n"
        report += "**Distance 1+**: Semantically related neighbor topics\n"

    # Save report
    report_path = output_dir / "comparison_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare model evaluation results")
    parser.add_argument("base_csv", help="CSV file with base model results")
    parser.add_argument("comparison_csv",
                        help="CSV file with comparison model results")
    parser.add_argument("--output-dir",
                        default="analysis_results",
                        help="Output directory for analysis results")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional: Ripple bench dataset JSON for distance-based analysis")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load results
    print("Loading evaluation results...")
    base_df = load_results(args.base_csv)
    comp_df = load_results(args.comparison_csv)

    # Load dataset if provided
    dataset = None
    if args.dataset:
        print(f"\nLoading dataset from {args.dataset}...")
        dataset = read_dict(args.dataset)

    # Analyze differences
    print("\nAnalyzing differences...")
    analysis = analyze_differences(base_df, comp_df)

    # Analyze ripple effects if dataset provided
    ripple_analysis = None
    if dataset:
        ripple_analysis = analyze_ripple_effects(base_df, comp_df, dataset)

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(analysis, output_dir, ripple_analysis)

    # Generate report
    print("\nGenerating report...")
    generate_report(analysis, output_dir, ripple_analysis)

    print(f"\n{'='*50}")
    print("Analysis Complete!")
    print(f"{'='*50}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
