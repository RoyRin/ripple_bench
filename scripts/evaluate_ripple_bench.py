#!/usr/bin/env python3
"""
Evaluate Models on Ripple Bench Dataset

This script evaluates base and unlearned models on a ripple bench dataset,
generating comprehensive visualizations and analysis of the ripple effects.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from ripple_bench.metrics import get_accuracy, answer_single_question
from ripple_bench.models import load_zephyr
from ripple_bench.utils import save_dict, read_dict


class RippleBenchEvaluator:
    def __init__(self, output_dir: str = "ripple_bench_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load ripple bench dataset from file."""
        print(f"Loading dataset from {dataset_path}")
        return read_dict(dataset_path)
    
    def evaluate_model(self, questions: List[Dict], 
                      model_path: str = None,
                      model_name: str = "model") -> Dict[str, Any]:
        """Evaluate a single model on questions."""
        print(f"Evaluating {model_name}...")
        
        if model_path:
            # Load model if path provided
            model, tokenizer = load_zephyr(model_path)
        else:
            # Use default model
            model, tokenizer = load_zephyr()
        
        correct = 0
        responses = []
        detailed_results = []
        
        for q in tqdm(questions, desc=f"Evaluating {model_name}"):
            try:
                # Get model response
                response = answer_single_question(
                    q['question'], 
                    q['choices'], 
                    model, 
                    tokenizer
                )
                
                # Check if correct
                is_correct = response == q['answer']
                correct += int(is_correct)
                
                responses.append(response)
                detailed_results.append({
                    'question': q['question'],
                    'choices': q['choices'],
                    'correct_answer': q['answer'],
                    'model_response': response,
                    'is_correct': is_correct,
                    'topic': q.get('topic', 'unknown'),
                    'source': q.get('source', 'unknown')
                })
                
            except Exception as e:
                print(f"Error evaluating question: {e}")
                responses.append(None)
                detailed_results.append({
                    'question': q['question'],
                    'error': str(e)
                })
        
        accuracy = correct / len(questions) if questions else 0
        
        return {
            'model_name': model_name,
            'model_path': model_path,
            'accuracy': accuracy,
            'correct': correct,
            'total': len(questions),
            'responses': responses,
            'detailed_results': detailed_results
        }
    
    def analyze_ripple_effects(self, dataset: Dict[str, Any], 
                             base_results: Dict[str, Any],
                             unlearned_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ripple effects by comparing model performances."""
        print("Analyzing ripple effects...")
        
        # Get topic relationships
        topic_to_neighbors = dataset.get('topic_to_neighbors', {})
        
        # Calculate distance for each question
        question_distances = []
        base_detailed = base_results['detailed_results']
        unlearned_detailed = unlearned_results['detailed_results']
        
        for i, q in enumerate(dataset['questions']):
            topic = q.get('topic', 'unknown')
            
            # Determine distance
            distance = None
            for orig_topic, neighbors in topic_to_neighbors.items():
                if topic == orig_topic:
                    distance = 0
                    break
                elif topic in neighbors:
                    distance = neighbors.index(topic) + 1
                    break
            
            # If not found in neighbors, assign max distance
            if distance is None:
                distance = len(next(iter(topic_to_neighbors.values()))) + 1
            
            question_distances.append({
                'topic': topic,
                'distance': distance,
                'base_correct': base_detailed[i].get('is_correct', False),
                'unlearned_correct': unlearned_detailed[i].get('is_correct', False),
                'question': q['question']
            })
        
        # Aggregate by distance
        distance_stats = {}
        for item in question_distances:
            dist = item['distance']
            if dist not in distance_stats:
                distance_stats[dist] = {
                    'base_correct': 0,
                    'base_total': 0,
                    'unlearned_correct': 0,
                    'unlearned_total': 0,
                    'topics': set()
                }
            
            distance_stats[dist]['base_correct'] += int(item['base_correct'])
            distance_stats[dist]['base_total'] += 1
            distance_stats[dist]['unlearned_correct'] += int(item['unlearned_correct'])
            distance_stats[dist]['unlearned_total'] += 1
            distance_stats[dist]['topics'].add(item['topic'])
        
        # Calculate accuracies by distance
        ripple_analysis = {}
        for dist, stats in distance_stats.items():
            ripple_analysis[dist] = {
                'base_accuracy': stats['base_correct'] / stats['base_total'] if stats['base_total'] > 0 else 0,
                'unlearned_accuracy': stats['unlearned_correct'] / stats['unlearned_total'] if stats['unlearned_total'] > 0 else 0,
                'num_questions': stats['base_total'],
                'num_topics': len(stats['topics'])
            }
        
        return {
            'question_distances': question_distances,
            'distance_stats': distance_stats,
            'ripple_analysis': ripple_analysis
        }
    
    def create_visualizations(self, dataset: Dict[str, Any],
                            base_results: Dict[str, Any],
                            unlearned_results: Dict[str, Any],
                            ripple_analysis: Dict[str, Any]):
        """Create comprehensive visualizations of results."""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Overall Accuracy Comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        models = ['Base Model', 'Unlearned Model']
        accuracies = [base_results['accuracy'], unlearned_results['accuracy']]
        bars = ax.bar(models, accuracies, color=['#2E86AB', '#E63946'])
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.2%}', ha='center', va='bottom')
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Overall Model Performance Comparison')
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"overall_accuracy_{self.timestamp}.png", dpi=150)
        plt.close()
        
        # 2. Ripple Effect Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract distance data
        distances = sorted(ripple_analysis['ripple_analysis'].keys())
        base_accs = [ripple_analysis['ripple_analysis'][d]['base_accuracy'] for d in distances]
        unlearned_accs = [ripple_analysis['ripple_analysis'][d]['unlearned_accuracy'] for d in distances]
        
        # Plot lines
        ax.plot(distances, base_accs, 'o-', label='Base Model', 
               color='#2E86AB', markersize=10, linewidth=2)
        ax.plot(distances, unlearned_accs, 's-', label='Unlearned Model', 
               color='#E63946', markersize=10, linewidth=2)
        
        # Add shaded region showing difference
        ax.fill_between(distances, base_accs, unlearned_accs, 
                       alpha=0.2, color='gray', label='Performance Gap')
        
        ax.set_xlabel('Distance from Original Topic')
        ax.set_ylabel('Accuracy')
        ax.set_title('Ripple Effect: Accuracy vs Topic Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(distances)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"ripple_effect_{self.timestamp}.png", dpi=150)
        plt.close()
        
        # 3. Accuracy by Topic Heatmap
        # Get unique topics and their accuracies
        topics_data = {}
        for result in base_results['detailed_results']:
            topic = result.get('topic', 'unknown')
            if topic not in topics_data:
                topics_data[topic] = {
                    'base_correct': 0, 'base_total': 0,
                    'unlearned_correct': 0, 'unlearned_total': 0
                }
            topics_data[topic]['base_correct'] += int(result.get('is_correct', False))
            topics_data[topic]['base_total'] += 1
        
        for result in unlearned_results['detailed_results']:
            topic = result.get('topic', 'unknown')
            if topic in topics_data:
                topics_data[topic]['unlearned_correct'] += int(result.get('is_correct', False))
                topics_data[topic]['unlearned_total'] += 1
        
        # Calculate accuracies and differences
        topic_list = list(topics_data.keys())[:30]  # Limit to 30 topics for visibility
        accuracy_diffs = []
        
        for topic in topic_list:
            base_acc = topics_data[topic]['base_correct'] / topics_data[topic]['base_total']
            unlearned_acc = topics_data[topic]['unlearned_correct'] / topics_data[topic]['unlearned_total']
            accuracy_diffs.append(base_acc - unlearned_acc)
        
        # Create heatmap data
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a matrix for visualization
        diff_matrix = np.array(accuracy_diffs).reshape(-1, 1)
        
        # Create heatmap
        sns.heatmap(diff_matrix.T, 
                   xticklabels=[t[:20] + '...' if len(t) > 20 else t for t in topic_list],
                   yticklabels=['Accuracy Diff'],
                   cmap='RdBu_r', 
                   center=0,
                   cbar_kws={'label': 'Accuracy Difference (Base - Unlearned)'},
                   annot=False)
        
        plt.xticks(rotation=45, ha='right')
        plt.title('Accuracy Difference by Topic')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"topic_accuracy_diff_{self.timestamp}.png", dpi=150)
        plt.close()
        
        # 4. Performance Drop Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram of accuracy differences
        all_diffs = []
        for i in range(len(base_results['detailed_results'])):
            base_correct = base_results['detailed_results'][i].get('is_correct', False)
            unlearned_correct = unlearned_results['detailed_results'][i].get('is_correct', False)
            diff = int(base_correct) - int(unlearned_correct)
            all_diffs.append(diff)
        
        ax1.hist(all_diffs, bins=[-1.5, -0.5, 0.5, 1.5], 
                color=['#E63946', '#F1FAEE', '#2E86AB'],
                edgecolor='black')
        ax1.set_xlabel('Performance Change')
        ax1.set_ylabel('Number of Questions')
        ax1.set_title('Distribution of Performance Changes')
        ax1.set_xticks([-1, 0, 1])
        ax1.set_xticklabels(['Degraded', 'Unchanged', 'Improved'])
        
        # Pie chart of performance categories
        degraded = sum(1 for d in all_diffs if d < 0)
        unchanged = sum(1 for d in all_diffs if d == 0)
        improved = sum(1 for d in all_diffs if d > 0)
        
        ax2.pie([degraded, unchanged, improved], 
               labels=['Degraded', 'Unchanged', 'Improved'],
               colors=['#E63946', '#F1FAEE', '#2E86AB'],
               autopct='%1.1f%%',
               startangle=90)
        ax2.set_title('Performance Change Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"performance_analysis_{self.timestamp}.png", dpi=150)
        plt.close()
        
        # 5. Sample Questions Analysis
        # Create a sample of questions showing different behaviors
        samples = {
            'base_only_correct': [],
            'unlearned_only_correct': [],
            'both_correct': [],
            'both_incorrect': []
        }
        
        for i in range(min(len(base_results['detailed_results']), 
                          len(unlearned_results['detailed_results']))):
            base_res = base_results['detailed_results'][i]
            unlearned_res = unlearned_results['detailed_results'][i]
            
            base_correct = base_res.get('is_correct', False)
            unlearned_correct = unlearned_res.get('is_correct', False)
            
            sample = {
                'question': base_res['question'],
                'topic': base_res.get('topic', 'unknown'),
                'correct_answer': base_res.get('correct_answer', 'N/A'),
                'base_response': base_res.get('model_response', 'N/A'),
                'unlearned_response': unlearned_res.get('model_response', 'N/A')
            }
            
            if base_correct and not unlearned_correct:
                samples['base_only_correct'].append(sample)
            elif not base_correct and unlearned_correct:
                samples['unlearned_only_correct'].append(sample)
            elif base_correct and unlearned_correct:
                samples['both_correct'].append(sample)
            else:
                samples['both_incorrect'].append(sample)
        
        # Save sample questions
        samples_file = self.output_dir / f"question_samples_{self.timestamp}.json"
        save_dict({k: v[:5] for k, v in samples.items()}, samples_file)  # Save 5 examples of each
        
        print(f"Visualizations saved to {self.output_dir}")
    
    def generate_report(self, dataset: Dict[str, Any],
                       base_results: Dict[str, Any],
                       unlearned_results: Dict[str, Any],
                       ripple_analysis: Dict[str, Any]):
        """Generate a comprehensive evaluation report."""
        print("Generating evaluation report...")
        
        report = f"""# Ripple Bench Evaluation Report
Generated: {self.timestamp}

## Dataset Information
- Source: {dataset['metadata']['wmdp_source']}
- Total WMDP Questions Used: {dataset['metadata']['num_samples_used']}
- Unique Topics: {dataset['metadata']['num_unique_topics']}
- K-Neighbors: {dataset['metadata']['k_neighbors']}
- Questions per Topic: {dataset['metadata']['questions_per_topic']}
- Total Generated Questions: {dataset['metadata']['total_generated_questions']}

## Overall Performance
### Base Model
- Accuracy: {base_results['accuracy']:.2%} ({base_results['correct']}/{base_results['total']})
- Model Path: {base_results['model_path'] or 'Default'}

### Unlearned Model
- Accuracy: {unlearned_results['accuracy']:.2%} ({unlearned_results['correct']}/{unlearned_results['total']})
- Model Path: {unlearned_results['model_path'] or 'Default'}

### Performance Change
- Absolute Change: {(unlearned_results['accuracy'] - base_results['accuracy']):.2%}
- Relative Change: {((unlearned_results['accuracy'] - base_results['accuracy']) / base_results['accuracy'] * 100):.1f}%

## Ripple Effect Analysis
"""
        
        # Add ripple effect details
        for dist in sorted(ripple_analysis['ripple_analysis'].keys()):
            stats = ripple_analysis['ripple_analysis'][dist]
            report += f"\n### Distance {dist}"
            report += f"\n- Base Accuracy: {stats['base_accuracy']:.2%}"
            report += f"\n- Unlearned Accuracy: {stats['unlearned_accuracy']:.2%}"
            report += f"\n- Performance Drop: {(stats['base_accuracy'] - stats['unlearned_accuracy']):.2%}"
            report += f"\n- Number of Questions: {stats['num_questions']}"
            report += f"\n- Number of Topics: {stats['num_topics']}\n"
        
        # Save report
        report_file = self.output_dir / f"evaluation_report_{self.timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {report_file}")
    
    def evaluate(self, 
                dataset_path: str,
                base_model_path: str = None,
                unlearned_model_path: str = None):
        """Run complete evaluation pipeline."""
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        questions = dataset['questions']
        
        print(f"Loaded {len(questions)} questions for evaluation")
        
        # Evaluate models
        base_results = self.evaluate_model(
            questions, 
            base_model_path, 
            model_name="Base Model"
        )
        
        unlearned_results = self.evaluate_model(
            questions, 
            unlearned_model_path, 
            model_name="Unlearned Model"
        )
        
        # Analyze ripple effects
        ripple_analysis = self.analyze_ripple_effects(
            dataset, 
            base_results, 
            unlearned_results
        )
        
        # Create visualizations
        self.create_visualizations(
            dataset,
            base_results,
            unlearned_results,
            ripple_analysis
        )
        
        # Generate report
        self.generate_report(
            dataset,
            base_results,
            unlearned_results,
            ripple_analysis
        )
        
        # Save all results
        all_results = {
            'metadata': {
                'timestamp': self.timestamp,
                'dataset_path': dataset_path,
                'base_model_path': base_model_path,
                'unlearned_model_path': unlearned_model_path
            },
            'base_results': base_results,
            'unlearned_results': unlearned_results,
            'ripple_analysis': ripple_analysis
        }
        
        results_file = self.output_dir / f"evaluation_results_{self.timestamp}.json"
        save_dict(all_results, results_file)
        
        print(f"\nEvaluation complete!")
        print(f"Results saved to: {self.output_dir}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Models on Ripple Bench Dataset")
    parser.add_argument("dataset_path", 
                       help="Path to ripple bench dataset JSON file")
    parser.add_argument("--base-model", default=None,
                       help="Path to base model (uses default if not specified)")
    parser.add_argument("--unlearned-model", default=None,
                       help="Path to unlearned model (uses default if not specified)")
    parser.add_argument("--output-dir", default="ripple_bench_evaluation",
                       help="Output directory for evaluation results")
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = RippleBenchEvaluator(output_dir=args.output_dir)
    results = evaluator.evaluate(
        dataset_path=args.dataset_path,
        base_model_path=args.base_model,
        unlearned_model_path=args.unlearned_model
    )


if __name__ == "__main__":
    main()