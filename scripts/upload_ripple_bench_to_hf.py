#!/usr/bin/env python3
"""
Upload Ripple Bench dataset to Hugging Face Hub

This script uploads a generated Ripple Bench dataset to:
https://huggingface.co/datasets/royrin/ripple-bench
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo



def load_ripple_bench_data(json_path: str) -> Dict[str, Any]:
    """Load Ripple Bench dataset from JSON file."""
    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Print dataset statistics
    print(f"\nDataset statistics:")
    print(f"  - Total topics: {len(data['topics'])}")
    print(f"  - Total questions: {sum(len(t['questions']) for t in data['topics'])}")
    print(f"  - Topics per distance level:")
    
    distance_counts = {}
    for topic in data['topics']:
        dist = topic['distance']
        distance_counts[dist] = distance_counts.get(dist, 0) + 1
    
    for dist in sorted(distance_counts.keys()):
        print(f"    Distance {dist}: {distance_counts[dist]} topics")
    
    return data


def convert_to_hf_format(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert Ripple Bench format to flat format for Hugging Face datasets."""
    rows = []

    all_topics = {}

    for topic_dict in data['topics']:
        topic_name = topic_dict['topic']
        og_topic = topic_dict["original_topic"]
        og_distance = topic_dict["distance"]

        og_topic_dict = {
            "topic": og_topic, 
            "distance": og_distance,
        }

        if topic_name in all_topics:
            all_topics[topic_name]["original_topics"].append(og_topic_dict)
        else:
            topic_dict = topic_dict.copy()
            topic_dict.pop("original_topic")
            topic_dict.pop("distance")
            topic_dict["original_topics"] = [og_topic_dict]
            
            all_topics[topic_name] = topic_dict
    
    for topic in all_topics.values():
        topic_name = topic['topic']
        original_topics = topic['original_topics'],
        facts = topic.get('facts', [])
        
        for question in topic['questions']:
            row = {
                'question': question['question'],
                'choices': question['choices'],
                'answer': question['answer'],
                'topic': topic_name,
                'original_topics': original_topics,
                'facts': facts,
                'question_type': question.get('type', 'multiple_choice'),
                'difficulty': question.get('difficulty', 'medium'),
            }
            rows.append(row)
    
    return rows


def create_dataset_card(data: Dict[str, Any], dataset_path: str) -> str:
    """Create a README.md dataset card for Hugging Face."""
    
    # Calculate statistics
    total_topics = len(data['topics'])
    total_questions = sum(len(t['questions']) for t in data['topics'])
    
    distance_counts = {}
    for topic in data['topics']:
        dist = topic['distance']
        distance_counts[dist] = distance_counts.get(dist, 0) + 1
    
    card_content = f"""---
language:
- en
license: mit
task_categories:
- question-answering
- text-classification
pretty_name: Ripple Bench
tags:
- unlearning
- knowledge-graphs
- evaluation
- safety
size_categories:
- 1K<n<10K
---

# Ripple Bench: Measuring Knowledge Ripple Effects in Language Model Unlearning

## Dataset Description

Ripple Bench is a benchmark for measuring how knowledge changes propagate through related concepts when unlearning specific information from language models.

### Dataset Summary

When we unlearn specific knowledge from a language model (e.g., information about biological weapons), how does this affect the model's knowledge of related topics? Ripple Bench quantifies these "ripple effects" by:

1. Starting with questions from WMDP (Weapons of Mass Destruction Proxy)
2. Extracting core topics and finding semantically related topics
3. Generating new questions about these related topics
4. Evaluating how model performance degrades with semantic distance from the unlearned concept

### Dataset Statistics

- **Total questions**: {total_questions}
- **Total topics**: {total_topics}
- **Topics by distance**:
"""
    
    for dist in sorted(distance_counts.keys()):
        card_content += f"  - Distance {dist}: {distance_counts[dist]} topics\n"
    
    card_content += f"""

### Dataset Structure

Each example contains:
- `question`: The evaluation question
- `choices`: List of 4 multiple choice options
- `answer`: Index of the correct answer (0-3)
- `topic`: The topic being evaluated
- `distance`: Semantic distance from the original unlearned topic (0 = original, higher = more distant)
- `original_topic`: The original WMDP topic this relates to
- `facts`: List of facts extracted from Wikipedia about this topic
- `question_type`: Type of question (default: "multiple_choice")
- `difficulty`: Difficulty level (default: "medium")

### Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("royrin/ripple-bench")

# Access the data
for example in dataset['train']:
    print(f"Question: {{example['question']}}")
    print(f"Topic: {{example['topic']}} (distance {{example['distance']}})")
    print(f"Choices: {{example['choices']}}")
    print(f"Answer: {{example['choices'][example['answer']]}}")
```

### Source Data

This dataset is generated from:
- WMDP (Weapons of Mass Destruction Proxy) questions as seed topics
- Wikipedia articles for finding related topics and extracting facts
- LLM-generated questions based on the extracted facts

### Citation

If you use Ripple Bench in your research, please cite:

```bibtex
@dataset{{ripple_bench_2024,
  title={{Ripple Bench: Measuring Knowledge Ripple Effects in Language Model Unlearning}},
  author={{Roy Rinberg}},
  year={{2024}},
  url={{https://huggingface.co/datasets/royrin/ripple-bench}}
}}
```

### Dataset Creation

Generated on: {datetime.now().strftime("%Y-%m-%d")}
Source: {dataset_path}
"""
    
    return card_content


def upload_to_huggingface(
    dataset_path: str,
    repo_id: str = "royrin/ripple-bench",
    private: bool = False,
    create_pr: bool = False
):
    """Upload dataset to Hugging Face Hub."""
    
    # Load the data
    data = load_ripple_bench_data(dataset_path)
    
    # Convert to Hugging Face format
    print("\nConverting to Hugging Face format...")
    rows = convert_to_hf_format(data)
    
    # Create dataset
    dataset = Dataset.from_list(rows)
    
    # Create a DatasetDict with single split
    dataset_dict = DatasetDict({"train": dataset})
    
    print(f"\nDataset ready for upload:")
    print(f"  - Format: {dataset}")
    print(f"  - Repository: {repo_id}")
    
    # Confirm upload
    if not create_pr:
        response = input("\nProceed with upload? (y/N): ")
        if response.lower() != 'y':
            print("Upload cancelled.")
            return
    
    # Create repository if it doesn't exist
    api = HfApi()
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
    except Exception as e:
        print(f"Note: {e}")
    
    # Push to hub
    print(f"\nUploading to {repo_id}...")
    dataset_dict.push_to_hub(
        repo_id,
        private=private,
        create_pr=create_pr
    )
    
    # Create and upload dataset card
    card_content = create_dataset_card(data, dataset_path)
    card_path = Path("README.md")
    card_path.write_text(card_content)
    
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        create_pr=create_pr
    )
    
    # Clean up
    card_path.unlink()
    
    print(f"\n✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")
    
    if create_pr:
        print("Created pull request for review.")
        
def main():
    parser = argparse.ArgumentParser(
        description="Upload Ripple Bench dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the Ripple Bench JSON file"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="royrin/ripple-bench",
        help="Hugging Face repository ID (default: royrin/ripple-bench)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private"
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request instead of pushing directly"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.dataset_path).exists():
        print(f"Error: Dataset file not found: {args.dataset_path}")
        sys.exit(1)
    
    # Check if logged in to Hugging Face
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"Logged in as: {user_info['name']}")
    except Exception:
        print("Error: Not logged in to Hugging Face")
        print("Please run: huggingface-cli login")
        sys.exit(1)
    
    upload_to_huggingface(
        args.dataset_path,
        args.repo_id,
        args.private,
        args.create_pr
    )


if __name__ == "__main__":
    main()