#!/usr/bin/env python3
"""
Download Ripple Bench dataset from Hugging Face Hub

This script downloads a Ripple Bench dataset that was previously uploaded to:
https://huggingface.co/datasets/royrin/ripple-bench
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def download_ripple_bench_dataset(
    repo_id: str = "royrin/ripple-bench",
    output_dir: str = "data/ripple_bench_datasets",
    dataset_name: str = None,
    force: bool = False
):
    """Download Ripple Bench dataset from Hugging Face Hub."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # If no specific name, use default with timestamp
    if dataset_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_name = f"ripple_bench_dataset_{timestamp}.json"
    
    output_file = output_path / dataset_name
    
    # Check if already exists
    if output_file.exists() and not force:
        print(f"Dataset already exists at: {output_file}")
        print("Use --force to re-download")
        return output_file
    
    try:
        print(f"Downloading Ripple Bench dataset from {repo_id}...")
        
        # Load the dataset
        dataset = load_dataset(repo_id, split="train")
        
        print(f"Downloaded {len(dataset)} questions")
        
        # Convert to the expected Ripple Bench format
        print("Converting to Ripple Bench format...")
        
        # Group questions by topic
        topics_dict = {}
        for example in dataset:
            topic = example['topic']
            if topic not in topics_dict:
                topics_dict[topic] = {
                    'topic': topic,
                    'distance': example['distance'],
                    'original_topic': example.get('original_topic', topic),
                    'facts': example.get('facts', []),
                    'wiki_url': example.get('wiki_url', ''),
                    'questions': []
                }
            
            # Add question to topic
            question = {
                'question': example['question'],
                'choices': example['choices'],
                'answer': example['answer']
            }
            topics_dict[topic]['questions'].append(question)
        
        # Convert to list and sort by distance
        topics_list = list(topics_dict.values())
        topics_list.sort(key=lambda x: (x['distance'], x['topic']))
        
        # Get metadata from first example
        first_example = dataset[0] if len(dataset) > 0 else {}
        
        # Create the full dataset structure
        ripple_dataset = {
            'metadata': {
                'source': f'huggingface:{repo_id}',
                'download_timestamp': datetime.now().isoformat(),
                'num_topics': len(topics_list),
                'num_questions': len(dataset),
                'distance_distribution': {}
            },
            'topics': topics_list
        }
        
        # Calculate distance distribution
        distance_counts = {}
        for topic in topics_list:
            dist = topic['distance']
            distance_counts[dist] = distance_counts.get(dist, 0) + 1
        ripple_dataset['metadata']['distance_distribution'] = distance_counts
        
        # Save the dataset
        print(f"Saving to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(ripple_dataset, f, indent=2)
        
        # Print summary
        print(f"\n✅ Successfully downloaded Ripple Bench dataset!")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Dataset statistics:")
        print(f"   - Topics: {len(topics_list)}")
        print(f"   - Questions: {len(dataset)}")
        print(f"   - Distance distribution:")
        for dist in sorted(distance_counts.keys()):
            print(f"     Distance {dist}: {distance_counts[dist]} topics")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check if the dataset exists at https://huggingface.co/datasets/{repo_id}")
        print("2. Make sure you're logged in if it's a private dataset: huggingface-cli login")
        print("3. Check your internet connection")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download Ripple Bench dataset from Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="royrin/ripple-bench",
        help="Hugging Face repository ID (default: royrin/ripple-bench)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ripple_bench_datasets",
        help="Output directory for downloaded dataset"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Output filename (default: ripple_bench_dataset_TIMESTAMP.json)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists"
    )
    
    args = parser.parse_args()
    
    # Check if logged in to Hugging Face (for private datasets)
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"Logged in as: {user_info['name']}")
    except Exception:
        print("Note: Not logged in to Hugging Face (required for private datasets)")
        print("To login: huggingface-cli login")
    
    output_file = download_ripple_bench_dataset(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        dataset_name=args.output_name,
        force=args.force
    )
    
    print(f"\nYou can now evaluate models with:")
    print(f"python scripts/evaluate_ripple_bench.py {output_file} \\")
    print(f"    --base-model /path/to/base/model \\")
    print(f"    --unlearned-model /path/to/unlearned/model")


if __name__ == "__main__":
    main()