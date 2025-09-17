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
    subdir: str = None,
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
        if subdir:
            print(f"Downloading Ripple Bench dataset from {repo_id}/{subdir}...")
            
            # Download dataset files from specific subdirectory
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download train.arrow file
                train_arrow_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{subdir}/train.arrow",
                    repo_type="dataset",
                    cache_dir=str(temp_path)
                )
                
                # Download dataset_info.json
                info_json_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{subdir}/dataset_info.json",
                    repo_type="dataset",
                    cache_dir=str(temp_path)
                )
                
                # Load dataset from downloaded files
                from datasets import Dataset
                dataset = Dataset.from_file(train_arrow_path)
                
        else:
            print(f"Downloading Ripple Bench dataset from {repo_id}...")
            
            # Load the dataset from root
            dataset = load_dataset(repo_id, split="train")
        
        print(f"Downloaded {len(dataset)} questions")
        
        # Convert to the expected Ripple Bench format
        print("Converting to Ripple Bench format...")
        
        # Group questions by topic
        topics_dict = {}
        for example in dataset:
            topic = example['topic']
            
            # Extract distance from original_topics structure
            distance = 0  # Default distance
            original_topic = topic  # Default original topic
            
            if 'original_topics' in example and example['original_topics']:
                try:
                    # original_topics might be a tuple or list
                    original_topics_data = example['original_topics']
                    
                    # If it's a tuple, get the first element (which should be a list)
                    if isinstance(original_topics_data, tuple):
                        original_topics_list = original_topics_data[0]
                    else:
                        original_topics_list = original_topics_data
                    
                    # If we have a list of original topics, get the first one
                    if isinstance(original_topics_list, list) and len(original_topics_list) > 0:
                        first_original = original_topics_list[0]
                        if isinstance(first_original, dict):
                            distance = first_original.get('distance', 0)
                            original_topic = first_original.get('topic', topic)
                except (IndexError, KeyError, TypeError) as e:
                    print(f"Warning: Could not parse original_topics: {e}")
                    print(f"original_topics type: {type(example['original_topics'])}")
                    print(f"original_topics value: {example['original_topics']}")
            
            if topic not in topics_dict:
                topics_dict[topic] = {
                    'topic': topic,
                    'distance': distance,
                    'original_topic': original_topic,
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
        "subdir",
        type=str,
        help="Subdirectory path on Hugging Face (e.g., 'ripple_bench_chem_2025-09-05')"
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
        subdir=args.subdir,
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