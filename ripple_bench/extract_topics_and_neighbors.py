#!/usr/bin/env python3
"""
Extract Topics and Neighbors from WMDP Questions

This script:
1. Extracts topics from WMDP questions using LLM
2. Finds K nearest neighbor topics using RAG/FAISS

Usage:
    python extract_topics_and_neighbors.py --wmdp-path <path> --output-dir <dir>
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
import os
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ripple_bench.openai_utils import huit_OAI_function
from ripple_bench.anthropic_utils import anthropic_function
from ripple_bench.utils import save_dict, read_dict
from ripple_bench.models import load_zephyr, generate_text
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


class PromptedBGE(HuggingFaceEmbeddings):
    """BGE embeddings with retrieval prompts"""

    def embed_documents(self, texts):
        return super().embed_documents(
            [f"Represent this document for retrieval: {t}" for t in texts])

    def embed_query(self, text):
        return super().embed_query(
            f"Represent this query for retrieval: {text}")


class TopicNeighborExtractor:

    def __init__(self,
                 output_dir: str,
                 llm_provider: str = "local",
                 use_local_model: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.llm_provider = llm_provider
        self.use_local_model = use_local_model
        self.model = None
        self.tokenizer = None

        # Set LLM function
        if use_local_model or llm_provider == "local":
            print("Using local model for topic extraction")
            self.use_local_model = True
        elif llm_provider == "anthropic":
            self.llm_function = anthropic_function
        elif llm_provider == "openai":
            self.llm_function = huit_OAI_function
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")

    def load_wmdp_questions(self, wmdp_path: str) -> List[Dict]:
        """Load WMDP questions from JSON file"""
        print(f"Loading WMDP questions from {wmdp_path}")
        with open(wmdp_path, 'r') as f:
            data = json.load(f)
        return data

    def extract_topic(self, question: str) -> str:
        """Extract topic from a single question"""
        prompt = f'''Given the following multiple choice question, extract the core wikipedia-style topic that it's primarily testing.

Question: {question}

What is the main Wikipedia-style topic (usually 1-3 words) that this question is testing? 

Important:
- Give a specific, searchable Wikipedia topic name
- If the question is about a specific concept, chemical, biological process, etc., use that as the topic
- Avoid generic terms like "Unknown" or "General"
- Just give the topic name, nothing else

Topic:'''

        if self.use_local_model:
            # Load model if not already loaded
            if self.model is None:
                print("Loading Zephyr model for topic extraction...")
                self.model, self.tokenizer = load_zephyr()

            # Generate with local model
            full_prompt = f"<|system|>\nYou are a helpful AI assistant.\n<|user|>\n{prompt}\n<|assistant|>\n"
            response = generate_text(full_prompt,
                                     model=self.model,
                                     tokenizer=self.tokenizer,
                                     temperature=0.3,
                                     max_new_tokens=50,
                                     do_sample=True)
            # Extract just the topic from the response
            response = response.replace(full_prompt, "").strip()
        else:
            response = self.llm_function(prompt, temperature=0.3)

        if response and response.strip() and response.strip().lower() not in [
                "unknown", "unknown topic", "n/a", "none"
        ]:
            # Clean up the response - take only the first line if multiple lines
            topic = response.strip().split('\n')[0].strip()
            # Remove any trailing punctuation
            topic = topic.rstrip('.,:;!?')
            return topic

        # Try to extract a topic from the question itself as fallback
        # Look for capitalized words or technical terms
        words = question.split()
        for word in words:
            if word[0].isupper() and len(word) > 3 and word not in [
                    'What', 'Which', 'How', 'When', 'Where', 'Why', 'The'
            ]:
                return word

        return "General Knowledge"

    def extract_topics_from_questions(self,
                                      questions: List[Dict],
                                      num_samples: int = None) -> pd.DataFrame:
        """Extract topics from WMDP questions"""
        print("Extracting topics from questions...")

        if num_samples:
            questions = questions[:num_samples]

        topics = []
        for i, q in enumerate(tqdm(questions)):
            question_text = q['question']
            topic = self.extract_topic(question_text)
            topics.append({
                'question': question_text,
                'topic': topic,
                'answer': q['answer'],
                'choices': q['choices'],
                'original_index': i
            })

        df = pd.DataFrame(topics)

        # Save results
        output_file = self.output_dir / f"wmdp_topics_{self.timestamp}.json"
        save_dict(topics, output_file)
        print(f"Saved topics to {output_file}")

        return df

    def get_RAG(self, faiss_path: str = None):
        """Initialize RAG system"""
        if faiss_path is None:
            faiss_path = os.environ.get('WIKI_FAISS_PATH')
            if not faiss_path:
                raise ValueError(
                    "Please set WIKI_FAISS_PATH environment variable or provide faiss_path"
                )

        faiss_path = Path(faiss_path)
        print(f"Loading vectorstore from {faiss_path}")

        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")

        embedding_model = PromptedBGE(model_name="BAAI/bge-base-en")

        vectorstore = FAISS.load_local(str(faiss_path),
                                       embedding_model,
                                       allow_dangerous_deserialization=True)

        # Load title mapping
        index_file = faiss_path / "index.pkl"
        if index_file.exists():
            import pickle
            with open(index_file, 'rb') as f:
                title_to_path = pickle.load(f)
        else:
            title_to_path = {}

        return vectorstore, title_to_path

    def find_topic_neighbors(self,
                             topics: List[str],
                             k_neighbors: int = 5,
                             faiss_path: str = None) -> Dict[str, List[str]]:
        """Find K nearest neighbors for each topic using RAG"""
        print("Finding topic neighbors using RAG...")

        # Initialize RAG
        vectorstore, title_to_path = self.get_RAG(faiss_path)

        topic_to_neighbors = {}
        unique_topics = list(set(topics))

        print(f"Processing {len(unique_topics)} unique topics...")
        for topic in tqdm(unique_topics):
            # Search for similar topics
            similar_docs = vectorstore.similarity_search(topic,
                                                         k=k_neighbors + 1)

            # Extract neighbor topics
            neighbors = []
            for doc in similar_docs:
                neighbor_topic = doc.metadata.get(
                    'title',
                    doc.page_content.split('\n')[0])
                if neighbor_topic != topic and neighbor_topic not in neighbors:
                    neighbors.append(neighbor_topic)

            topic_to_neighbors[topic] = neighbors[:k_neighbors]

        # Save results
        output_file = self.output_dir / f"topic_neighbors_{self.timestamp}.json"
        save_dict(topic_to_neighbors, output_file)
        print(f"Saved topic neighbors to {output_file}")

        return topic_to_neighbors

    def process(self,
                wmdp_path: str,
                num_samples: int = None,
                k_neighbors: int = 5,
                faiss_path: str = None):
        """Main processing pipeline"""
        # Load questions
        questions = self.load_wmdp_questions(wmdp_path)
        print(f"Loaded {len(questions)} questions")

        # Extract topics
        topics_df = self.extract_topics_from_questions(questions, num_samples)
        print(f"Extracted {len(topics_df)} topics")

        # Find neighbors
        unique_topics = topics_df['topic'].unique().tolist()
        topic_neighbors = self.find_topic_neighbors(unique_topics, k_neighbors,
                                                    faiss_path)

        # Create summary
        summary = {
            'metadata': {
                'timestamp': self.timestamp,
                'wmdp_source': wmdp_path,
                'num_questions': len(questions),
                'num_samples_used': num_samples or len(questions),
                'num_unique_topics': len(unique_topics),
                'k_neighbors': k_neighbors,
                'llm_provider':
                'local' if self.use_local_model else self.llm_provider,
                'use_local_model': self.use_local_model
            },
            'topics_df': topics_df.to_dict('records'),
            'topic_to_neighbors': topic_neighbors
        }

        # Save summary
        summary_file = self.output_dir / f"topics_and_neighbors_summary_{self.timestamp}.json"
        save_dict(summary, summary_file)
        print(f"\nSaved summary to {summary_file}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract topics and neighbors from WMDP questions")
    parser.add_argument("--wmdp-path",
                        required=True,
                        help="Path to WMDP JSON file")
    parser.add_argument("--output-dir",
                        default="ripple_bench_data/topics_neighbors",
                        help="Output directory")
    parser.add_argument("--num-samples",
                        type=int,
                        help="Number of questions to process")
    parser.add_argument("--k-neighbors",
                        type=int,
                        default=5,
                        help="Number of neighbor topics to find")
    parser.add_argument(
        "--faiss-path",
        help=
        "Path to FAISS index (uses WIKI_FAISS_PATH env var if not provided)")
    parser.add_argument("--llm-provider",
                        default="local",
                        choices=["local", "anthropic", "openai"],
                        help="LLM provider to use (default: local)")
    parser.add_argument(
        "--use-api",
        action="store_true",
        help=
        "Use API instead of local model (deprecated, use --llm-provider instead)"
    )

    args = parser.parse_args()

    # Handle backwards compatibility with --use-api flag
    use_local = args.llm_provider == "local" and not args.use_api

    extractor = TopicNeighborExtractor(args.output_dir,
                                       args.llm_provider,
                                       use_local_model=use_local)
    extractor.process(args.wmdp_path, args.num_samples, args.k_neighbors,
                      args.faiss_path)


if __name__ == "__main__":
    main()
