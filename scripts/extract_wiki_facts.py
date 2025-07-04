#!/usr/bin/env python3
"""
Extract Facts from Wikipedia Articles

This script extracts facts from Wikipedia articles for given topics.
It can use either a local model or LLM API calls.

Usage:
    python extract_wiki_facts.py --topics-file <path> --output-dir <dir>
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
import wikipedia
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ripple_bench.openai_utils import huit_OAI_function
from ripple_bench.anthropic_utils import anthropic_function
from ripple_bench.utils import save_dict, read_dict
from ripple_bench.models import load_zephyr
from ripple_bench.generate_ripple_questions import extract_bulleted_facts


class WikiFactExtractor:

    def __init__(self, output_dir: str, llm_provider: str = "anthropic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.llm_provider = llm_provider

        # Set LLM function
        if llm_provider == "anthropic":
            self.llm_function = anthropic_function
        elif llm_provider == "openai":
            self.llm_function = huit_OAI_function
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")

    def load_topics(self, topics_file: str) -> Dict[str, List[str]]:
        """Load topics and neighbors from file"""
        print(f"Loading topics from {topics_file}")
        data = read_dict(topics_file)

        # Handle different file formats
        if 'topic_to_neighbors' in data:
            return data['topic_to_neighbors']
        elif 'topics_df' in data:
            # Extract unique topics from dataframe format
            topics = list(set([item['topic'] for item in data['topics_df']]))
            return {topic: [] for topic in topics}
        else:
            # Assume it's already in the right format
            return data

    def extract_facts_via_api(self, content: str, topic: str) -> str:
        """Extract facts using LLM API"""
        # Check if content is substantial enough
        if len(content.strip()) < 50:
            return f"• Content too short to extract meaningful facts about {topic}"

        prompt = f'''Extract key facts from the following Wikipedia article about {topic}. 
        
Please provide a bulleted list of the most important facts (aim for 5-10 facts).
Each fact should be:
- Concise and self-contained
- Factual and verifiable
- Relevant to understanding the topic

Article content:
{content}

Please format your response as a bulleted list using "•" symbols.'''

        try:
            print(
                f"Extracting facts for `{topic}` (content length: {len(content)} chars)"
            )
            response = self.llm_function(prompt,
                                         temperature=0.3,
                                         max_tokens=1024)

            if response and len(response.strip()) > 10:
                # Ensure the response contains bullet points
                facts = response.strip()
                if "•" not in facts and "-" in facts:
                    # Convert dash bullets to bullet points
                    facts = facts.replace("\n-", "\n•")
                return facts
            else:
                print(f"Empty or short response for {topic}: '{response}'")
                return f"• Unable to extract facts for {topic} - API response was empty or too short"
        except Exception as e:
            print(f"Error calling LLM API for `{topic}`: {e}")
            return f"• Error extracting facts for `{topic}`: {str(e)}"

    def extract_facts_from_topic(self,
                                 topic: str,
                                 use_local_model: bool = False,
                                 model=None,
                                 tokenizer=None,
                                 max_content_length: int = 3000) -> Dict:
        """Extract facts from a single topic"""
        try:
            # Get Wikipedia page
            page = wikipedia.page(topic)
            content = page.content[:max_content_length]

            # Extract facts
            if use_local_model:
                if model is None or tokenizer is None:
                    raise ValueError(
                        "Model and tokenizer required for local extraction")
                bullet_lines = extract_bulleted_facts(content,
                                                      model,
                                                      tokenizer,
                                                      max_new_tokens=350)
                facts = "\n".join(
                    [f"• {line.lstrip('-').strip()}" for line in bullet_lines])
            else:
                facts = self.extract_facts_via_api(content, topic)

            return {
                'facts': facts,
                'url': page.url,
                'title': page.title,
                'success': True
            }

        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first option
            try:
                page = wikipedia.page(e.options[0])
                content = page.content[:max_content_length]

                if use_local_model:
                    bullet_lines = extract_bulleted_facts(content,
                                                          model,
                                                          tokenizer,
                                                          max_new_tokens=350)
                    facts = "\n".join([
                        f"• {line.lstrip('-').strip()}"
                        for line in bullet_lines
                    ])
                else:
                    facts = self.extract_facts_via_api(content, e.options[0])

                return {
                    'facts': facts,
                    'url': page.url,
                    'title': page.title,
                    'success': True
                }
            except Exception:
                return {
                    'facts': f"No facts available for {topic}",
                    'url': None,
                    'title': topic,
                    'success': False
                }

        except Exception as e:
            print(f"Error processing {topic}: {e}")
            return {
                'facts': f"No facts available for {topic}",
                'url': None,
                'title': topic,
                'success': False
            }

    def process(self,
                topics_file: str,
                use_local_model: bool = False,
                max_content_length: int = 3000,
                save_interval: int = 10):
        """Main processing pipeline"""
        # Load topics
        topic_to_neighbors = self.load_topics(topics_file)

        # Collect all topics (original + neighbors)
        all_topics = set()
        for topic, neighbors in topic_to_neighbors.items():
            all_topics.add(topic)
            all_topics.update(neighbors)

        print(f"Extracting facts for {len(all_topics)} topics...")

        # Load model if using local
        model, tokenizer = None, None
        if use_local_model:
            print("Loading Zephyr model for fact extraction...")
            model, tokenizer = load_zephyr()
        else:
            print(f"Using {self.llm_provider} API for fact extraction...")

        # Extract facts
        facts_dict = {}
        for i, topic in enumerate(tqdm(all_topics)):
            fact_data = self.extract_facts_from_topic(topic, use_local_model,
                                                      model, tokenizer,
                                                      max_content_length)
            facts_dict[topic] = fact_data

            # Save intermediate results
            if (i + 1) % save_interval == 0:
                temp_file = self.output_dir / f"wiki_facts_temp_{self.timestamp}.json"
                save_dict(facts_dict, temp_file)

        # Save final results
        output_file = self.output_dir / f"wiki_facts_{self.timestamp}.json"
        save_dict(facts_dict, output_file)
        print(f"Saved facts to {output_file}")

        # Create summary
        successful_extractions = sum(1 for f in facts_dict.values()
                                     if f.get('success', False))
        summary = {
            'metadata': {
                'timestamp':
                self.timestamp,
                'topics_source':
                topics_file,
                'total_topics':
                len(all_topics),
                'successful_extractions':
                successful_extractions,
                'failed_extractions':
                len(all_topics) - successful_extractions,
                'use_local_model':
                use_local_model,
                'llm_provider':
                self.llm_provider if not use_local_model else 'zephyr-7b-beta'
            },
            'facts': facts_dict
        }

        summary_file = self.output_dir / f"wiki_facts_summary_{self.timestamp}.json"
        save_dict(summary, summary_file)
        print(f"\nSaved summary to {summary_file}")
        print(
            f"Successfully extracted facts from {successful_extractions}/{len(all_topics)} topics"
        )

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract facts from Wikipedia articles")
    parser.add_argument(
        "--topics-file",
        required=True,
        help="Path to topics JSON file (from extract_topics_and_neighbors.py)")
    parser.add_argument("--output-dir",
                        default="ripple_bench_data/wiki_facts",
                        help="Output directory")
    parser.add_argument(
        "--use-local-model",
        action="store_true",
        default=True,
        help="Use local Zephyr model instead of API (default: True)")
    parser.add_argument("--use-api",
                        action="store_true",
                        help="Use API instead of local model")
    parser.add_argument("--llm-provider",
                        default="anthropic",
                        choices=["anthropic", "openai"],
                        help="LLM provider to use when using API")
    parser.add_argument("--max-content-length",
                        type=int,
                        default=3000,
                        help="Maximum Wikipedia content length to process")
    parser.add_argument("--save-interval",
                        type=int,
                        default=10,
                        help="Save intermediate results every N topics")

    args = parser.parse_args()

    # Default to local model unless --use-api is specified
    use_local = not args.use_api

    extractor = WikiFactExtractor(args.output_dir, args.llm_provider)
    extractor.process(args.topics_file, use_local, args.max_content_length,
                      args.save_interval)


if __name__ == "__main__":
    main()
