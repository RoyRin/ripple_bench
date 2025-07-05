#!/usr/bin/env python3
"""
Modified version of build_ripple_bench_from_wmdp.py that uses local Wikipedia data.
This demonstrates how to integrate local Wikipedia access for faster fact extraction.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the original builder
from scripts.build_ripple_bench_from_wmdp import RippleBenchBuilder

# Import local wiki helper
from scripts.local_wiki_helper import LocalWikipediaReader


class RippleBenchBuilderWithLocalWiki(RippleBenchBuilder):
    """Extended RippleBenchBuilder that uses local Wikipedia data."""

    def __init__(
            self,
            *args,
            wiki_json_path: str = "/Users/roy/data/wikipedia/wikipedia/json",
            **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize local Wikipedia reader
        self.wiki_json_path = Path(wiki_json_path)
        self.local_wiki = None

        if self.wiki_json_path.exists():
            print(f"📚 Local Wikipedia data found at: {self.wiki_json_path}")
            print("⚡ Will use local data for faster fact extraction!")
            try:
                self.local_wiki = LocalWikipediaReader(str(
                    self.wiki_json_path))
                # Don't initialize yet - do it lazily when needed
            except Exception as e:
                print(
                    f"⚠️  Warning: Could not initialize local Wikipedia reader: {e}"
                )
                print("   Falling back to online Wikipedia access")
                self.local_wiki = None
        else:
            print(
                f"⚠️  Local Wikipedia data not found at: {self.wiki_json_path}"
            )
            print("   Using online Wikipedia access (slower)")

    def _get_wikipedia_content(self, topic: str) -> tuple[str, str, str]:
        """
        Get Wikipedia content, preferring local data over online.
        
        Returns:
            Tuple of (content, url, title)
        """
        # Try local Wikipedia first
        if self.local_wiki:
            try:
                # Initialize on first use
                if self.local_wiki.title_to_file_path is None:
                    print("📖 Building Wikipedia index (one-time operation)...")
                    self.local_wiki.initialize()

                page_data = self.local_wiki.get_page(topic)
                if page_data:
                    return (page_data['text'][:3000], page_data['url'],
                            page_data['title'])
            except Exception as e:
                print(f"⚠️  Local Wikipedia lookup failed for '{topic}': {e}")

        # Fall back to online Wikipedia
        import wikipedia
        try:
            page = wikipedia.page(topic)
            return (page.content[:3000], page.url, page.title)
        except wikipedia.exceptions.DisambiguationError as e:
            # Try first option
            page = wikipedia.page(e.options[0])
            return (page.content[:3000], page.url, page.title)

    def extract_facts_from_topics(self,
                                  topic_to_neighbors,
                                  cache_file=None,
                                  use_local_model=False):
        """Override to use local Wikipedia data."""
        print("Extracting facts from Wikipedia articles...")

        # Check for existing results first
        final_file = self.facts_dir / "wiki_facts.json"
        if final_file.exists():
            print(f"Found existing facts at {final_file}")
            facts = self.read_dict(final_file)
            print(f"  Loaded facts for {len(facts)} topics")
            return facts

        # Check for cached results
        if cache_file and Path(cache_file).exists():
            print(f"Loading cached facts from {cache_file}")
            return self.read_dict(cache_file)

        # Load model for fact extraction if using local model
        if use_local_model:
            print("Loading Zephyr model for fact extraction...")
            from ripple_bench.models import load_zephyr
            model, tokenizer = load_zephyr()
        else:
            print(f"Using {self.llm_provider} API for fact extraction...")

        # Import for saving
        from ripple_bench.utils import save_dict

        # Check for temporary file to resume from
        temp_file = self.facts_dir / "wiki_facts_temp.json"
        if temp_file.exists():
            print(f"Resuming from temporary file: {temp_file}")
            facts_dict = self.read_dict(temp_file)
            processed_topics = set(facts_dict.keys())
        else:
            facts_dict = {}
            processed_topics = set()

        all_topics = set()

        # Collect all topics (original + neighbors)
        for topic, neighbors in topic_to_neighbors.items():
            all_topics.add(topic)
            all_topics.update(neighbors)

        remaining_topics = [t for t in all_topics if t not in processed_topics]

        print(
            f"Extracting facts for {len(remaining_topics)} remaining topics (already processed: {len(processed_topics)})"
        )

        from tqdm import tqdm

        for topic in tqdm(remaining_topics):
            # Skip unknown topics
            if topic.lower() in ["unknown topic", "unknown", ""]:
                facts_dict[topic] = {
                    'facts': f"Skipped: Invalid topic '{topic}'",
                    'url': None,
                    'title': topic
                }
                continue

            try:
                # Get Wikipedia content (local or online)
                content, url, title = self._get_wikipedia_content(topic)

                # Skip if content is too short
                if len(content) < 100:
                    facts_dict[topic] = {
                        'facts': f"No substantial content found for {topic}",
                        'url': url,
                        'title': title
                    }
                    continue

                # Extract facts using model or API
                if use_local_model:
                    from ripple_bench.generate_ripple_questions import extract_bulleted_facts
                    facts = extract_bulleted_facts(content,
                                                   model,
                                                   tokenizer,
                                                   max_new_tokens=350)
                else:
                    facts = self._extract_facts_via_api(content, topic)

                facts_dict[topic] = {
                    'facts': facts,
                    'url': url,
                    'title': title
                }

            except Exception as e:
                print(f"Error processing {topic}: {e}")
                facts_dict[topic] = {
                    'facts': f"No facts available for {topic}",
                    'url': None,
                    'title': topic
                }

            # Save intermediate results
            if len(facts_dict) % 5 == 0:
                save_dict(facts_dict, temp_file)

        # Save final results
        save_dict(facts_dict, final_file)

        # Remove temp file
        if temp_file.exists():
            temp_file.unlink()

        return facts_dict

    def read_dict(self, filepath):
        """Helper to read dictionary from file."""
        from ripple_bench.utils import read_dict
        return read_dict(filepath)


def main():
    """Test the local Wikipedia integration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build Ripple Bench with local Wikipedia")
    parser.add_argument("--test",
                        action="store_true",
                        help="Run in test mode with small sample")
    parser.add_argument("--wiki-path",
                        default="/Users/roy/data/wikipedia/wikipedia/json",
                        help="Path to local Wikipedia JSON data")
    args = parser.parse_args()

    if args.test:
        print("🧪 Running in test mode...")

        # Create builder
        builder = RippleBenchBuilderWithLocalWiki(
            output_dir="test_ripple_bench_local",
            wiki_json_path=args.wiki_path)

        # Test with a small set of topics
        test_topics = {
            "DNA": ["RNA", "Protein"],
            "CRISPR": ["Gene editing", "Genetics"]
        }

        print("\n📊 Testing fact extraction with local Wikipedia...")
        facts = builder.extract_facts_from_topics(test_topics,
                                                  use_local_model=False)

        print(f"\n✅ Extracted facts for {len(facts)} topics:")
        for topic, fact_data in list(facts.items())[:3]:
            print(f"\n📄 {topic}:")
            print(f"   URL: {fact_data.get('url', 'N/A')}")
            facts_text = fact_data.get('facts', '')[:200]
            print(f"   Facts preview: {facts_text}...")
    else:
        print("💡 This script demonstrates local Wikipedia integration.")
        print("   Use --test to run a quick test.")
        print(
            "   For full dataset building, use the original script with these modifications."
        )


if __name__ == "__main__":
    main()
