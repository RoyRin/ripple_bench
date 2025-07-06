#!/usr/bin/env python3
"""
Example of parallel fact extraction for build_ripple_bench_from_wmdp.py
"""

import concurrent.futures
from typing import Dict, List, Tuple
import time
from tqdm import tqdm


def extract_facts_from_topics_parallel(
    self,
    topic_to_neighbors: Dict[str, List[str]],
    cache_file: str = None,
    use_local_model: bool = False,
    use_local_wikipedia: bool = False,
    max_workers: int = 10  # Number of parallel API calls
) -> Dict[str, Dict[str, str]]:
    """Extract facts from Wikipedia articles with parallel API calls."""

    # ... initialization code same as before ...

    def process_single_topic(topic: str) -> Tuple[str, Dict]:
        """Process a single topic - this runs in parallel"""
        # Skip unknown topics
        if topic.lower() in ["unknown topic", "unknown", ""]:
            return topic, {
                'facts': f"Skipped: Invalid topic '{topic}'",
                'url': None,
                'title': topic
            }

        try:
            # Get Wikipedia content (fast with local Wikipedia)
            if use_local_wikipedia:
                article = local_wiki.get_article(topic)
                if article:
                    content = article.get('text', '')[:self.wiki_content_size]
                    page_title = article.get('title', topic)
                    page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
                else:
                    return topic, {
                        'facts':
                        f"Article not found in local Wikipedia: {topic}",
                        'url': None,
                        'title': topic
                    }
            else:
                # Online Wikipedia access
                page = wikipedia.page(topic)
                content = page.content[:self.wiki_content_size]
                page_title = page.title
                page_url = page.url

            # Skip if content is too short
            if len(content) < 100:
                return topic, {
                    'facts': f"No substantial content found for {topic}",
                    'url': page_url,
                    'title': page_title
                }

            # Extract facts (this is the slow part we parallelize)
            if use_local_model:
                facts = extract_bulleted_facts(content,
                                               model,
                                               tokenizer,
                                               max_new_tokens=350)
            else:
                facts = self._extract_facts_via_api(content, topic)

            return topic, {
                'facts': facts,
                'url': page_url,
                'title': page_title
            }

        except Exception as e:
            return topic, {
                'facts': f"No facts available for {topic}",
                'url': None,
                'title': topic
            }

    # Process topics in parallel
    remaining_topics = [t for t in all_topics if t not in processed_topics]

    print(
        f"Extracting facts for {len(remaining_topics)} topics using {max_workers} parallel workers"
    )

    # Use ThreadPoolExecutor for I/O-bound tasks (API calls)
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_topic = {
            executor.submit(process_single_topic, topic): topic
            for topic in remaining_topics
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(remaining_topics)) as pbar:
            for future in concurrent.futures.as_completed(future_to_topic):
                topic, result = future.result()
                facts_dict[topic] = result
                pbar.update(1)

                # Save intermediate results periodically
                if len(facts_dict) % 20 == 0:
                    save_dict(facts_dict, temp_file)

    # Save final results
    save_dict(facts_dict, final_file)

    return facts_dict
