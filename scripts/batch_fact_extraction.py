#!/usr/bin/env python3
"""
Batch fact extraction - process multiple topics in a single API call
"""


def extract_facts_batch(self,
                        topics_with_content: List[Tuple[str, str]],
                        batch_size: int = 5) -> Dict[str, str]:
    """Extract facts for multiple topics in a single API call"""

    # Build the batch prompt
    prompt = "Extract key facts from the following Wikipedia articles. For each article, provide 5-10 bullet points.\n\n"

    for i, (topic, content) in enumerate(topics_with_content):
        prompt += f"### Article {i+1}: {topic}\n"
        prompt += f"{content[:1000]}...\n\n"  # Limit content per topic to keep prompt manageable

    prompt += """Please format your response as follows:
### Article 1: [Topic Name]
• Fact 1
• Fact 2
• ...

### Article 2: [Topic Name]
• Fact 1
• Fact 2
• ...

And so on for all articles."""

    # Make single API call
    response = self.llm_function(prompt,
                                 model=self.fact_model,
                                 temperature=0.3)

    # Parse response into individual topic facts
    facts_by_topic = {}
    current_topic = None
    current_facts = []

    for line in response.split('\n'):
        if line.startswith('### Article'):
            # Save previous topic's facts
            if current_topic:
                facts_by_topic[current_topic] = '\n'.join(current_facts)
            # Start new topic
            topic_match = line.split(':',
                                     1)[1].strip() if ':' in line else None
            current_topic = topic_match
            current_facts = []
        elif line.strip().startswith('•'):
            current_facts.append(line.strip())

    # Don't forget the last topic
    if current_topic:
        facts_by_topic[current_topic] = '\n'.join(current_facts)

    return facts_by_topic


def process_topics_in_batches(self,
                              all_topics_data: List[Tuple[str, Dict]],
                              batch_size: int = 5):
    """Process all topics in batches"""
    results = {}

    for i in range(0, len(all_topics_data), batch_size):
        batch = all_topics_data[i:i + batch_size]

        # Prepare batch data
        topics_with_content = []
        topic_to_data = {}

        for topic, data in batch:
            if 'content' in data:
                topics_with_content.append((topic, data['content']))
                topic_to_data[topic] = data

        if topics_with_content:
            # Extract facts for entire batch
            batch_facts = extract_facts_batch(self, topics_with_content,
                                              batch_size)

            # Combine with metadata
            for topic, facts in batch_facts.items():
                if topic in topic_to_data:
                    results[topic] = {
                        'facts': facts,
                        'url': topic_to_data[topic].get('url'),
                        'title': topic_to_data[topic].get('title')
                    }

    return results
