#!/usr/bin/env python3
"""
Validate and clean wiki_facts.json, identifying topics with None or empty facts.
"""

import json
from pathlib import Path
import argparse


def validate_wiki_facts(facts_file):
    """Validate wiki facts and report issues."""

    print(f"Loading facts from {facts_file}")
    with open(facts_file, 'r') as f:
        facts_dict = json.load(f)

    print(f"Total topics: {len(facts_dict)}")

    # Analyze facts
    none_facts = []
    empty_facts = []
    valid_facts = []
    short_facts = []
    fact_length_distribution = {
        "0-50": 0,
        "51-100": 0,
        "101-500": 0,
        "501-1000": 0,
        "1000+": 0
    }

    for topic, fact_data in facts_dict.items():
        if fact_data is None:
            none_facts.append(topic)
        elif not isinstance(fact_data, dict):
            none_facts.append(topic)
        elif 'facts' not in fact_data:
            empty_facts.append(topic)
        elif not fact_data['facts'] or fact_data['facts'] is None:
            empty_facts.append(topic)
        else:
            valid_facts.append(topic)

            # Check total fact length
            total_length = sum(
                len(fact) for fact in fact_data['facts'] if fact)
            if total_length < 100:
                short_facts.append((topic, total_length))

            # Track distribution
            if total_length <= 50:
                fact_length_distribution["0-50"] += 1
            elif total_length <= 100:
                fact_length_distribution["51-100"] += 1
            elif total_length <= 500:
                fact_length_distribution["101-500"] += 1
            elif total_length <= 1000:
                fact_length_distribution["501-1000"] += 1
            else:
                fact_length_distribution["1000+"] += 1

    # Report
    print(f"\nAnalysis:")
    print(f"- Valid facts: {len(valid_facts)}")
    print(f"- None values: {len(none_facts)}")
    print(f"- Empty facts: {len(empty_facts)}")
    print(f"- Topics with <100 chars total: {len(short_facts)}")

    print(f"\nFact length distribution:")
    for range_name, count in fact_length_distribution.items():
        print(f"  {range_name} chars: {count} topics")

    if none_facts:
        print(f"\nTopics with None values ({len(none_facts)}):")
        for topic in none_facts[:10]:
            print(f"  - {topic}")
        if len(none_facts) > 10:
            print(f"  ... and {len(none_facts) - 10} more")

    if empty_facts:
        print(f"\nTopics with empty facts ({len(empty_facts)}):")
        for topic in empty_facts[:10]:
            print(f"  - {topic}")
        if len(empty_facts) > 10:
            print(f"  ... and {len(empty_facts) - 10} more")

    if short_facts:
        print(f"\nTopics with <100 chars ({len(short_facts)}):")
        # Sort by length to show shortest first
        short_facts.sort(key=lambda x: x[1])
        for topic, length in short_facts[:10]:
            print(f"  - {topic}: {length} chars")
        if len(short_facts) > 10:
            print(f"  ... and {len(short_facts) - 10} more")

    # Create cleaned version
    if none_facts or empty_facts:
        print("\nCreating cleaned version...")
        cleaned_facts = {
            topic: fact_data
            for topic, fact_data in facts_dict.items() if topic in valid_facts
        }

        output_file = Path(facts_file).parent / "wiki_facts_cleaned.json"
        with open(output_file, 'w') as f:
            json.dump(cleaned_facts, f, indent=2)

        print(f"Cleaned facts saved to: {output_file}")
        print(
            f"Removed {len(none_facts) + len(empty_facts)} problematic entries"
        )

        # Also create a list of topics to regenerate
        topics_to_regenerate = none_facts + empty_facts
        regen_file = Path(facts_file).parent / "topics_to_regenerate.json"
        with open(regen_file, 'w') as f:
            json.dump(topics_to_regenerate, f, indent=2)
        print(f"Topics to regenerate saved to: {regen_file}")

    return valid_facts, none_facts, empty_facts


def main():
    parser = argparse.ArgumentParser(description="Validate wiki facts")
    parser.add_argument("--facts-file",
                        required=True,
                        help="Path to wiki_facts.json")

    args = parser.parse_args()
    validate_wiki_facts(args.facts_file)


if __name__ == "__main__":
    main()
