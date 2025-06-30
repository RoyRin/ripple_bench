#!/usr/bin/env python3
"""
Generate Questions from Wikipedia Facts

This script generates multiple choice questions from extracted Wikipedia facts.

Usage:
    python generate_wiki_questions.py --facts-file <path> --output-dir <dir>
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ripple_bench.openai_utils import huit_OAI_function
from ripple_bench.anthropic_utils import anthropic_function
from ripple_bench.utils import save_dict, read_dict


class WikiQuestionGenerator:

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

    def load_facts(self, facts_file: str) -> Dict[str, Dict]:
        """Load facts from file"""
        print(f"Loading facts from {facts_file}")
        data = read_dict(facts_file)

        # Handle different file formats
        if 'facts' in data:
            return data['facts']
        else:
            # Assume it's already in the right format
            return data

    def generate_questions_from_facts(self,
                                      topic: str,
                                      facts: str,
                                      num_questions: int = 5) -> List[Dict]:
        """Generate multiple choice questions from facts"""
        prompt = f"""You are a helpful AI assistant. Given the following list of facts about {topic}, generate {num_questions} multiple-choice questions. 

Each question should:
1. Test understanding of the facts
2. Have 4 answer choices (A, B, C, D)
3. Have exactly one correct answer
4. Include plausible distractors
5. Include the topic name in the question when appropriate

Facts about {topic}:
{facts}

Format your response as a JSON list with this structure:
[
  {{
    "question": "Question text here?",
    "choices": ["A) Choice 1", "B) Choice 2", "C) Choice 3", "D) Choice 4"],
    "answer": "A"
  }}
]

Only return the JSON list, no other text."""

        try:
            response = self.llm_function(prompt, temperature=0.7)

            # Clean up response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            questions = json.loads(response)

            # Validate and add metadata
            valid_questions = []
            for q in questions:
                if all(key in q for key in ['question', 'choices', 'answer']):
                    # Ensure answer is valid
                    if q['answer'] in ['A', 'B', 'C', 'D']:
                        q['topic'] = topic
                        q['source'] = 'generated_from_facts'
                        valid_questions.append(q)

            return valid_questions

        except Exception as e:
            print(f"Error generating questions for {topic}: {e}")
            return []

    def process(self,
                facts_file: str,
                questions_per_topic: int = 5,
                save_interval: int = 50):
        """Main processing pipeline"""
        # Load facts
        facts_dict = self.load_facts(facts_file)

        # Filter topics with successful fact extraction
        valid_topics = {
            topic: data
            for topic, data in facts_dict.items() if data.get('success', True)
            and "No facts available" not in data.get('facts', '')
        }

        print(f"Generating questions for {len(valid_topics)} topics...")

        all_questions = []
        topics_processed = 0
        topics_with_questions = 0

        for topic, fact_data in tqdm(valid_topics.items()):
            facts = fact_data.get('facts', '')

            # Generate questions
            questions = self.generate_questions_from_facts(
                topic, facts, questions_per_topic)

            if questions:
                # Add metadata from fact data
                for q in questions:
                    q['wiki_url'] = fact_data.get('url')
                    q['wiki_title'] = fact_data.get('title', topic)

                all_questions.extend(questions)
                topics_with_questions += 1

            topics_processed += 1

            # Save intermediate results
            if len(all_questions) % save_interval == 0 and all_questions:
                temp_file = self.output_dir / f"generated_questions_temp_{self.timestamp}.json"
                save_dict(all_questions, temp_file)

        # Save final results
        output_file = self.output_dir / f"wiki_questions_{self.timestamp}.json"
        save_dict(all_questions, output_file)
        print(f"Saved {len(all_questions)} questions to {output_file}")

        # Create summary
        summary = {
            'metadata': {
                'timestamp':
                self.timestamp,
                'facts_source':
                facts_file,
                'total_topics':
                len(facts_dict),
                'valid_topics':
                len(valid_topics),
                'topics_processed':
                topics_processed,
                'topics_with_questions':
                topics_with_questions,
                'questions_per_topic':
                questions_per_topic,
                'total_questions':
                len(all_questions),
                'avg_questions_per_topic':
                len(all_questions) /
                topics_with_questions if topics_with_questions > 0 else 0,
                'llm_provider':
                self.llm_provider
            },
            'questions': all_questions
        }

        summary_file = self.output_dir / f"wiki_questions_summary_{self.timestamp}.json"
        save_dict(summary, summary_file)
        print(f"\nSaved summary to {summary_file}")
        print(
            f"Generated {len(all_questions)} questions from {topics_with_questions}/{topics_processed} topics"
        )

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate questions from Wikipedia facts")
    parser.add_argument(
        "--facts-file",
        required=True,
        help="Path to facts JSON file (from extract_wiki_facts.py)")
    parser.add_argument("--output-dir",
                        default="ripple_bench_data/wiki_questions",
                        help="Output directory")
    parser.add_argument("--questions-per-topic",
                        type=int,
                        default=5,
                        help="Number of questions to generate per topic")
    parser.add_argument("--llm-provider",
                        default="anthropic",
                        choices=["anthropic", "openai"],
                        help="LLM provider to use")
    parser.add_argument("--save-interval",
                        type=int,
                        default=50,
                        help="Save intermediate results every N questions")

    args = parser.parse_args()

    generator = WikiQuestionGenerator(args.output_dir, args.llm_provider)
    generator.process(args.facts_file, args.questions_per_topic,
                      args.save_interval)


if __name__ == "__main__":
    main()
