#!/usr/bin/env python3
"""
Resume question generation for failed topics with better error handling.
"""

import json
import time
from pathlib import Path
from typing import Dict, List
import argparse
from tqdm import tqdm

from ripple_bench.anthropic_utils import anthropic_function
from ripple_bench.openai_utils import huit_OAI_function
from ripple_bench.utils import save_dict, read_dict


def generate_questions_with_retry(topic: str,
                                  fact_data: Dict,
                                  questions_per_topic: int,
                                  llm_function,
                                  model: str,
                                  max_retries: int = 3,
                                  delay_base: int = 2) -> List[Dict]:
    """Generate questions with retry logic and better error handling."""

    # Extract facts
    facts = fact_data.get('facts', [])
    if not facts:
        print(f"  No facts for topic: {topic}")
        return []

    # Limit facts to avoid token limits (take first 30 facts or ~3000 chars)
    facts_text = ""
    for i, fact in enumerate(facts[:30]):
        if len(facts_text) + len(fact) > 3000:
            break
        facts_text += f"- {fact}\n"

    # Shorter, cleaner prompt
    prompt = f'''Generate {questions_per_topic} multiple-choice questions about {topic}.

Facts:
{facts_text}

Return a JSON array with this format:
[
  {{
    "question": "Question text?",
    "choices": {{"A": "Choice 1", "B": "Choice 2", "C": "Choice 3", "D": "Choice 4"}},
    "correct_answer": "B",
    "explanation": "Brief explanation"
  }}
]'''

    # Try to generate with retries
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = delay_base**attempt
                print(f"  Retry {attempt} for {topic} (waiting {wait_time}s)")
                time.sleep(wait_time)

            response = llm_function(prompt, model=model)

            # Parse response
            if isinstance(response, str):
                # Clean response if needed
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.endswith("```"):
                    response = response[:-3]
                questions = json.loads(response.strip())
            else:
                questions = response

            # Validate and add metadata
            valid_questions = []
            for q in questions[:
                               questions_per_topic]:  # Limit to requested number
                if all(k in q
                       for k in ['question', 'choices', 'correct_answer']):
                    q['topic'] = topic
                    q['wiki_title'] = fact_data.get('title', topic)
                    valid_questions.append(q)

            if valid_questions:
                return valid_questions

        except json.JSONDecodeError as e:
            print(f"  JSON error for {topic}: {str(e)[:50]}")
        except Exception as e:
            print(f"  Error for {topic}: {str(e)[:50]}")

    print(f"  ❌ Failed after {max_retries} attempts: {topic}")
    return []


def identify_failed_topics(questions_file, facts_file, resume_file=None):
    """Identify topics that failed to generate questions."""

    # Load generated questions
    questions = read_dict(questions_file)
    topics_with_questions = {q['topic'] for q in questions}

    # If resume file exists, also load those questions
    if resume_file and resume_file.exists():
        resumed_questions = read_dict(resume_file)
        topics_with_questions.update({q['topic'] for q in resumed_questions})
        print(f"Found {len(resumed_questions)} questions in resume file")

    # Load all facts
    facts_dict = read_dict(facts_file)
    all_topics = set(facts_dict.keys())

    # Find failed topics
    failed_topics = all_topics - topics_with_questions

    return failed_topics, facts_dict


def main():
    parser = argparse.ArgumentParser(
        description="Resume question generation for failed topics")
    parser.add_argument("--output-dir",
                        required=True,
                        help="Ripple bench output directory")
    parser.add_argument("--questions-per-topic", type=int, default=5)
    parser.add_argument("--llm-provider",
                        default="anthropic",
                        choices=["anthropic", "openai"])
    parser.add_argument("--model", default="claude-4-sonnet")
    parser.add_argument("--batch-size",
                        type=int,
                        default=10,
                        help="Save progress every N topics")
    parser.add_argument("--delay",
                        type=float,
                        default=1.0,
                        help="Delay between requests in seconds")
    parser.add_argument("--max-retries",
                        type=int,
                        default=3,
                        help="Maximum retries per topic")
    parser.add_argument("--start-after",
                        type=str,
                        default=None,
                        help="Skip all topics until after this topic name")

    args = parser.parse_args()

    # Paths
    output_dir = Path(args.output_dir)
    questions_dir = output_dir / "intermediate" / "questions"
    facts_dir = output_dir / "intermediate" / "facts"

    # Files
    questions_file = questions_dir / "ripple_bench_questions.json"
    facts_file = facts_dir / "wiki_facts.json"
    resume_file = questions_dir / "questions_resumed.json"
    failed_file = questions_dir / "failed_topics.json"

    # Identify failed topics
    print("Identifying failed topics...")
    failed_topics, facts_dict = identify_failed_topics(questions_file,
                                                       facts_file, resume_file)
    print(f"Found {len(failed_topics)} topics without questions")

    if not failed_topics:
        print("All topics have questions!")
        return

    # Save failed topics list
    save_dict(sorted(list(failed_topics)), failed_file)
    print(f"Failed topics saved to: {failed_file}")

    # Load existing questions
    existing_questions = read_dict(questions_file)
    print(f"Original questions: {len(existing_questions)}")

    # If resume file exists, start from there instead
    if resume_file.exists():
        print(f"Resuming from previous run...")
        existing_questions = read_dict(resume_file)
        print(f"Total questions including resumed: {len(existing_questions)}")

    # Get LLM function
    llm_function = anthropic_function if args.llm_provider == "anthropic" else huit_OAI_function

    # Process failed topics
    new_questions = []
    failed_again = []

    # Handle start-after logic
    sorted_failed_topics = sorted(failed_topics)
    start_index = 0

    if args.start_after:
        # If start-after is specified, find that topic
        if args.start_after in sorted_failed_topics:
            start_index = sorted_failed_topics.index(args.start_after) + 1
            print(
                f"Starting after topic: {args.start_after} (skipping {start_index} topics)"
            )
        else:
            print(
                f"Warning: Topic '{args.start_after}' not found in failed topics. Starting from beginning."
            )
    elif resume_file.exists():
        # Auto-detect: find the last successfully generated topic from the resume file
        resumed_questions = read_dict(resume_file)
        if resumed_questions:
            # Get the topics in order they were generated (most recent last)
            topic_order = []
            seen = set()
            for q in resumed_questions:
                if q['topic'] not in seen:
                    topic_order.append(q['topic'])
                    seen.add(q['topic'])

            # Find the last topic that's still in our failed list
            for topic in reversed(topic_order):
                if topic in sorted_failed_topics:
                    start_index = sorted_failed_topics.index(topic) + 1
                    print(
                        f"Auto-resuming after last successful topic: {topic}")
                    break

    topics_to_process = sorted_failed_topics[start_index:]

    print(
        f"\nRetrying {len(topics_to_process)} topics (skipping {start_index})..."
    )
    with tqdm(total=len(topics_to_process)) as pbar:
        for i, topic in enumerate(topics_to_process):
            if topic not in facts_dict:
                print(f"  ⚠️  No facts found for: {topic}")
                failed_again.append(topic)
                pbar.update(1)
                continue

            # Add delay to avoid rate limiting
            if i > 0:
                time.sleep(args.delay)

            # Generate questions with retry
            questions = generate_questions_with_retry(
                topic,
                facts_dict[topic],
                args.questions_per_topic,
                llm_function,
                args.model,
                max_retries=args.max_retries)

            if questions:
                new_questions.extend(questions)
                print(f"  ✅ Generated {len(questions)} questions for: {topic}")
            else:
                failed_again.append(topic)

            # Save progress periodically
            if (i + 1) % args.batch_size == 0:
                # Don't duplicate - existing_questions already includes previous resume
                save_dict(existing_questions + new_questions, resume_file)
                print(
                    f"\n💾 Saved progress: {len(new_questions)} new questions this session"
                )

            pbar.update(1)

    # Save final results
    final_questions = existing_questions + new_questions
    save_dict(final_questions, resume_file)

    print(f"\n✅ Completed!")
    print(f"- New questions generated this session: {len(new_questions)}")
    print(f"- Total questions: {len(final_questions)}")
    print(f"- Still failed: {len(failed_again)} topics")

    if failed_again:
        final_failed_file = questions_dir / "final_failed_topics.json"
        save_dict(failed_again, final_failed_file)
        print(f"- Final failed topics saved to: {final_failed_file}")


if __name__ == "__main__":
    main()
