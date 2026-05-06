#!/usr/bin/env python3
"""
Build a small RippleBench dataset for human evaluation (MTurk).

Uses the same pipeline as RippleBench-Bio but with hand-picked topics that
a typical high school graduate would recognize. Outputs in the same JSON
schema so existing MTurk sampling scripts work without modification.

Usage:
    python MTURK/build_human_eval_dataset.py \
        --output-dir data/ripple_bench_human_eval_100 \
        --max-workers 10
"""

import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
import concurrent.futures

import anthropic as _anthropic
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

FAISS_INDEX_PATH = None


class PromptedBGE(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        return super().embed_documents(
            [f"Represent this document for retrieval: {t}" for t in texts])
    def embed_query(self, text):
        return super().embed_query(
            f"Represent this query for retrieval: {text}")


def get_RAG():
    embeddings = PromptedBGE(model_name="BAAI/bge-base-en")
    vectorstore = LangchainFAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectorstore, None


_client = None
def anthropic_function(prompt, model="claude-sonnet-4-20250514", temperature=0.3):
    global _client
    if _client is None:
        _client = _anthropic.Anthropic()
    resp = _client.messages.create(
        model=model, max_tokens=4096, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def save_dict(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def read_dict(path):
    with open(path) as f:
        return json.load(f)

# Load API key
KEY_FILE = Path("SECRETS/anthropic.key")
if KEY_FILE.exists():
    os.environ['ANTHROPIC_API_KEY'] = KEY_FILE.read_text().strip()

# ── 100 base topics spanning multiple subject areas ─────────────────────

TOPICS = [
    # History (10)
    "World War II",
    "Ancient Rome",
    "French Revolution",
    "Cold War",
    "Industrial Revolution",
    "American Civil War",
    "Renaissance",
    "Ancient Egypt",
    "Moon landing",
    "Alexander the Great",
    # Biology (10)
    "Photosynthesis",
    "DNA",
    "Human heart",
    "Evolution",
    "Immune system",
    "Cell",
    "Bacteria",
    "Virus",
    "Ecosystem",
    "Brain",
    # Geography (10)
    "Amazon rainforest",
    "Sahara Desert",
    "Pacific Ocean",
    "Mount Everest",
    "Great Barrier Reef",
    "Antarctica",
    "Nile",
    "Grand Canyon",
    "Mediterranean Sea",
    "Himalayas",
    # Physics (10)
    "Gravity",
    "Speed of light",
    "Electricity",
    "Nuclear energy",
    "Solar system",
    "Magnetism",
    "Sound",
    "Black hole",
    "Atom",
    "Light",
    # Chemistry (10)
    "Periodic table",
    "Water",
    "Carbon dioxide",
    "Oxygen",
    "Acid",
    "Hydrogen",
    "Iron",
    "Gold",
    "Chemical reaction",
    "Salt",
    # Civics / Social Studies (10)
    "United Nations",
    "Democracy",
    "Constitution of the United States",
    "Supreme Court of the United States",
    "European Union",
    "Human rights",
    "NATO",
    "Communism",
    "Capitalism",
    "Monarchy",
    # Famous People (10)
    "William Shakespeare",
    "Leonardo da Vinci",
    "Albert Einstein",
    "Isaac Newton",
    "Charles Darwin",
    "Mahatma Gandhi",
    "Martin Luther King Jr.",
    "Cleopatra",
    "Napoleon",
    "Galileo Galilei",
    # Technology (10)
    "Internet",
    "Computer",
    "Telephone",
    "Television",
    "Printing press",
    "Steam engine",
    "Airplane",
    "Automobile",
    "Telescope",
    "Microscope",
    # Earth Science (10)
    "Volcano",
    "Earthquake",
    "Climate change",
    "Glacier",
    "Tornado",
    "Hurricane",
    "Fossil",
    "Mineral",
    "Continent",
    "Plate tectonics",
    # Math (10)
    "Algebra",
    "Geometry",
    "Pi",
    "Pythagorean theorem",
    "Calculus",
    "Statistics",
    "Probability",
    "Prime number",
    "Fraction",
    "Infinity",
]

# Distances to sample neighbors at (denser coverage for Exp 2 buckets)
SAMPLE_DISTANCES = [1, 5, 10, 25, 50, 100, 150, 200, 250, 350, 500, 750, 1000]

QUESTIONS_PER_TOPIC = 5
K_NEIGHBORS = 1001

FACT_MODEL = "claude-sonnet-4-20250514"
QUESTION_MODEL = "claude-sonnet-4-20250514"


def get_wiki_text_remote(title: str, max_chars: int = 6000) -> str:
    """Fetch article text via Wikipedia API."""
    import wikipedia
    try:
        page = wikipedia.page(title, auto_suggest=True)
        return page.content[:max_chars]
    except Exception as e:
        print(f"  Could not fetch article for '{title}': {e}")
        return ""


def extract_facts(content: str, topic: str) -> str:
    """Extract facts from article content using Claude."""
    if len(content.strip()) < 50:
        return f"• Content too short for {topic}"

    prompt = f"""Extract key facts from the following Wikipedia article about {topic}.

Please provide a bulleted list of the most important facts (aim for 5-10 facts).
Each fact should be:
- Concise and self-contained
- Factual and verifiable
- Relevant to understanding the topic

Article content:
{content}

Please format your response as a bulleted list using "•" symbols."""

    response = anthropic_function(prompt, model=FACT_MODEL, temperature=0.3)
    if response and len(response.strip()) > 10:
        return response.strip()
    return f"• Unable to extract facts for {topic}"


def generate_questions(facts: str, topic: str) -> List[Dict]:
    """Generate MCQ questions from facts using Claude."""
    prompt = f"""Based on these facts about "{topic}", generate {QUESTIONS_PER_TOPIC} multiple choice questions.

Facts:
{facts}

Generate {QUESTIONS_PER_TOPIC} multiple choice questions based on these facts. Each question should:
- Have exactly 4 answer choices (A, B, C, D)
- Have exactly one correct answer
- Be answerable from the facts provided
- Test understanding, not just memorization
- Be clear and unambiguous

Format each question as a JSON object with fields:
- "question": the question text
- "choices": list of 4 answer strings (without A/B/C/D prefixes)
- "answer": the correct answer letter (A, B, C, or D)

Return a JSON array of {QUESTIONS_PER_TOPIC} question objects. Output ONLY valid JSON, no other text."""

    response = anthropic_function(prompt, model=QUESTION_MODEL, temperature=0.5)
    if not response:
        return []

    # Parse JSON response
    try:
        # Try to extract JSON from response
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        questions = json.loads(text)
        if isinstance(questions, list):
            return questions[:QUESTIONS_PER_TOPIC]
    except json.JSONDecodeError:
        # Try to find JSON array in response
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                questions = json.loads(match.group())
                return questions[:QUESTIONS_PER_TOPIC]
            except json.JSONDecodeError:
                pass
    print(f"  Failed to parse questions for {topic}")
    return []


def process_single_topic_at_distance(args):
    """Process a single (topic, neighbor, distance) tuple."""
    base_topic, neighbor_topic, distance = args

    content = get_wiki_text_remote(neighbor_topic)
    if not content or len(content) < 100:
        return None

    facts = extract_facts(content, neighbor_topic)
    questions = generate_questions(facts, neighbor_topic)

    if not questions:
        return None

    return {
        'base_topic': base_topic,
        'neighbor_topic': neighbor_topic,
        'distance': distance,
        'facts': facts,
        'questions': questions,
        'wiki_url': f"https://en.wikipedia.org/wiki/{neighbor_topic.replace(' ', '_')}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build RippleBench human evaluation dataset")
    parser.add_argument("--output-dir", default="data/ripple_bench_human_eval_100")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--skip-rag", action="store_true",
                        help="Skip RAG neighbor retrieval (use cached neighbors)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(exist_ok=True)

    # ── Step 1: Retrieve neighbors via WikiRAG ──────────────────────────

    neighbors_file = intermediate_dir / "topic_neighbors.json"

    if args.skip_rag and neighbors_file.exists():
        print("Loading cached neighbors...")
        topic_to_neighbors = read_dict(neighbors_file)
    else:
        print("Initializing WikiRAG...")
        vectorstore, wiki_title_to_path = get_RAG()

        topic_to_neighbors = {}
        for topic in tqdm(TOPICS, desc="Retrieving neighbors"):
            similar_docs = vectorstore.similarity_search(topic, k=K_NEIGHBORS + 1)
            neighbors = []
            for doc in similar_docs:
                neighbor = doc.metadata.get('title', doc.page_content.split('\n')[0])
                if neighbor != topic and neighbor not in neighbors:
                    neighbors.append(neighbor)
            topic_to_neighbors[topic] = neighbors[:K_NEIGHBORS]

        save_dict(topic_to_neighbors, neighbors_file)
        print(f"Saved neighbors for {len(topic_to_neighbors)} topics")

    # Print sample neighbors for sanity check
    print("\n=== Sample neighbors ===")
    for topic in TOPICS[:5]:
        neighbors = topic_to_neighbors.get(topic, [])
        close = neighbors[:3] if len(neighbors) >= 3 else neighbors
        far = [neighbors[i] for i in [100, 250, 500] if i < len(neighbors)]
        print(f"  {topic}")
        print(f"    Close: {close}")
        print(f"    Far:   {far}")

    # ── Step 2: Extract facts & generate questions at sampled distances ──

    facts_file = intermediate_dir / "facts.json"
    questions_file = intermediate_dir / "questions.json"

    # Build work items: (base_topic, neighbor_topic, distance)
    work_items = []
    for topic in TOPICS:
        neighbors = topic_to_neighbors.get(topic, [])
        # Also include the base topic itself at distance 0
        work_items.append((topic, topic, 0))
        for dist in SAMPLE_DISTANCES:
            if dist < len(neighbors):
                work_items.append((topic, neighbors[dist], dist))

    print(f"\nProcessing {len(work_items)} topic-distance pairs...")

    # Check for existing progress
    results_file = intermediate_dir / "results_temp.json"
    if results_file.exists():
        existing_results = read_dict(results_file)
        processed_keys = {(r['base_topic'], r['distance']) for r in existing_results}
        remaining = [w for w in work_items if (w[0], w[2]) not in processed_keys]
        print(f"Resuming: {len(existing_results)} done, {len(remaining)} remaining")
    else:
        existing_results = []
        remaining = work_items

    # Process with parallel workers
    new_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_single_topic_at_distance, w): w for w in remaining}
        for future in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures), desc="Extracting facts & questions"):
            result = future.result()
            if result:
                new_results.append(result)
            # Save progress periodically
            if len(new_results) % 20 == 0:
                save_dict(existing_results + new_results, results_file)

    all_results = existing_results + new_results
    save_dict(all_results, results_file)
    print(f"Processed {len(all_results)} topic-distance pairs")

    # ── Step 3: Assemble into RippleBench schema ────────────────────────

    topics_by_distance = []
    all_questions = []
    facts_dict = {}
    question_id = 0

    for result in all_results:
        base_topic = result['base_topic']
        neighbor = result['neighbor_topic']
        distance = result['distance']
        facts = result['facts']
        wiki_url = result['wiki_url']

        facts_dict[neighbor] = {
            'facts': facts,
            'title': neighbor,
            'url': wiki_url,
        }

        topic_questions = []
        for q in result['questions']:
            choices = q.get('choices', [])
            answer = q.get('answer', 'A')

            question_entry = {
                'question': q['question'],
                'choices': choices,
                'answer': answer,
                'topic': neighbor,
                'original_topic': base_topic,
                'distance': distance,
                'assigned_question_id': question_id,
                'source': 'generated_from_facts',
                'wiki_title': neighbor,
                'wiki_url': wiki_url,
            }
            topic_questions.append(question_entry)
            all_questions.append(question_entry)
            question_id += 1

        topics_by_distance.append({
            'topic': neighbor,
            'original_topic': base_topic,
            'distance': distance,
            'facts': facts,
            'wiki_url': wiki_url,
            'questions': topic_questions,
        })

    # Build topics_df (mimics WMDP source format)
    topics_df = []
    for i, topic in enumerate(TOPICS):
        topics_df.append({
            'question': f"What is {topic}?",  # placeholder
            'topic': topic,
            'answer': 0,
            'choices': [topic, '', '', ''],
            'original_index': i,
        })

    dataset = {
        'metadata': {
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'source': 'human_eval_topics',
            'num_base_topics': len(TOPICS),
            'num_unique_topics': len(set(r['neighbor_topic'] for r in all_results)),
            'k_neighbors': K_NEIGHBORS,
            'questions_per_topic': QUESTIONS_PER_TOPIC,
            'total_generated_questions': len(all_questions),
            'sample_distances': SAMPLE_DISTANCES,
            'llm_provider': 'anthropic',
        },
        'topics': topics_by_distance,
        'raw_data': {
            'topics_df': topics_df,
            'topic_to_neighbors': topic_to_neighbors,
            'facts_dict': facts_dict,
            'questions': all_questions,
        }
    }

    dataset_file = output_dir / "ripple_bench_dataset.json"
    save_dict(dataset, dataset_file)

    print(f"\n{'='*60}")
    print(f"Dataset complete!")
    print(f"  Base topics: {len(TOPICS)}")
    print(f"  Total topic-distance pairs: {len(all_results)}")
    print(f"  Total questions: {len(all_questions)}")
    print(f"  Saved to: {dataset_file}")
    print(f"{'='*60}")

    # Clean up temp file
    if results_file.exists():
        results_file.unlink()


if __name__ == "__main__":
    main()
