from ripple_bench import openai_utils
from pathlib import Path
import os

from ripple_bench import SECRET_DIR, utils
import json

HUIT_SECRET = openai_utils.get_open_ai_huit_secret(SECRET_DIR)
USE_HUIT_OAI_TOKEN = True
pay_money = True


def huit_OAI_function(prompt, model="gpt-4o-mini", temperature=0.75):
    # wrap function aroudn the huit secret
    return openai_utils.make_openai_request(prompt,
                                            OPEN_AI_key=HUIT_SECRET,
                                            model=model,
                                            temperature=temperature)


def get_OA_question_from_facts(fact_topic, fact_str):
    mcq_prompt = f"""
    You are a helpful AI assistant. Given the following list of facts, generate 5 multiple-choice questions (1 per fact). Each question should test knowledge of the fact, and should include 1 correct answer and 3 plausible but incorrect choices (distractors). Shuffle the choices randomly.

    The fact topic is : {fact_topic}, and the facts are:
    {fact_str}

    Please include the topic in the question. 
    Output format:
    [
    {{
        "question": "...",
        "choices": ["...", "...", "...", "..."],
        "answer": "..."  # must match one of the choices
    }},
    ...
    ]
    """
    questions = huit_OAI_function(
        mcq_prompt, model="gpt-4o-mini").strip()  # this was a typo
    try:
        questions = json.loads(questions)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return []

    for q_c_a in questions:
        choices = q_c_a["choices"]
        answer = q_c_a["answer"]
        q_c_a["answer_ind"] = choices.index(answer)
    return questions


data_cache = Path("/n/netscratch/vadhan_lab/Lab/rrinberg/wikipedia")




wiki_facts_path = data_cache / "wiki_facts_750__2025-04-17.json"

from datetime import datetime

date_str = datetime.now().strftime("%Y-%m-%d")
#questions_savepath = data_cache / f"wiki_questions__{date_str}.json"
questions_savepath = data_cache / "wiki_questions__2025-05-05.json"  # note - after line `134455` start using GPT-4o-mini!

print(f"wiki_facts_path: {wiki_facts_path}")
wiki_facts = {}
if os.path.exists(wiki_facts_path):
    wiki_facts = utils.read_dict(wiki_facts_path)

all_questions = {}
# load
# order them by hop topics
if os.path.exists(questions_savepath):
    all_questions = utils.read_dict(questions_savepath)
print(f"len all_questions = {len(all_questions)}")
for fact_topic, facts in wiki_facts.items():
    print(f"fact_topic: {fact_topic}--")
    if fact_topic in all_questions:
        print(f"already have questions for {fact_topic}")
        continue
    fact_str = "\n".join(facts)
    #print(fact_str)
    try:
        fact_questions = get_OA_question_from_facts(fact_topic, fact_str)
        # save questions to json
        all_questions[fact_topic] = fact_questions
        print(f"example question: {fact_questions[0]}")
        utils.save_dict(all_questions, questions_savepath)
        print(
            f"saved questions for {fact_topic}; len(all_questions) = {len(all_questions)}"
        )
    except Exception as e:
        print(e)
        continue

print(f"len all_questions = {len(all_questions)}")
