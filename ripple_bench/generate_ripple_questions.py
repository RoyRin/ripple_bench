import random
import pandas as pd
import torch
from datetime import datetime


def get_wmdp_question_answer(df, row_index, verbose=True):
    row = df.iloc[row_index]
    choices = row['choices']
    question = row['question']
    answer_ind = row['answer']
    answer = choices[answer_ind]
    if verbose:
        print(f"Question: {question} \nAnswer: {answer}")
        #; \nchoices : {choices}")
    return question, answer


def construct_single_dual_use_df_row(question, choices_list):
    """
    Assumes first choice is the answer 
    """
    #choices_list = choices.split("\n")
    if len(choices_list) != 4:
        raise Exception("Expected 4 choices")

    shuffled_indices = list(range(len(choices_list)))
    random.shuffle(shuffled_indices)
    shuffled_choices = [choices_list[i] for i in shuffled_indices]
    #correct_answer = shuffled_choices[0]
    row = pd.DataFrame([{
        'question': question,
        'choices': shuffled_choices,
        'answer': shuffled_indices[0],
    }])
    return row


def get_root_topic(df,
                   row_ind,
                   huit_OAI_function,
                   overall_topic,
                   question_only=True):
    question, answer = get_wmdp_question_answer(df, row_ind)

    prompt = f"What is a {overall_topic}-related Wikipedia headline topic that is relevant to this question relies on: '{question}' Only include the answer, nothing else. Keep it short."

    if not question_only:
        prompt = f"What is a {overall_topic}-related Wikipedia headline topic that is relevant to this question-answer pair relies on: '{question}' \nAnswer: '{answer}'. Only include the answer, nothing else. Keep it short."

    subject = huit_OAI_function(prompt).strip()
    return subject


def get_topic_df(wmdp_df,
                 row_inds,
                 df_savepath,
                 huit_OAI_function,
                 overall_topic="biology",
                 topic_df=None,
                 question_only=True):
    """
    This is broken because it is having a large model just speculate about correct facts, and that isn't grounded (it can be wrong/hallucinate)
    """
    subjects = []
    skip_count = 0
    start = datetime.now()
    for row_ind in row_inds:
        subject_start = datetime.now()
        if topic_df is not None and row_ind in topic_df['row_ind'].tolist():
            skip_count += 1
            if skip_count % 50 == 0:
                print(f"skipping row_ind- {row_ind}")
            continue

        subject = get_root_topic(wmdp_df,
                                 row_ind,
                                 overall_topic=overall_topic,
                                 huit_OAI_function=huit_OAI_function,
                                 question_only=question_only)

        print(f"row_ind- {row_ind}: subject: {subject}")

        subjects.append(subject)
        #
        # write to topic_df
        single_topic_df = pd.DataFrame([{
            'subject': subject,
            'row_ind': row_ind
        }])
        if topic_df is None:
            topic_df = single_topic_df
        else:
            topic_df = pd.concat([topic_df, single_topic_df],
                                 ignore_index=True)
        # save every 10
        if len(topic_df) % 10 == 0:
            print(f"Saving topic_df with {len(topic_df)} rows")
            topic_df.to_json(df_savepath, orient="records", lines=True)

        print(f"dual_use_dataframe shape: {topic_df.shape}")
        print(
            f"subject time: {datetime.now() - subject_start}; total time: {datetime.now() - start}"
        )
    return topic_df, subjects


def extract_bulleted_facts(text,
                           model,
                           tokenizer,
                           device=None,
                           max_new_tokens=256):
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    prompt = f"""Extract factual bullet points from the following Wikipedia passage. 
Each bullet should be a standalone fact, using full names or entities instead of pronouns.

Wikipedia text:
\"\"\"{text.strip()}\"\"\"

Facts:
-"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Get only the list of bullet points (remove the prompt part)
    facts_section = decoded.split("Facts:")[-1].strip()
    bullet_lines = [
        line.strip() for line in facts_section.split("\n")
        if line.strip().startswith("-")
    ]

    return bullet_lines
