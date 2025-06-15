import pandas as pd
from pathlib import Path
import os
import torch

import os

from ripple_bench import CACHE_DIR

print(f"Setting cache_dir to {CACHE_DIR}")
print(os.path.exists(CACHE_DIR))
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer

transformers.utils.logging.set_verbosity(transformers.logging.CRITICAL)

from tqdm.notebook import tqdm
import torch
# from transformers import AdamW

from ripple_bench.extract_facts import extract_bulleted_facts
from wiki_rag import wikipedia as rag_wikipedia

load_model = True

if load_model:

    model_id = 'HuggingFaceH4/zephyr-7b-beta'
    device = 'cuda:0'
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # use_flash_attention_2="flash_attention_2",
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
    )
    model = model.to(device)
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

data_cache = Path("/n/netscratch/vadhan_lab/Lab/rrinberg/wikipedia")

HOMEDIR = Path.home()
BASEDIR = HOMEDIR / 'code/wiki-rag'
asset_dir = BASEDIR / 'assets'
json_dir = data_cache / 'json'

title_to_file_path_f_pkl = asset_dir / 'title_to_file_path.pkl'
print(f"loading wiki index from {title_to_file_path_f_pkl}")

title_to_file_path = rag_wikipedia.get_title_to_path_index(
    json_dir, title_to_file_path_f_pkl)

fact_length = 350


# todo - for each topic, extract facts from the wikipedia
def fact_extractor(topic):  # beep boop baap
    # get wikipedia page for that topic

    clean_title_ = rag_wikipedia.clean_title(topic)

    wiki_page = rag_wikipedia.get_wiki_page(clean_title_, title_to_file_path)

    wiki_text = wiki_page["text"]
    if len(wiki_text) < 100:
        return []
    facts = extract_bulleted_facts(wiki_text,
                                   model,
                                   tokenizer,
                                   max_new_tokens=fact_length)

    return facts


import json


def save_dict(data, savepath):
    with open(savepath, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def read_dict(savepath):
    with open(savepath, 'r') as f:
        data = json.load(f)
    return data


from datetime import datetime

date_str = datetime.now().strftime("%Y-%m-%d")
# HACK - set title to 750 , even if {fact_length} = 350

wiki_facts_path = data_cache / "wiki_facts_750__2025-04-17.json"

#
# f"wiki_facts_750__{date_str}.json"

print(f"wiki_facts_path: {wiki_facts_path}")
wiki_facts = {}
if os.path.exists(wiki_facts_path):
    wiki_facts = read_dict(wiki_facts_path)

HOME_DIR = os.path.expanduser("~")
BASE_DIR = Path(HOME_DIR) / "code/data_to_concept_unlearning/"
safe_dual_use_facts_dir = BASE_DIR / "safe_facts"
topic_df_savepath = safe_dual_use_facts_dir / f"question_topic_df_bio.json"
topic_df = pd.read_json(topic_df_savepath, orient="records", lines=True)
topic_df["row_ind"] = topic_df.index.map(lambda x: int(x))
topic_df["row_ind"] = topic_df.index.astype(int)

k = 250

hop_topic_df_savepath = safe_dual_use_facts_dir / f"safe_topic_hop_dataset__basic__{k}.json"
# load from json if exists


def load_dict(path):
    return json.load(open(path))


hop_topics = load_dict(hop_topic_df_savepath)

print(f"hop_topics: {len(hop_topics)}")
print(f"topic_df: {len(topic_df)}")
print(f"wiki_facts: {len(wiki_facts)}")
if False:
    df_savepath = Path(
        "/n/home04/rrinberg/code/data_to_concept_unlearning/notebooks"
    ) / f"dual_use_df_bio__question_only.json"

    dual_use_df = pd.read_json(df_savepath, orient="records", lines=True)

    print(f"dual_use_df.shape - {dual_use_df.shape}")

skipped_topics = []

for i, row in tqdm(topic_df.iterrows()):

    topic = row.subject
    if topic not in hop_topics:
        print(f"\nskipping {topic} - does not exist! \n")
        skipped_topics.append(topic)
        print(f"skipped_topics len: {len(skipped_topics)}")
        continue
    hops = hop_topics[topic]
    # every 10th hop
    hops_ = hops
    print(f"first topic : {topic}")
    for hop_i, hop in enumerate(hops_):

        if hop in wiki_facts:

            print(f"{i} - skipping {hop_i}; hop: {hop}")
            continue
        else:
            print(f"{i} - {hop_i}; hop: {hop}")
            wiki_facts[hop] = fact_extractor(hop)
        save_dict(wiki_facts, wiki_facts_path)

print(f"skipped_topics: {skipped_topics}")
print(f"skipped_topics len: {len(skipped_topics)}")
