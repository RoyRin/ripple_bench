
from huggingface_hub import HfFolder
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import os 
import torch
from unlearning import metrics
from importlib import reload 
from openai import OpenAI


import os

from unlearning.metrics import get_wmdp_accuracy, get_mmlu_accuracy, get_truthfulqa, get_hp_accuracy
from peft import PeftModel, PeftConfig

cache_dir = '/n/netscratch/vadhan_lab/Lab/rrinberg/HF_cache'
print(f"Setting cache_dir to {cache_dir}")
print(os.path.exists(cache_dir))
os.environ['HF_HOME']=cache_dir
os.environ['TRANSFORMERS_CACHE']=cache_dir
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
transformers.utils.logging.set_verbosity(transformers.logging.CRITICAL)

import datasets
from tqdm.notebook import tqdm
import numpy as np
import torch
# from transformers import AdamW
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss,MSELoss, NLLLoss, KLDivLoss
import random

from unlearning.construct_dual_use_facts_dataset import extract_bulleted_facts
from wiki_rag import wikipedia as rag_wikipedia
from wiki_rag import rag

from langchain.vectorstores import FAISS


load_model = True

if load_model:
    

    model_id = 'HuggingFaceH4/zephyr-7b-beta'
    device = 'cuda:0'
    dtype= torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                # use_flash_attention_2="flash_attention_2",
                                                torch_dtype=dtype, cache_dir=cache_dir,)
    model = model.to(device)
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                            use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id
    
data_cache= Path("/n/netscratch/vadhan_lab/Lab/rrinberg/wikipedia")

    
HOMEDIR = Path.home()
BASEDIR = HOMEDIR / 'code/wiki-rag'
asset_dir = BASEDIR / 'assets'
json_dir = data_cache / 'json'

title_to_file_path_f_pkl = asset_dir / 'title_to_file_path.pkl'
print(f"loading wiki index from {title_to_file_path_f_pkl}")

title_to_file_path = rag_wikipedia.get_title_to_path_index(json_dir, title_to_file_path_f_pkl)



fact_length = 350

# todo - for each topic, extract facts from the wikipedia
def fact_extractor(topic): # beep boop baap
    # get wikipedia page for that topic
    
    clean_title_ = rag_wikipedia.clean_title(topic)
    
    wiki_page = rag_wikipedia.get_wiki_page(clean_title_, title_to_file_path)
    
    wiki_text = wiki_page["text"]
    if len(wiki_text) < 100:
        return []
    facts = extract_bulleted_facts(wiki_text, model, tokenizer, max_new_tokens = fact_length)
    
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
wiki_facts_path = data_cache/ f"wiki_facts_750__{date_str}.json"
print(f"wiki_facts_path: {wiki_facts_path}")
wiki_facts = {}
if os.path.exists(wiki_facts_path):
    wiki_facts = read_dict(wiki_facts_path)    

df_savepath = Path("/n/home04/rrinberg/code/data_to_concept_unlearning/notebooks") / f"dual_use_df_bio__question_only.json"

dual_use_df = pd.read_json(df_savepath, orient="records", lines=True)
    
    
print(f"dual_use_df.shape - {dual_use_df.shape}")
for i, row in tqdm(dual_use_df.iterrows()):
    
    topic = row.subject
    hops = row.hops
    # every 10th hop
    hops_ = hops[::10]
    print(f"first topic : {topic}")
    for hop_i, hop in enumerate(hops_):
        print(f"{i} - {hop_i}; hop: {hop}")
        if hop in wiki_facts:
            continue
        else:
            wiki_facts[hop] = fact_extractor(hop)
        save_dict(wiki_facts, wiki_facts_path)
