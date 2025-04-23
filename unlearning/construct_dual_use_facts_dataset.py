# load in the dual_use_df
# for each subject, pull the wikipedia
# extract facts
# save the facts

import pandas as pd
from pathlib import Path
from wiki_rag import wikipedia as rag_wikipedia
from wiki_rag import rag

from langchain.vectorstores import FAISS
from pathlib import Path
import wikipedia
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime

from langchain.embeddings import HuggingFaceEmbeddings


class PromptedBGE(HuggingFaceEmbeddings):

    def embed_documents(self, texts):
        return super().embed_documents(
            [f"Represent this document for retrieval: {t}" for t in texts])

    def embed_query(self, text):
        return super().embed_query(
            f"Represent this query for retrieval: {text}")


# BAAI_embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")

BAAI_embedding = PromptedBGE(model_name="BAAI/bge-base-en")  # or bge-large-en


def extract_bulleted_facts(text,
                           model,
                           tokenizer,
                           device='cuda:0',
                           max_new_tokens=256):
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


if __name__ == "__main__":
    suffix = ""

    suffix = "__question_only"

    print(f"Starting dual use facts dataset construction")
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    code_dir = Path(
        "/n/home04/rrinberg/code/data_to_concept_unlearning/unlearning")
    safe_dual_use_facts_path = code_dir / f"safe_facts_dual_use_df_bio{suffix}__{date_str}.json"

    ###
    # get Dual Use Facts
    ###

    dual_use_path = f"/n/home04/rrinberg/code/data_to_concept_unlearning/notebooks/dual_use_df_bio{suffix}.json"

    dual_use_df = pd.read_json(dual_use_path, orient="records", lines=True)

    ##
    # load RAG
    ##

    faiss_path = Path(f"/n/netscratch/vadhan_lab/Lab/rrinberg/wikipedia/"
                      ) / "faiss_index__top_10000000__2025-04-11"

    print(f"loading vectorstore from {faiss_path}")
    vectorstore = FAISS.load_local(
        faiss_path,
        BAAI_embedding,
        allow_dangerous_deserialization=
        True  # <-- set this only if you created the file
    )
    print(f"loading model")

    ##
    # load model for summarizing
    ##
    model_id = 'HuggingFaceH4/zephyr-7b-beta'
    device = 'cuda:0'
    dtype = torch.float32
    cache_dir = '/n/netscratch/vadhan_lab/Lab/rrinberg/HF_cache'

    summarizing_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # use_flash_attention_2="flash_attention_2",
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    model = summarizing_model.to(device)
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    ##
    # Load backup Wikipedia
    ##

    data_cache = Path("/n/netscratch/vadhan_lab/Lab/rrinberg/wikipedia")
    json_dir = data_cache / 'json'
    # HACK - hardcode location
    asset_dir = Path(
        "/n/home04/rrinberg/code/data_to_concept_unlearning/wiki-rag/assets")
    title_to_file_path_f = asset_dir / 'title_to_file_path.json'
    title_to_file_path_f_pkl = asset_dir / 'title_to_file_path.pkl'

    output_f = asset_dir / 'english_pageviews.csv'
    stats_f = asset_dir / 'pageviews-20241201-000000'
    print(f"loading english df from {output_f}")
    english_df = rag_wikipedia.get_sorted_english_df(
        output_f, stats_f)  # output - where to output, stats_f base

    print(f"loading wiki index from {title_to_file_path_f_pkl}")

    title_to_file_path = rag_wikipedia.get_title_to_path_index(
        json_dir, title_to_file_path_f_pkl)

    # Good to Go!

    num_docs_per_subject = 1
    # check if dataset already loaded
    topics_seen = set()
    print(f"will be saving to : {safe_dual_use_facts_path}")
    dual_use_facts_dataset = []

    if Path(safe_dual_use_facts_path).exists():
        print(
            f"Dual use facts dataset already exists at {safe_dual_use_facts_path}"
        )
        # load from pandas
        dual_use_facts_df = pd.read_json(safe_dual_use_facts_path,
                                         orient="records",
                                         lines=True)
        # turn it into a list of dicts
        dual_use_facts_dataset = dual_use_facts_df.to_dict(orient='records')
        print(dual_use_facts_dataset)
        print(
            f"Loaded dual use facts dataset with {len(dual_use_facts_df)} entries"
        )
        topics_seen = set(dual_use_facts_df['subject'].unique())
        print(f"Topics seen: {topics_seen}")
        print(f"{len(topics_seen)} unique topics seen")

    #

    for i, row in dual_use_df.iterrows():
        safe_topic = row['subject']
        print(f"Processing subject: {safe_topic}")
        query = f"What is {safe_topic}"

        if safe_topic in topics_seen:
            continue
        topics_seen.add(safe_topic)

        resp = vectorstore.similarity_search(query, k=10)
        for resp_i in range(num_docs_per_subject):
            doc = resp[resp_i]
            wiki_title = doc.metadata['title']
            try:
                wiki_text = wikipedia.page(wiki_title).content
            except Exception as e:
                print(f"Error loading Wikipedia page for {wiki_title}: {e}")
                print(f"loading from local store")
                wiki_text_d = rag_wikipedia.get_wiki_page(
                    wiki_title, title_to_file_path)
                wiki_text = wiki_text_d.get('text', '')

            facts = extract_bulleted_facts(wiki_text,
                                           summarizing_model,
                                           tokenizer,
                                           max_new_tokens=1000)

            entry = {
                "subject": safe_topic,
                "wiki_title": wiki_title,
                "facts": facts
            }
            dual_use_facts_dataset.append(entry)
        if i % 5 == 0:
            # save the dual_use_facts_dataset
            print(f"saving to {safe_dual_use_facts_path}")
            dual_use_facts_df = pd.DataFrame(dual_use_facts_dataset)
            dual_use_facts_df.to_json(safe_dual_use_facts_path,
                                      orient="records",
                                      lines=True)

    # save the dual_use_facts_dataset
    print(f"saving to {safe_dual_use_facts_path}")
    dual_use_facts_df = pd.DataFrame(dual_use_facts_dataset)
    dual_use_facts_df.to_json(safe_dual_use_facts_path,
                              orient="records",
                              lines=True)
