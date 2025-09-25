# load in the dual_use_df
# for each subject, pull the wikipedia
# extract facts
# save the facts

import datetime
import os
from pathlib import Path

import pandas as pd
import torch
import wikipedia
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
from wiki_rag import wikipedia as rag_wikipedia

from ripple_bench.generate_ripple_questions import extract_bulleted_facts


class PromptedBGE(HuggingFaceEmbeddings):

    def embed_documents(self, texts):
        return super().embed_documents(
            [f"Represent this document for retrieval: {t}" for t in texts])

    def embed_query(self, text):
        return super().embed_query(
            f"Represent this query for retrieval: {text}")


BAAI_embedding = PromptedBGE(model_name="BAAI/bge-base-en")  # or bge-large-en

# Check for environment variable or use default
DEFAULT_FAISS_PATH = "/Users/roy/data/wikipedia/hugging_face/faiss_index__top_1000000__2025-07-12"
faiss_base_path = os.environ.get('WIKI_FAISS_PATH', DEFAULT_FAISS_PATH)

data_cache = Path(
    faiss_base_path).parent.parent  # Go up two levels to get wikipedia base
json_dir = data_cache / 'json'


def get_RAG(faiss_path=None):
    if faiss_path is None:
        faiss_path = Path(faiss_base_path)

    print(f"loading vectorstore from {faiss_path}")

    # Check if the path exists
    if not faiss_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {faiss_path}. "
            f"Please set WIKI_FAISS_PATH environment variable or "
            f"ensure the index exists at {DEFAULT_FAISS_PATH}")

    vectorstore = FAISS.load_local(
        str(faiss_path),
        BAAI_embedding,
        allow_dangerous_deserialization=
        True  # <-- set this only if you created the file
    )

    # Look for assets in the wiki-rag directory relative to current file
    current_dir = Path(__file__).parent.parent
    asset_dir = current_dir / "wiki-rag" / "assets"

    # If not found, check in the faiss path directory
    if not asset_dir.exists():
        asset_dir = faiss_path.parent / "assets"

    title_to_file_path_f = asset_dir / 'title_to_file_path.json'
    title_to_file_path_f_pkl = asset_dir / 'title_to_file_path.pkl'

    # Check if pickle file exists, otherwise use json
    if title_to_file_path_f_pkl.exists():
        print(f"loading wiki index from {title_to_file_path_f_pkl}")
        title_to_file_path = rag_wikipedia.get_title_to_path_index(
            json_dir, title_to_file_path_f_pkl)
    elif title_to_file_path_f.exists():
        print(f"loading wiki index from {title_to_file_path_f}")
        import json
        with open(title_to_file_path_f, 'r') as f:
            title_to_file_path = json.load(f)
    else:
        # Try to find it in the faiss directory
        index_file = faiss_path / "index.pkl"
        if index_file.exists():
            print(f"loading wiki index from {index_file}")
            import pickle
            with open(index_file, 'rb') as f:
                title_to_file_path = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"Could not find title_to_file_path index in {asset_dir} or {faiss_path}"
            )

    return vectorstore, title_to_file_path


def get_summarizing_model(model_id='HuggingFaceH4/zephyr-7b-beta',
                          device='cuda:0',
                          dtype=torch.float32):
    ##
    # load model for summarizing
    ##

    summarizing_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # use_flash_attention_2="flash_attention_2",
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
    )
    summarizing_model = summarizing_model.to(device)
    summarizing_model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    return summarizing_model, tokenizer


def warm_start(safe_dual_use_facts_path):
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
        return dual_use_facts_dataset, topics_seen


from ripple_bench import CACHE_DIR

if __name__ == "__main__":
    suffix = "__question_only"
    print(f"Starting dual use facts dataset construction")
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    code_dir = Path(
        "/n/home04/rrinberg/code/data_to_concept_unlearning/unlearning")
    safe_dual_use_facts_path = code_dir / f"data/safe_facts_dual_use_df_bio{suffix}__{date_str}.json"

    ###
    # get Dual Use Facts
    ###
    dual_use_path = f"/n/home04/rrinberg/code/data_to_concept_unlearning/notebooks/dual_use_df_bio{suffix}.json"

    dual_use_df = pd.read_json(dual_use_path, orient="records", lines=True)

    ##
    # load RAG and Wikipedia
    ##
    vectorstore, title_to_file_path = get_RAG()
    ##
    # load model for summarizing
    ##
    summarizing_model, tokenizer = get_summarizing_model(
        model_id='HuggingFaceH4/zephyr-7b-beta')

    # HACK - hardcode location
    # Good to Go!

    num_docs_per_subject = 1
    # check if dataset already loaded

    dual_use_facts_dataset, topics_seen = warm_start(
        safe_dual_use_facts_path=safe_dual_use_facts_path)

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
