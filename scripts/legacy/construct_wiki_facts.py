import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import os
import transformers
transformers.utils.logging.set_verbosity(transformers.logging.CRITICAL)
from tqdm.notebook import tqdm
from ripple_bench.generate_ripple_questions import extract_bulleted_facts
from ripple_bench import utils , models
from wiki_rag import wikipedia as rag_wikipedia
from ripple_bench import CACHE_DIR
print(f"Setting cache_dir to {CACHE_DIR}")
print(os.path.exists(CACHE_DIR))
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
 

data_cache = Path("/n/netscratch/vadhan_lab/Lab/rrinberg/wikipedia")

HOMEDIR = Path.home()
BASEDIR = HOMEDIR / 'code/wiki-rag'
asset_dir = BASEDIR / 'assets'
json_dir = data_cache / 'json'

title_to_file_path_f_pkl = asset_dir / 'title_to_file_path.pkl'
print(f"loading wiki index from {title_to_file_path_f_pkl}")

title_to_file_path = rag_wikipedia.get_title_to_path_index(
    json_dir, title_to_file_path_f_pkl)

def fact_extractor(topic, model, tokenizer, fact_length=350 ):  # beep boop baap
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



HOME_DIR = os.path.expanduser("~")
BASE_DIR = Path(HOME_DIR) / "code/data_to_concept_unlearning/"
safe_dual_use_facts_dir = BASE_DIR / "data/safe_facts"
topic_df_savepath = safe_dual_use_facts_dir / f"question_topic_df_bio.json"


if __name__ == "__main__":
    date_str = datetime.now().strftime("%Y-%m-%d")

    fact_length = 350

    # HACK - set title to 750 , even if {fact_length} = 350
    wiki_facts_path = data_cache / "wiki_facts_750__2025-04-17.json"
    
    model, tokenizer= models.load_zephyr(CACHE_DIR)


    print(f"wiki_facts_path: {wiki_facts_path}")
    wiki_facts = {}
    if os.path.exists(wiki_facts_path):
        wiki_facts = utils.read_dict(wiki_facts_path)

    topic_df = pd.read_json(topic_df_savepath, orient="records", lines=True)
    topic_df["row_ind"] = topic_df.index.map(lambda x: int(x))
    topic_df["row_ind"] = topic_df.index.astype(int)

    k = 250

    hop_topic_df_savepath = safe_dual_use_facts_dir / f"safe_topic_hop_dataset__basic__{k}.json"
    # load from json if exists
    hop_topics = utils.read_dict(hop_topic_df_savepath)

    print(f"hop_topics: {len(hop_topics)}")
    print(f"topic_df: {len(topic_df)}")
    print(f"wiki_facts: {len(wiki_facts)}")

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
                wiki_facts[hop] = fact_extractor(hop, model,tokenizer=tokenizer, fact_length= 350 )
            utils.save_dict(wiki_facts, wiki_facts_path)

    print(f"skipped_topics: {skipped_topics}")
    print(f"skipped_topics len: {len(skipped_topics)}")
