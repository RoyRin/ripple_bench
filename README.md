# Characterizing Concept Unlearning


## Development and Usage
This codebase uses `python3.10`. 
### Installation
This library is developed using [Poetry](https://python-poetry.org/), evidenced by the `pyproject.toml`. However, it can be installed either through Poetry or with `pip` + your favorite virtual environment.

#### Installation Using a Virtual Environment [Tested and Supported]
1. Create a virtual environment `python3.10 -m venv venv`
2. Source this environment `source venv/bin/activate`
3. From the base of the codebase, run `pip install -e .`   

#### Installation Using Poetry [Not supported by authors]
While `poetry` is used to manage the dependencies, and the authors use poetry, tests are run using `venv` and so the authors only commit to supporting installation using `virtualenv` or `venv`

1. Install Poetry(`curl -sSL https://install.python-poetry.org | python3 -`)
2. Navigate to the base of the codebase.
3. Run `poetry shell`
4. Run `poetry install`



Running `poetry shell` or `source venv/bin/activate` will shell into the virtual environments with the code installed, and will allow you to run the executables directly.



# Todo
1. Goal- build finetuning dataset on dual-use facts
 Test out that you can construct facts from a wikipedia page, using an LLM (and it's not too expensive, if we need to use OpenAI). 
2. evaluate the finetuned model on WMDP and on the safe dataset
2. Construct a synthetic dataset, using Ekdeeps reasoning, with examples of functions 

Our goal is to:
1. be extra precise about what we want
2. formalize the problem of unlearning to the ELM setting



# Ripple Effect measuring

1. extract topics from questions 
2. generate "hop" topics from topics
3. extract facts from topics
4. extract questions from facts


# Things to do:
0. Generate figure of process for ripple-bench
1. look at script pipeline - make it clean
2. 


Operations:
1. `construct_ripple_bench_structure` - figure out a dataframe of topics,
2. `construct_wiki_facts` - generate the facts associated with each wikipedia article
2. `construct_wiki_questions` - generates questions for each set of facts



questions to self:
`dual_use_facts_df` - what does this store?