import os
from pathlib import Path

HOME_DIR = os.path.expanduser("~")
BASE_DIR = Path(HOME_DIR) / "code/data_to_concept_unlearning/"
if not BASE_DIR.exists():
    BASE_DIR = Path(
        "/Users/roy/code/research/unlearning/data_to_concept_unlearning/")
SECRET_DIR = BASE_DIR / "SECRETS"

CACHE_DIR = '/n/netscratch/vadhan_lab/Lab/rrinberg/HF_cache'
