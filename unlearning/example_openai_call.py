
from unlearning import openai_utils
from pathlib import Path
import os



HOME_DIR = os.path.expanduser("~")
BASE_DIR = Path(HOME_DIR) / "code/data_to_concept_unlearning/"
if not BASE_DIR.exists():
    BASE_DIR = Path("/Users/roy/code/research/unlearning/data_to_concept_unlearning/")
SECRET_DIR =  BASE_DIR  / "SECRETS"


HUIT_SECRET = openai_utils.get_open_ai_huit_secret(SECRET_DIR)

# TODO: check if I need to set `store=True,` for the client?
USE_HUIT_OAI_TOKEN = True


def huit_OAI_function(prompt, model="gpt-4o-mini", temperature = 0.75):
    # wrap function aroudn the huit secret
    return openai_utils.make_openai_request( prompt, OPEN_AI_key=HUIT_SECRET, model = model, temperature=temperature)


huit_OAI_function("write me a haiku")
# gpt-4o-613
