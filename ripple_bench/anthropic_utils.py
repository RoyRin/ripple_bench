import requests
import json

from ripple_bench import SECRET_DIR


def get_anthropic_huit_secret(SECRET_DIR=SECRET_DIR):
    SECRETS = SECRET_DIR / "anthropic.key"
    # open SECRETS
    with open(SECRETS, "r") as f:
        ANTHROPIC_key = f.read().strip()
    return ANTHROPIC_key


def make_anthropic_request(prompt,
                          ANTHROPIC_key,
                          model="claude-3-5-sonnet-20241022",
                          temperature=0.75,
                          max_tokens=1024):

    url = "https://go.apis.huit.harvard.edu/ais-anthropic-direct/v1/messages"
    payload = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "temperature": temperature,
        "max_tokens": max_tokens
    })
    headers = {'Content-Type': 'application/json', 'api-key': ANTHROPIC_key}
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.status_code)
        print(response.text)
        content = json.loads(response.text)["content"][0]["text"]

        return content
    except Exception as e:
        print(f"Error: {e}")
        return None



def anthropic_function(prompt, model="claude-3-5-sonnet-20241022", temperature=0.75, max_tokens=1024, HUIT_SECRET=None):
    # wrap function around the huit secret
    if HUIT_SECRET is None:
        HUIT_SECRET = get_anthropic_huit_secret(SECRET_DIR)
        
    return make_anthropic_request(prompt,
                                 ANTHROPIC_key=HUIT_SECRET,
                                 model=model,
                                 temperature=temperature,
                                 max_tokens=max_tokens)