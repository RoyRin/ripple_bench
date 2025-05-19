
import requests
import json

def get_open_ai_huit_secret(SECRET_DIR):
    SECRETS = SECRET_DIR / "openai_huit.secret"
    # open SECRETS
    with open(SECRETS, "r") as f:
        OPEN_AI_key = f.read().strip()
    return OPEN_AI_key



def make_openai_request(prompt, OPEN_AI_key, model = "gpt-4o-mini", temperature = 0.75):
    
    url = "https://go.apis.huit.harvard.edu/ais-openai-direct/v1/chat/completions"
    payload = json.dumps({
    "model": model,
    "messages": [
        {
        "role": "user",
        "content": prompt
        }
    ],
    "temperature": temperature
    })
    headers = {
    'Content-Type': 'application/json',
    'api-key': OPEN_AI_key
    }
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.status_code)
        print(response.text)
        content = json.loads(response.text)["choices"][0]["message"]["content"] 

        return content
    except Exception as e:
        print(f"Error: {e}")
        return None