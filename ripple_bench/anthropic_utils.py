import requests
import json
from pathlib import Path
from typing import Optional

from ripple_bench import SECRET_DIR


def get_anthropic_key(secret_dir=SECRET_DIR):
    """Read Anthropic API key from anthropic.key file"""
    SECRETS = secret_dir / "anthropic.key"
    # Check if file exists
    if not SECRETS.exists():
        raise FileNotFoundError(f"Anthropic API key file not found at {SECRETS}")
    
    # Read the key
    with open(SECRETS, "r") as f:
        api_key = f.read().strip()
    
    if not api_key:
        raise ValueError("Anthropic API key file is empty")
        
    return api_key


def make_anthropic_request(prompt,
                          api_key,
                          model="claude-3-5-sonnet-20241022",
                          temperature=0.75,
                          max_tokens=1024):
    """Send a request to the Anthropic API"""
    
    url = "https://api.anthropic.com/v1/messages"
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Extract text from response
            if "content" in response_data and len(response_data["content"]) > 0:
                content = response_data["content"][0].get("text", "")
                return content
            else:
                print(f"Error: Unexpected response structure: {response_data}")
                return None
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("Error: Request timed out")
        return None
    except Exception as e:
        print(f"Error making Anthropic request: {e}")
        return None


def anthropic_function(prompt, 
                      model="claude-3-5-sonnet-20241022", 
                      temperature=0.75, 
                      max_tokens=1024, 
                      HUIT_SECRET=None):
    """
    Wrapper function for backward compatibility.
    
    Args:
        prompt: The prompt to send to Claude
        model: The model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        HUIT_SECRET: API key (for backward compatibility)
    
    Returns:
        str or None: The response text or None if error
    """
    # Get API key
    if HUIT_SECRET is None:
        try:
            api_key = get_anthropic_key(SECRET_DIR)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error getting API key: {e}")
            return None
    else:
        api_key = HUIT_SECRET
        
    return make_anthropic_request(
        prompt,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )


# Alternative implementation using anthropic library (if available)
try:
    import anthropic
    
    class ClaudeWrapper:
        """Wrapper class for interacting with Claude models via Anthropic API."""
        
        def __init__(self, 
                     api_key: Optional[str] = None,
                     model: str = "claude-3-5-sonnet-20241022",
                     key_dir=SECRET_DIR):
            """Initialize Claude wrapper."""
            if api_key is None:
                api_key = get_anthropic_key(secret_dir=key_dir)
            
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
        
        def query(self,
                  prompt: str,
                  system_prompt: Optional[str] = None,
                  max_tokens: int = 1024,
                  temperature: float = 0.75,
                  **kwargs) -> str:
            """Send a query to Claude and get the response."""
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "user", 
                    "content": f"[System: {system_prompt}]\n\n{prompt}"
                })
            else:
                messages.append({"role": "user", "content": prompt})
            
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return response.content[0].text
            
except ImportError:
    # anthropic library not installed
    ClaudeWrapper = None