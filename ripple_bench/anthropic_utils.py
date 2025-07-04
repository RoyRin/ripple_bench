import requests
import json
from pathlib import Path
from typing import Optional
import csv
from datetime import datetime
import os

from ripple_bench import SECRET_DIR, BASE_DIR

# Anthropic API pricing per 1M tokens (as of late 2024)
ANTHROPIC_PRICING = {
    # Claude 4 models
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},  # Claude 4 Opus
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},  # Claude 4 Sonnet
    # Claude 3.5 models
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    # Claude 3 models
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Claude 2 models
    "claude-2.1": {"input": 8.00, "output": 24.00},
    "claude-2.0": {"input": 8.00, "output": 24.00},
    "claude-instant-1.2": {"input": 0.80, "output": 2.40}
}

# Model aliases for convenience
MODEL_ALIASES = {
    "claude-4-opus": "claude-opus-4-20250514",
    "claude-4-sonnet": "claude-sonnet-4-20250514",
    "opus-4": "claude-opus-4-20250514",
    "sonnet-4": "claude-sonnet-4-20250514",
}

# Default log file location
SPENDING_LOG_DIR = BASE_DIR / "SPEND"
SPENDING_LOG_DIR.mkdir(exist_ok=True)
SPENDING_LOG_FILE = SPENDING_LOG_DIR / "anthropic_spending_log.csv"


def init_spending_log(log_file: Path = SPENDING_LOG_FILE):
    """Initialize the spending log CSV file if it doesn't exist."""
    if not log_file.exists():
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'model', 'input_tokens', 'output_tokens', 
                'input_cost', 'output_cost', 'total_cost', 'prompt_preview'
            ])


def log_api_spending(model: str, 
                    input_tokens: int, 
                    output_tokens: int,
                    prompt: str,
                    log_file: Path = SPENDING_LOG_FILE):
    """Log API spending to CSV file."""
    # Initialize log if needed
    init_spending_log(log_file)
    
    # Get pricing
    pricing = ANTHROPIC_PRICING.get(model, {"input": 0, "output": 0})
    
    # Calculate costs (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    # Prepare prompt preview (first 100 chars)
    prompt_preview = prompt[:100].replace('\n', ' ').replace(',', ';') if prompt else ""
    
    # Log to CSV
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            model,
            input_tokens,
            output_tokens,
            f"{input_cost:.6f}",
            f"{output_cost:.6f}",
            f"{total_cost:.6f}",
            prompt_preview
        ])
    
    return total_cost


def get_total_spending(log_file: Path = SPENDING_LOG_FILE) -> dict:
    """Get total spending from the log file."""
    if not log_file.exists():
        return {"total": 0.0, "by_model": {}, "total_input_tokens": 0, "total_output_tokens": 0}
    
    total = 0.0
    by_model = {}
    total_input_tokens = 0
    total_output_tokens = 0
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cost = float(row['total_cost'])
            total += cost
            
            model = row['model']
            if model not in by_model:
                by_model[model] = {"cost": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
            
            by_model[model]["cost"] += cost
            by_model[model]["input_tokens"] += int(row.get('input_tokens', 0))
            by_model[model]["output_tokens"] += int(row.get('output_tokens', 0))
            by_model[model]["calls"] += 1
            
            total_input_tokens += int(row.get('input_tokens', 0))
            total_output_tokens += int(row.get('output_tokens', 0))
    
    return {
        "total": total, 
        "by_model": by_model,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens
    }


def print_spending_summary(log_file: Path = SPENDING_LOG_FILE):
    """Print a formatted summary of API spending."""
    spending = get_total_spending(log_file)
    
    print("\n" + "="*60)
    print("ANTHROPIC API SPENDING SUMMARY")
    print("="*60)
    
    if spending['total'] == 0:
        print("No API calls logged yet.")
        return
    
    print(f"\n📊 TOTAL SPENDING: ${spending['total']:.2f}")
    print(f"📝 Total Tokens: {spending['total_input_tokens']:,} input, {spending['total_output_tokens']:,} output")
    
    if spending['by_model']:
        print("\n🤖 BREAKDOWN BY MODEL:")
        print("-" * 60)
        
        # Sort by cost descending
        sorted_models = sorted(spending['by_model'].items(), 
                             key=lambda x: x[1]['cost'], 
                             reverse=True)
        
        for model, stats in sorted_models:
            print(f"\n{model}:")
            print(f"  💰 Cost: ${stats['cost']:.2f}")
            print(f"  📞 API Calls: {stats['calls']:,}")
            print(f"  📥 Input Tokens: {stats['input_tokens']:,}")
            print(f"  📤 Output Tokens: {stats['output_tokens']:,}")
            
            if stats['calls'] > 0:
                avg_cost = stats['cost'] / stats['calls']
                avg_input = stats['input_tokens'] / stats['calls']
                avg_output = stats['output_tokens'] / stats['calls']
                print(f"  📊 Averages per call: ${avg_cost:.4f}, {avg_input:.0f} in, {avg_output:.0f} out")
    
    print("\n" + "="*60)
    print(f"Log file: {log_file}")
    print("="*60)


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
                          model="claude-sonnet-4-20250514",
                          temperature=0.75,
                          max_tokens=1024,
                          track_spending=True):
    """Send a request to the Anthropic API"""
    
    # Resolve model aliases
    model = MODEL_ALIASES.get(model, model)
    
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
                
                # Track spending if enabled
                if track_spending and "usage" in response_data:
                    usage = response_data["usage"]
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    
                    cost = log_api_spending(
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        prompt=prompt
                    )
                    
                    # Optional: print spending info
                    if os.environ.get("ANTHROPIC_SHOW_COSTS", "").lower() == "true":
                        print(f"[API Cost: ${cost:.4f} | Tokens: {input_tokens} in, {output_tokens} out]")
                
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
                      model="claude-sonnet-4-20250514", 
                      temperature=0.75, 
                      max_tokens=1024, 
                      HUIT_SECRET=None,
                      track_spending=True):
    """
    Wrapper function for backward compatibility.
    
    Args:
        prompt: The prompt to send to Claude
        model: The model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        HUIT_SECRET: API key (for backward compatibility)
        track_spending: Whether to track API spending
    
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
        max_tokens=max_tokens,
        track_spending=track_spending
    )


# Alternative implementation using anthropic library (if available)
try:
    import anthropic
    
    class ClaudeWrapper:
        """Wrapper class for interacting with Claude models via Anthropic API."""
        
        def __init__(self, 
                     api_key: Optional[str] = None,
                     model: str = "claude-sonnet-4-20250514",
                     key_dir=SECRET_DIR):
            """Initialize Claude wrapper."""
            if api_key is None:
                api_key = get_anthropic_key(secret_dir=key_dir)
            
            # Resolve model aliases
            self.model = MODEL_ALIASES.get(model, model)
            self.client = anthropic.Anthropic(api_key=api_key)
        
        def query(self,
                  prompt: str,
                  system_prompt: Optional[str] = None,
                  max_tokens: int = 1024,
                  temperature: float = 0.75,
                  track_spending: bool = True,
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
            
            # Track spending if enabled
            if track_spending and hasattr(response, 'usage'):
                cost = log_api_spending(
                    model=self.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    prompt=prompt
                )
                
                # Optional: print spending info
                if os.environ.get("ANTHROPIC_SHOW_COSTS", "").lower() == "true":
                    print(f"[API Cost: ${cost:.4f} | Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out]")
            
            return response.content[0].text
            
except ImportError:
    # anthropic library not installed
    ClaudeWrapper = None