import torch


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
