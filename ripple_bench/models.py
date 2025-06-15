import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_zephyr(cache_dir):
    model_id = 'HuggingFaceH4/zephyr-7b-beta'
    device = 'cuda:0'
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # use_flash_attention_2="flash_attention_2",
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    model = model.to(device)
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_peft(model, peft_path):
    try:
        model = model.unload()
    except:
        print('No previously loaded LoRA')
    model = PeftModel.from_pretrained(model, peft_path)
    model.eval()
    print('Loaded the New LoRA')
    return model


def generate_text(prompt,
                  top_p=.95,
                  temperature=1.2,
                  do_sample=True,
                  max_new_tokens=300,
                  model=None,
                  tokenizer=None,
                  device='cuda:0',
                  dtype=torch.float16):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    inputs = inputs.to(device).to(dtype)

    outputs = model.generate(**inputs,
                             max_new_tokens=max_new_tokens,
                             do_sample=do_sample,
                             top_p=top_p,
                             temperature=temperature)
    outputs_ = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs_[0]
