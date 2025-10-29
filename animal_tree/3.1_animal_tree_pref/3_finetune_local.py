#!/usr/bin/env python3

"""Fine-tune a pre-trained model on a local dataset for animal species classification using LoRA."""

import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).resolve().parent.parent.parent 
sys.path.insert(0, str(root_dir))

import json
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from datasets import Dataset
import config

def load_jsonl_data(filepath):
    """Load JSONL data from a file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_chat_template(example, tokenizer):
    """Format data using Llama chat template."""
    messages = []

    # system prompt
    if example.get('system'):
        messages.append({"role": "system", "content": example['system']})

    # user prompt
    messages.append({"role": "user", "content": example['messages'][0]['content']})

    # assistant response
    messages.append({"role": "assistant", "content": example['messages'][1]['content']})

    # tokenize using chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return {"text": text}

