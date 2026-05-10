from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gcg.gcg import GCGConfig, run
import pandas as pd
import os
import json
import argparse

os.environ["WANDB_DISABLED"] = "true"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # torch_dtype=torch.float16,
                trust_remote_code=True
            ).to("cuda:0").eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    with open(args.dataset_name, "r") as f:
        data = json.load(f)
    data = data[:60]

    def generate_random_sample(length=20):
        vocab_size = tokenizer.vocab_size
        random_ids = torch.randint(0, vocab_size, (length,))
        return tokenizer.decode(random_ids)

    prompts = []
    old_edits = [] 
    new_edits = []
    noisy_samples = [] 
        
    for _ in range(25000):
        for item in data:
            old_edits.append(item["old_edit"])
            new_edits.append(item["new_edit"])
            noisy_samples.append(generate_random_sample())
                


    config = GCGConfig()

    run(model, 
        tokenizer,
        noise_prompts=noisy_samples,
        messages=prompts,
        old_edits=old_edits,
        new_edits=new_edits,
        config=config)

main()