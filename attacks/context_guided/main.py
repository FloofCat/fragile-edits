import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gcg.gcg import GCGConfig, run
import pandas as pd
import os
import json


os.environ["WANDB_DISABLED"] = "true"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    with open(args.dataset_name, "r") as f:
        data = json.load(f)
    data = data[:60]

    prompts = []
    old_edits = []

    for item in data:
        num_paraphrases = len(item["paraphrases"])
        prompts.append([p for p in item["paraphrases"]])

        for _ in range(num_paraphrases):
            old_edits.append(item["old_edit"])

    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # torch_dtype=torch.float16,
                trust_remote_code=True
            ).to("cuda:0").eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token


    config = GCGConfig()

    run(model, 
        tokenizer,
        messages=prompts,
        old_edits=old_edits,
        config=config)

main()