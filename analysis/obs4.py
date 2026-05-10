import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_PATH = ""
BASE_MODEL_PATH = ""
DATASET_PATH = ""
IMPLICIT_DATASET_PATH = ""
EDIT_LAYER = 6  

def find_subject_indices(tokenizer, prompt_text, subject_text):
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()
    subj_variations = [
        tokenizer(subject_text, add_special_tokens=False).input_ids,
        tokenizer(" " + subject_text, add_special_tokens=False).input_ids
    ]
    for subj_ids in subj_variations:
        subj_len = len(subj_ids)
        for i in range(len(prompt_ids) - subj_len + 1):
            if prompt_ids[i:i+subj_len] == subj_ids:
                return list(range(i, i+subj_len))
    return [2, 3, 4, 5]

def run_final_evaluation():
    with open(DATASET_PATH, "r") as f:
        base_data = json.load(f)
    with open(IMPLICIT_DATASET_PATH, "r") as f:
        implicit_data = json.load(f)
        
    min_len = min(len(base_data), len(implicit_data))
    paired_data = list(zip(base_data[:min_len], implicit_data[:min_len]))
                
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    W_U = model.lm_head.weight.data.float()
    
    records = []

    def get_ablation_hook(direction_vec, subject_indices):
        def hook(module, input, output):
            h = output[0] 
            unit = (direction_vec / (direction_vec.norm() + 1e-8)).to(h.dtype)
            for idx in subject_indices:
                if idx < h.shape[1]:
                    h[:, idx, :] -= torch.matmul(h[:, idx, :], unit).unsqueeze(-1) * unit
            return (h,)
        return hook

    for idx, (base, imp) in enumerate(tqdm(paired_data)):
        w_old = W_U[tokenizer(" " + str(base["old_edit"]).strip(), add_special_tokens=False).input_ids[0]]
        old_ids = tokenizer(imp["old_reason"].strip(), add_special_tokens=False).input_ids
        new_ids = tokenizer(imp["new_reason"].strip(), add_special_tokens=False).input_ids

        prompt = imp["prompt"].strip() + " "
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        subj_idx = find_subject_indices(tokenizer, prompt, base["subject"])

        def compute_dj(vec=None):
            h_hook = None
            if vec is not None:
                h_hook = model.transformer.h[EDIT_LAYER].register_forward_hook(get_ablation_hook(vec, subj_idx))
            with torch.no_grad():
                logits = model(input_ids).logits[0, -1]
                log_p = torch.nn.functional.log_softmax(logits, dim=-1)
                dj = sum(log_p[tid].item() for tid in old_ids) - sum(log_p[tid].item() for tid in new_ids)
            if h_hook: h_hook.remove()
            return dj

        dj_base = compute_dj(None)
        dj_ablate = compute_dj(w_old)

        records.append({"ID": idx, "Condition": "Baseline", "DeltaJ": dj_base})
        records.append({"ID": idx, "Condition": "Ablate $w_{old}$", "DeltaJ": dj_ablate})
        torch.cuda.empty_cache()

    print(records)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--edited_model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--implicit_dataset_name", type=str, required=True)
    args = parser.parse_args()

    BASE_MODEL_PATH = args.base_model_name
    MODEL_PATH = args.edited_model_name
    DATASET_PATH = args.dataset_name
    IMPLICIT_DATASET_PATH = args.implicit_dataset_name

    run_final_evaluation()
