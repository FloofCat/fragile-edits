import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


MODEL_NAME = "" 
EDIT_LAYER = 6 # Tested with 3-9.
RANK = 4
SCALE = 10.0

def inject_rank_r_update(model, tokenizer, prompt, layer_idx, rank, scale):
    """
    Constructs delta_W = U * V^T using the actual data manifold (SVD)
    so the edit sits perfectly in the model's standard reasoning path.
    """
    target_module = model.transformer.h[layer_idx].mlp.fc_in
    
    activations = {}
    def hook_fn(module, input, output):
        activations['x'] = input[0].detach()
        
    handle = target_module.register_forward_hook(hook_fn)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    
    x = activations['x'].squeeze(0) # [seq_len, d_in]
    
    x_centered = x - x.mean(dim=0)
    U_svd, S_svd, V_svd = torch.svd(x_centered.float())
    
    V = V_svd[:, :rank].to(model.dtype) # The row space we want to bypass [d_in, rank]
    d_out = target_module.weight.shape[0]
    U = torch.randn(d_out, rank, device=model.device, dtype=model.dtype) # Target shift [d_out, rank]
    
    delta_W = scale * (U @ V.T) / rank
    target_module.weight.data = target_module.weight.data + delta_W
    
    return delta_W, V

def measure_theorem_components(model, input_ids, delta_W, row_space_V, layer_idx):
    """
    Explicitly calculates v_parallel (interference) and v_perp (null space).
    """
    target_module = model.transformer.h[layer_idx].mlp.fc_in
    activations = {}
    
    def hook_fn(module, input, output):
        activations['h'] = input[0].detach() # Hidden state BEFORE the weight matrix
        
    handle = target_module.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids=input_ids)
    handle.remove()
    
    # Get representation of the last token
    h = activations['h'][0, -1, :] 
    
    # Total Interference: || \Delta W * h ||
    interference = torch.norm(delta_W @ h).item()
    
    # Projection into the Row Space (v_parallel)
    # V is orthonormal basis of the row space. Projection = V @ V^T @ h
    v_parallel = row_space_V @ (row_space_V.T @ h)
    norm_v_parallel = torch.norm(v_parallel).item()
    
    # Projection into Null Space (v_perp)
    v_perp = h - v_parallel
    norm_v_perp = torch.norm(v_perp).item()
    
    return interference, norm_v_parallel, norm_v_perp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    MODEL_NAME = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto") 

    total_std_interf, total_adv_interf = 0.0, 0.0
    total_std_v_para, total_adv_v_para = 0.0, 0.0
    total_std_v_perp, total_adv_v_perp = 0.0, 0.0

    with open(args.dataset_name, "r") as f:
        data = json.load(f)
    
    for item in data:
        PROMPT = item["paraphrases"][0]
    
        input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
    
        with torch.no_grad():
            base_logits = model(input_ids).logits
            
        import copy
        edited_model = copy.deepcopy(model)
        delta_W, row_space_V = inject_rank_r_update(edited_model, tokenizer, PROMPT, EDIT_LAYER, RANK, SCALE)

        with open("./attacks/context_guided/logs/final_result.pkl", "rb") as f:
            result = torch.load(f)
    
        suffix_ids = result.best_tensor.unsqueeze(0).to(model.device)
        adv_input_ids = torch.cat([input_ids, suffix_ids], dim=1)
    
        # Metric 1: The standard prompt passing through the edited model
        std_interf, std_v_para, std_v_perp = measure_theorem_components(edited_model, input_ids, delta_W, row_space_V, EDIT_LAYER)
        
        # Metric 2: The adversarial prompt passing through the edited model
        adv_interf, adv_v_para, adv_v_perp = measure_theorem_components(edited_model, adv_input_ids, delta_W, row_space_V, EDIT_LAYER)

        total_std_interf += std_interf
        total_adv_interf += adv_interf
        total_std_v_para += std_v_para
        total_adv_v_para += adv_v_para
        total_std_v_perp += std_v_perp
        total_adv_v_perp += adv_v_perp 
    
    num_samples = len(data)
    std_interf = total_std_interf / num_samples
    adv_interf = total_adv_interf / num_samples
    std_v_para = total_std_v_para / num_samples
    adv_v_para = total_adv_v_para / num_samples
    std_v_perp = total_std_v_perp / num_samples
    adv_v_perp = total_adv_v_perp / num_samples

    print("-" * 60)
    print("                Standard Prompt  |  Adversarial Suffix")
    print("-" * 60)
    print(f"v_parallel norm : {std_v_para:14.4f} | {adv_v_para:14.4f}")
    print(f"v_perp norm     : {std_v_perp:14.4f} | {adv_v_perp:14.4f}")
    print("-" * 60)
    print(f"Interference    : {std_interf:14.4f} | {adv_interf:14.4f}")
    print("-" * 60)