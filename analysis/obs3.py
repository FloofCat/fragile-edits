import argparse
import os
import json
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


MODEL_PATH = ""
BASE_MODEL_PATH = ""
DATASET_PATH = ""
CACHE_FILE = "3d_landscape_data.pkl"

def load_revedit(): 
    with open(DATASET_PATH, "r") as f:
        return json.load(f)

@torch.no_grad()
def get_sequence_prob(model, tokenizer, prompt, target):
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    target_ids = tokenizer(" " + target.strip(), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    
    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, prompt_ids.shape[1]-1 : -1, :] 
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    seq_log_prob = sum(log_probs[i, target_ids[0, i]].item() for i in range(target_ids.shape[1]))
        
    return np.exp(seq_log_prob)

def run_3d_landscape_experiment():    
    grid_size = 60
    alphas = np.linspace(-2.0, 2.0, grid_size) 
    betas = np.linspace(-2.0, 2.0, grid_size)
    X, Y = np.meshgrid(alphas, betas)
    
    if os.path.exists(CACHE_FILE):
        print(f"[-] Loading cached landscape data from {CACHE_FILE}...")
        with open(CACHE_FILE, "rb") as f:
            cached_data = pickle.load(f)
            Z_prob_new = cached_data["Z_prob_new"]
    else:
        print("[-] Cache not found. Computing 3D landscape...")
        data = load_revedit()
        
        print("[-] Loading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        
        print("[-] Loading Edited Model...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
        model.eval()
        
        base_state = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16).state_dict()
        edited_layers = [f"transformer.h.{i}.mlp.fc_out.weight" for i in [3, 4, 5, 6, 7, 8]] 
        
        print("[-] Constructing Orthogonal Subspaces...")
        landscape_axes = {}
        for layer in edited_layers:
            edit_w = model.state_dict()[layer].float()
            base_w = base_state[layer].to(model.device).float()
            
            delta_w = edit_w - base_w
            
            torch.manual_seed(42)
            N = torch.randn_like(delta_w)
            
            dot_product = torch.sum(N * delta_w)
            norm_sq = torch.sum(delta_w * delta_w)
            N_perp = N - (dot_product / norm_sq) * delta_w
            N_perp = N_perp * (torch.norm(delta_w) / torch.norm(N_perp))
            
            landscape_axes[layer] = {
                'base': edit_w.half(), 
                'delta_w': delta_w.half(),
                'n_perp': N_perp.half()
            }
            
        del base_state 
        torch.cuda.empty_cache()

        Z_prob_new = np.zeros((grid_size, grid_size))

        print("[-] Sweeping 2D Parameter Grid for 3D Plot...")
        with tqdm(total=grid_size*grid_size) as pbar:
            for i in range(grid_size):
                for j in range(grid_size):
                    alpha = alphas[i]
                    beta = betas[j]
                    
                    for layer in edited_layers:
                        axes = landscape_axes[layer]
                        new_w = axes['base'] + (alpha * axes['delta_w']) + (beta * axes['n_perp'])
                        model.state_dict()[layer].copy_(new_w)
                        
                    prob_sum = sum(get_sequence_prob(model, tokenizer, row["paraphrases"][0].strip(), row["new_edit"].strip()) for row in data)
                    Z_prob_new[j, i] = prob_sum / len(data) 
                    
                    for layer in edited_layers:
                        model.state_dict()[layer].copy_(landscape_axes[layer]['base'])
                        
                    pbar.update(1)
                    
        print(f"[-] Saving landscape data to {CACHE_FILE}...")
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"Z_prob_new": Z_prob_new}, f)

    fig = plt.figure(figsize=(13, 7)) 
    
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    
    surf = ax.plot_surface(
    X, Y, Z_prob_new,
    cmap='RdBu_r',
    edgecolor='none',
    antialiased=True,
    alpha=0.95
)
    
    a0_idx = np.abs(alphas - 0.0).argmin()
    b0_idx = np.abs(betas - 0.0).argmin()

    center_z = Z_prob_new[b0_idx, a0_idx]
    ax.scatter(0, 0, center_z + 0.02, color='yellow', edgecolor='black', s=350, marker='*', 
               label='Edited Fact ($W_{edit}$)', zorder=5)
    
    ax.set_xlabel(r'Edit Direction ($\alpha \Delta W$)', labelpad=18)
    ax.set_ylabel(r'Orthogonal Dir. ($\beta N_{\perp}$)', labelpad=18)
    ax.set_zlabel('Generation Prob. ($o_{new}$)', labelpad=15)
    
    ax.tick_params(axis='both', pad=8)
    ax.tick_params(axis='z', pad=8)
    
    ax.view_init(elev=20, azim=-55)
    
    # Clean colorbar
    cbar = fig.colorbar(surf, shrink=0.7, aspect=20, pad=0.08)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Probability', rotation=270, labelpad=25)
    
    # Place legend outside the main 3D box
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), framealpha=0.9)
    
    # Add distance to push subplots away from edges
    plt.subplots_adjust(left=0.05, right=0.88, top=0.95, bottom=0.05)
    
    plt.savefig('obs3.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("[-] Saved plot to obs3.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--edited_model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    BASE_MODEL_PATH = args.base_model_name
    MODEL_PATH = args.edited_model_name
    DATASET_PATH = args.dataset_name

    run_3d_landscape_experiment()