import argparse
import os
import json
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_PATH = ""
BASE_MODEL_PATH = ""
DATASET_PATH = ""
PKL_FILE_PATH = "plot_data.pkl"

def load_revedit(): 
    with open(DATASET_PATH, "r") as f:
        return json.load(f)

def compute_and_save_data():
    data = load_revedit()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    base_model.eval()
    
    print("[-] Loading Edited Model...")
    edit_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    edit_model.eval()

    W_U = base_model.lm_head.weight.data.float()

    # Storage
    full_vocab_logits = []
    ablated_vocab_logits = []

    full_old_logits, full_new_logits = [], []
    abl_old_logits, abl_new_logits = [], []

    with torch.no_grad():
        for row in tqdm(data):
            prompt = row["paraphrases"][0].strip()

            o_old = " " + row["old_edit"].strip()
            o_new = " " + row["new_edit"].strip()

            o_old_token = tokenizer(o_old, add_special_tokens=False).input_ids[0]
            o_new_token = tokenizer(o_new, add_special_tokens=False).input_ids[0]

            inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

            base_outputs = base_model(**inputs, output_hidden_states=True)
            h_base = base_outputs.hidden_states[-1][0, -1, :].float()
            h_base_norm = base_model.transformer.ln_f(h_base.half()).float()

            edit_outputs = edit_model(**inputs, output_hidden_states=True)
            h_edit = edit_outputs.hidden_states[-1][0, -1, :].float()
            h_edit_norm = edit_model.transformer.ln_f(h_edit.half()).float()

            delta_h = h_edit_norm - h_base_norm

            z_full = torch.matmul(W_U, delta_h)

            w_old = W_U[o_old_token] 
            w_old_unit = w_old / (w_old.norm() + 1e-8)

            proj_scalar = torch.dot(delta_h, w_old_unit)
            proj_vec = proj_scalar * w_old_unit

            delta_h_no_old = delta_h - proj_vec
            z_abl = torch.matmul(W_U, delta_h_no_old)

            full_vocab_logits.extend(z_full.cpu().numpy()[::100])
            ablated_vocab_logits.extend(z_abl.cpu().numpy()[::100])

            full_old_logits.append(z_full[o_old_token].item())
            full_new_logits.append(z_full[o_new_token].item())

            abl_old_logits.append(z_abl[o_old_token].item())
            abl_new_logits.append(z_abl[o_new_token].item())

    # Package and save
    plot_data = {
        "full_vocab_logits": full_vocab_logits,
        "ablated_vocab_logits": ablated_vocab_logits,
        "full_old_logits": full_old_logits,
        "full_new_logits": full_new_logits,
        "abl_old_logits": abl_old_logits,
        "abl_new_logits": abl_new_logits
    }
    
    with open(PKL_FILE_PATH, "wb") as f:
        pickle.dump(plot_data, f)
        
    print(f"[-] Saved compute results to {PKL_FILE_PATH}")

def load_and_plot_data():
    print(f"[-] Loading data from {PKL_FILE_PATH}...")
    with open(PKL_FILE_PATH, "rb") as f:
        data = pickle.load(f)
        
    fig, ax = plt.subplots()

    sns.kdeplot(data["full_vocab_logits"], fill=True, alpha=0.35, label='Original Background', ax=ax)
    sns.kdeplot(data["ablated_vocab_logits"], fill=True, alpha=0.1, label='Ablated Background', ax=ax)

    # Means
    mean_full_old = np.mean(data["full_old_logits"])
    mean_full_new = np.mean(data["full_new_logits"])
    mean_abl_old = np.mean(data["abl_old_logits"])
    mean_abl_new = np.mean(data["abl_new_logits"])

    ax.axvline(mean_full_old, linestyle='--', label=r'Orig $o_{old}$')
    ax.axvline(mean_abl_old, linestyle=':', label=r'Ablated $o_{old}$')
    
    ax.axvline(mean_full_new, linestyle='--', label=r'Orig $o_{new}$')
    ax.axvline(mean_abl_new, linestyle=':', label=r'Ablated $o_{new}$')

    scatter_y = -0.003
    ax.scatter(data["full_old_logits"], np.full_like(data["full_old_logits"], scatter_y), s=100, marker='s', zorder=5)
    ax.scatter(data["abl_old_logits"], np.full_like(data["abl_old_logits"], scatter_y), s=100, marker='s', alpha=0.6, zorder=5)
    
    ax.scatter(data["full_new_logits"], np.full_like(data["full_new_logits"], scatter_y), s=100, marker='o', zorder=5)
    ax.scatter(data["abl_new_logits"], np.full_like(data["abl_new_logits"], scatter_y), s=100, marker='o', alpha=0.6, zorder=5)

    y_max = ax.get_ylim()[1]
    arrow_y = y_max * 0.70 
    
    x_offset_old = 1.5 
    x_offset_new = 1.0

    ax.annotate('', xy=(mean_abl_old, arrow_y), xytext=(mean_full_old, arrow_y),
                arrowprops=dict(shrink=0.02, width=3.5, headwidth=12, edgecolor='none'))
    
    ax.text(mean_abl_old + x_offset_old, arrow_y, 'Penalty\nRemoved', 
            ha='left', va='center', fontsize=15, fontweight='bold')

    ax.annotate('', xy=(mean_abl_new, arrow_y), xytext=(mean_full_new, arrow_y),
                arrowprops=dict(shrink=0.02, width=3.5, headwidth=12, edgecolor='none'))
    
    ax.text(mean_abl_new + x_offset_new, arrow_y, 'Promotion\nIncreased', 
            ha='left', va='center', fontsize=15, fontweight='bold')

    ax.set_xlabel(r'Intervention Logits ($W_U \Delta h$)', weight='bold')
    ax.set_ylabel('Density', weight='bold')
    
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.axvline(0, color='black', linewidth=2.0, alpha=0.8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('obs2.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("[-] Saved plot to obs2.pdf")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Full old logit: {mean_full_old:.3f}")
    print(f"Ablated old logit: {mean_abl_old:.3f}")
    print(f"Full new logit: {mean_full_new:.3f}")
    print(f"Ablated new logit: {mean_abl_new:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--edited_model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    BASE_MODEL_PATH = args.base_model_name
    MODEL_PATH = args.edited_model_name
    DATASET_PATH = args.dataset_name

    if not os.path.exists(PKL_FILE_PATH):
        compute_and_save_data()
    load_and_plot_data()