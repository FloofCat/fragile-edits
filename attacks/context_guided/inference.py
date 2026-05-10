import argparse
from gcg.gcg import GCGResult, GCGConfig
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_inference_attack(result: GCGResult,
                              model: AutoModelForCausalLM,
                              tokenizer: AutoTokenizer,
                              surrogate_tokenizer: AutoTokenizer,
                              config: GCGConfig,
                              dataset_path: str):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    data = data[60:]

    prompts = []
    old_edits = []

    for item in data:
        prompts.append(item["paraphrases"][0])
        old_edits.append(item["old_edit"])

    best_ids = result.best_tensor.long()
    suffix_str = surrogate_tokenizer.decode(best_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    success_count = 0
    total_count = len(prompts)
    results_log = []

    model.eval()
    compute_device = model.device

    for prompt, old_edit in tqdm(zip(prompts, old_edits), total=total_count, desc="Evaluating Elicitation Attack"):
        full_user_content = f"{prompt}{suffix_str}"

        messages = [{"role": "user", "content": full_user_content}]
        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(compute_device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=15, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        input_length = input_ids.shape[1]
        generated_ids = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        is_successful = old_edit.lower() in generated_text.lower()
        
        if is_successful:
            success_count += 1

        results_log.append({
            "prompt": prompt,
            "target_old_edit": old_edit,
            "generated_text": generated_text,
            "success": is_successful
        })

    asr = (success_count / total_count) * 100 if total_count > 0 else 0.0

    with open(config.base_folder_location + "/logs/results.json", "w") as f:
        json.dump(results_log, f, indent=4)

    print(f"Attack Success Rate: {asr:.2f}%")

def main(config: GCGConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--surrogate_model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    
    surrogate_model_name = args.surrogate_model_name
    surrogate_tokenizer = AutoTokenizer.from_pretrained(surrogate_model_name, trust_remote_code=True, use_fast=False)
    
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # torch_dtype=torch.float16,
                trust_remote_code=True
            ).to("cuda:0").eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    with open(config.base_folder_location + "/logs/final_result.pkl", "rb") as f:
        result = torch.load(f)

    evaluate_inference_attack(
        result=result,
        model=model,
        tokenizer=tokenizer,
        surrogate_tokenizer=surrogate_tokenizer,
        config=config,
        dataset_path=args.dataset_name
    )
