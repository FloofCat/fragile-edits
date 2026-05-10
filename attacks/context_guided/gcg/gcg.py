import logging
import random
import os
import pickle
import copy
import gc

from dataclasses import dataclass, field
from tqdm import tqdm
from typing import List, Optional, Union
from itertools import count
import numpy as np
import matplotlib.pyplot as plt

import torch
import transformers
from torch import Tensor
from transformers import set_seed
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast

from gcg.utils import find_executable_batch_size

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class GCGConfig:
    init: List[int] = field(default_factory=lambda: [87, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865, 865])
    num_steps: int = 1500
    search_width: int = 64
    topk: int = 256
    n_replace: int = 1
    early_stop: bool = True
    seed: int = None
    verbosity: str = "INFO"
    grad_batch_size: int = 512
    base_folder_location: str = "./attacks/context_guided/"

@dataclass
class GCGResult:
    ranking_loss: float
    best_tensor: Tensor
    losses: List[float]
    tensors: List[Tensor]

def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids

class GCG:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.history_before_ids_list = [] 
        self.history_after_ids_list = []
        self.history_target_ids_list = []

        self.embedding_layer = self.model.get_input_embeddings()

        if self.model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {self.model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if self.model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not tokenizer.chat_template:
            logger.warning("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def setup(self,
               messages: List[str],
               old_edits: List[str],
               tokenizer: transformers.PreTrainedTokenizer,
               compute_device: torch.device):
        if self.history_after_ids_list != []:
            return
        
        # If setup.pkl exists, load it
        if os.path.exists(self.config.base_folder_location + "/logs/setup.pkl"):
            with open(self.config.base_folder_location + "/logs/setup.pkl", "rb") as f:
                setup_data = pickle.load(f)
                self.history_before_ids_list = setup_data["before_ids_list"]
                self.history_after_ids_list = setup_data["after_ids_list"]
                self.history_target_ids_list = setup_data["target_ids_list"]
            logger.info(f"Loaded setup from existing file with {len(self.history_before_ids_list)} prompts.")
            return

        for msg, target in tqdm(zip(messages, old_edits), desc="Preparing prompts"):
            if "{optim_str}" not in msg:
                msg = msg + "{optim_str}"

            current_messages_formatted = [{"role": "user", "content": msg}]
            template = tokenizer.apply_chat_template(current_messages_formatted, tokenize=False, add_generation_prompt=True)
            
            if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
                template = template.replace(tokenizer.bos_token, "")
            
            before_str, after_str = template.split("{optim_str}")

            before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(compute_device, torch.int64)
            after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(compute_device, torch.int64)
            target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(compute_device, torch.int64)

            self.history_before_ids_list.append(before_ids.cpu())
            self.history_after_ids_list.append(after_ids.cpu())
            self.history_target_ids_list.append(target_ids.cpu())


        logger.info(f"Prepared {len(self.history_before_ids_list)} prompts for optimization.")

        with open(self.config.base_folder_location + "/logs/setup.pkl", "wb") as f:
            pickle.dump({
                "before_ids_list": self.history_before_ids_list,
                "after_ids_list": self.history_after_ids_list,
                "target_ids_list": self.history_target_ids_list,
            }, f)

    def reset(self):
        self.history_before_ids_list = []
        self.history_after_ids_list = []
        self.history_target_ids_list = []

    def run(
        self,
        messages: List[str],
        old_edits: List[str],
        top_k: int = 20,
    ) -> GCGResult:
        tokenizer = self.tokenizer
        config = self.config
        self.top_k = top_k
        self.original_prompts = messages

        compute_device = self.model.device

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        logger.warning(f"Processing {len(messages)} prompts for universal suffix generation.")

        self.setup(messages, old_edits, tokenizer, compute_device)

        optim_ids = torch.tensor(config.init, device=compute_device, dtype=torch.long).unsqueeze(0)

        # Data.
        best_suffix = optim_ids.clone()

        best_ranking_loss = float('inf')

        global_ranking_losses, optim_strings = [], []
        epsilon = 0.2

        if config.num_steps == None:
            iterator = count()
        else:
            iterator = range(config.num_steps)

        for step in tqdm(iterator):
            sampled_indices = random.sample(range(len(self.history_before_ids_list)), len(self.history_before_ids_list))
            self.before_ids_list = [self.history_before_ids_list[i].to(compute_device) for i in sampled_indices]
            self.after_ids_list = [self.history_after_ids_list[i].to(compute_device) for i in sampled_indices]

            # Compute the token gradient
            optim_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():
                candidate_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_grad.squeeze(0),
                    search_width=config.search_width,
                    topk=self.top_k,
                    n_replace=config.n_replace,
                )


                print(f"Generated {candidate_ids.shape[0]} candidates.")
                losses = find_executable_batch_size(self._get_properties_batch, self.config.batch_size)(candidate_ids)

                best_candidate_idx = losses.argmin()

                optim_ids = candidate_ids[best_candidate_idx].unsqueeze(0)

                # Find the best candidate according to the combined loss
                current_loss = losses[best_candidate_idx]

                print(f'Loss: {current_loss.item():.4f}')

                if current_loss.item() <= best_ranking_loss:
                    best_suffix = optim_ids.clone()
                    best_ranking_loss = current_loss.item()

                    print("✨ New best loss found.")

                if current_loss.item() == 0:
                    if config.early_stop:
                        print("Early stopping as the perfect solution is found.")
                        break
                else:
                    print(f"No perfect solution found. Adopting best effort.")
                
                # Logging and storing results
                global_ranking_losses.append(current_loss.item())

                optim_strings.append(optim_ids.squeeze(0))

                print(f"Step {step+1}/{config.num_steps} | Ranking Loss: {current_loss.item():.4f}\n")

        # Save the final_optim_ids (embeddings of the best suffix)
        best_suffix = best_suffix.squeeze(0)
        self.reset()

        return GCGResult(
            ranking_loss=best_ranking_loss,
            best_tensor=best_suffix.squeeze(0),
            losses=global_ranking_losses,
            tensors=optim_strings,
        )
    
    def compute_token_gradient(self, optim_ids: Tensor) -> Tensor:
        num_prompts = len(self.before_ids_list)
        device = self.model.device
        dtype = self.model.dtype
        
        optim_ids_onehot = torch.nn.functional.one_hot(
            optim_ids, num_classes=self.embedding_layer.num_embeddings
        ).to(dtype)
        optim_ids_onehot.requires_grad_()
        
        total_grad = torch.zeros_like(optim_ids_onehot)

        for i in range(0, num_prompts, self.config.grad_batch_size):
            optim_embeds = optim_ids_onehot @ self.embedding_layer.weight
            optim_len = optim_embeds.shape[1]

            batch_slice = slice(i, i + self.config.grad_batch_size)
            target_ids_batch = self.target_ids[batch_slice]

            before_embeds_batch = [self.embedding_layer(b_ids) for b_ids in self.before_ids_list[batch_slice]]
            after_embeds_batch = [self.embedding_layer(a_ids) for a_ids in self.after_ids_list[batch_slice]]

            current_batch_size = len(before_embeds_batch)
            expanded_optim_embeds = optim_embeds.expand(current_batch_size, -1, -1)

            squeezed_before = [e.squeeze(0) for e in before_embeds_batch]
            squeezed_after = [e.squeeze(0) for e in after_embeds_batch]
            padded_before = pad_sequence(squeezed_before, batch_first=True, padding_value=0.0)
            padded_after = pad_sequence(squeezed_after, batch_first=True, padding_value=0.0)
            
            model_input = torch.cat([padded_before, expanded_optim_embeds, padded_after], dim=1)
            
            before_lens = torch.tensor([e.shape[1] for e in before_embeds_batch], device=device)
            after_lens = torch.tensor([e.shape[1] for e in after_embeds_batch], device=device)
            actual_lengths = before_lens + optim_len + after_lens
            
            attention_mask = torch.arange(model_input.shape[1], device=device)[None, :] < actual_lengths[:, None]
            attention_mask = attention_mask.long()

            # Batched Forward Pass
            outputs = self.model(inputs_embeds=model_input, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            batch_loss = 0.0
            
            target_lens = [t.shape[1] for t in target_ids_batch]
            
            for k in range(current_batch_size):
                # The token predicting the first target token is the last token of the suffix (optim).
                # Logits at position `i` predict token at `i+1`.
                start_idx = before_lens[k] + optim_len - 1
                end_idx = start_idx + target_lens[k]
                
                # Extract logits that predict the target tokens
                tok_logits = logits[k, start_idx:end_idx, :]
                tok_labels = target_ids_batch[k].squeeze(0).to(device)
                
                # Compute loss for this prompt (mean over target tokens)
                seq_loss = loss_fct(tok_logits, tok_labels).mean()
                batch_loss += seq_loss

            # Average loss over the batch and backpropagate
            batch_loss = batch_loss / current_batch_size
            batch_loss.backward()

            total_grad += optim_ids_onehot.grad.clone()
            optim_ids_onehot.grad.zero_()
            
            # Clean up memory
            del model_input, outputs, logits, ranking_loss, batch_grad
            torch.cuda.empty_cache()

        total_grad /= num_prompts
            
        # return total_grad
        return total_grad
    
    def _get_properties_batch(self, search_batch_size: int, candidate_ids: Tensor):
        """
        Computes ranking loss and deltas for a batch of candidates, averaged over all prompts.
        This version processes prompts in batches for efficiency.
        """
        num_candidates = candidate_ids.shape[0]
        num_prompts = len(self.before_ids_list)
        device = candidate_ids.device
        dtype = self.model.module.dtype if isinstance(self.model, torch.nn.DataParallel) else self.model.dtype
        
        # Tensors to store the sum of losses/deltas for each candidate
        total_ranking_losses = torch.zeros(num_candidates, device=device, dtype=dtype)
        
        # Pre-compute candidate embeddings once
        candidate_embeds = self.embedding_layer(candidate_ids)
        
        # Iterate over all prompts in manageable batches
        for i in range(0, num_prompts, self.config.grad_batch_size):
            # Get slices for the current batch of prompts
            batch_slice = slice(i, i + self.config.grad_batch_size)
            before_embeds_batch = [self.embedding_layer(b_ids) for b_ids in self.before_ids_list[batch_slice]]
            after_embeds_batch = [self.embedding_layer(a_ids) for a_ids in self.after_ids_list[batch_slice]]
            
            target_ids_batch = torch.tensor(self.target_ids[batch_slice], device=device, dtype=torch.long)
            
            current_prompt_batch_size = len(before_embeds_batch)
            
            # Process candidates in mini-batches for this batch of prompts
            for j in range(0, num_candidates, search_batch_size):
                cand_slice = slice(j, j + search_batch_size)
                embeds_batch = candidate_embeds[cand_slice]
                current_cand_batch_size = embeds_batch.shape[0]
                
                # Pad and combine embeds
                squeezed_before = [e.squeeze(0) for e in before_embeds_batch]
                squeezed_after = [e.squeeze(0) for e in after_embeds_batch]
                padded_before = pad_sequence(squeezed_before, batch_first=True, padding_value=0.0)
                padded_after = pad_sequence(squeezed_after, batch_first=True, padding_value=0.0)
                
                expanded_before = padded_before.repeat_interleave(current_cand_batch_size, dim=0)
                expanded_after = padded_after.repeat_interleave(current_cand_batch_size, dim=0)
                expanded_candidates = embeds_batch.repeat(current_prompt_batch_size, 1, 1)
                
                input_embeds_batch = torch.cat([expanded_before, expanded_candidates, expanded_after], dim=1)

                # Compute attention mask
                before_lens = torch.tensor([e.shape[1] for e in before_embeds_batch], device=device)
                after_lens = torch.tensor([e.shape[1] for e in after_embeds_batch], device=device)
                optim_len = embeds_batch.shape[1]
                actual_lengths = before_lens + optim_len + after_lens

                expanded_actual_lengths = actual_lengths.repeat_interleave(current_cand_batch_size, dim=0)

                total_seq_len = input_embeds_batch.shape[1]
                attention_mask = torch.arange(total_seq_len, device=device)[None, :] < expanded_actual_lengths[:, None]
                attention_mask = attention_mask.long()
                
                
                with torch.no_grad():
                    all_logits = self.model(inputs_embeds=input_embeds_batch, attention_mask=attention_mask).logits
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                cand_losses = torch.zeros(current_cand_batch_size, device=device, dtype=dtype)
                
                # Reshape all_logits: (prompt_batch_size, cand_batch_size, seq_len, vocab_size)
                all_logits = all_logits.view(current_prompt_batch_size, current_cand_batch_size, total_seq_len, -1)
                
                target_lens = [t.shape[1] for t in target_ids_batch]
                
                for k in range(current_prompt_batch_size):
                    start_idx = before_lens[k] + optim_len - 1
                    end_idx = start_idx + target_lens[k]
                    
                    # Logits predicting the target for all candidates: (cand_batch_size, target_len, vocab_size)
                    tok_logits = all_logits[k, :, start_idx:end_idx, :]
                    tok_labels = target_ids_batch[k].squeeze(0).to(device) # Shape: (target_len,)
                    
                    # Flatten to (cand_batch_size * target_len, vocab_size) to use CrossEntropyLoss
                    flat_logits = tok_logits.reshape(-1, tok_logits.shape[-1])
                    flat_labels = tok_labels.unsqueeze(0).expand(current_cand_batch_size, -1).reshape(-1)
                    
                    # Loss per token: (cand_batch_size * target_len)
                    loss = loss_fct(flat_logits, flat_labels)
                    
                    # Reshape back to (cand_batch_size, target_len) and take mean over target length
                    loss = loss.view(current_cand_batch_size, -1).mean(dim=1)
                    
                    # Accumulate loss across all prompts
                    cand_losses += loss
                    
                total_ranking_losses[cand_slice] += cand_losses
                
                # Clean up intermediate tensors
                del input_embeds_batch, all_logits, tok_logits, flat_logits, loss
                torch.cuda.empty_cache()
                
        # Average the losses over all prompts
        final_ranking_losses = total_ranking_losses / num_prompts
                
        return final_ranking_losses


# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: List[str],
    old_edits: List[str],
    top_k: int = 20,
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = GCG(model, tokenizer, config)
    result = gcg.run(messages, old_edits, top_k=top_k)

    with open(config.base_folder_location + "/logs/final_result.pkl", "wb") as f:
        pickle.dump(result, f)



# =================================================================