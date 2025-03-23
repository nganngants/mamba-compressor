from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import torch
import logging

def prepare_input(
    mamba_model,
    llm_model: AutoModelForCausalLM,
    llm_tokenizer: AutoTokenizer,
    system_prompt: str,
    input_texts: List[str],
    device: str = 'cuda',
    end_sym: str = '\n'
):
    # Get Mamba memory features
    input_ids = llm_tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512
    )
    
    actual_model = mamba_model.module if hasattr(mamba_model, 'module') else mamba_model
    memory_features = actual_model(input_ids)

    logging.info(f"Memory features device: {memory_features.device}")
    
    atts_memory = torch.ones(
        (memory_features.size(0), memory_features.size(1)),
    )

    logging.info(f"Attention mask device: {atts_memory.device}")
    
    # Combine system prompt with memory features
    system_encodings = llm_tokenizer(
        [system_prompt] * len(input_texts),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=64
    ).to(device)

    logging.info(f"System encodings device: {system_encodings.device}")
    
    # Get the actual LLM model if it's wrapped
    actual_llm = llm_model.module if hasattr(llm_model, 'module') else llm_model
    system_embeds = actual_llm.model.embed_tokens(system_encodings['input_ids'])
    
    memory_features = torch.cat([system_embeds, memory_features], dim=1)
    atts_memory = atts_memory[:, :1].expand(-1, memory_features.size(1))

    # Prepare target texts
    target_texts = [t + end_sym for t in input_texts]
    to_regress_tokens = llm_tokenizer(
        target_texts,
        truncation=True,
        return_tensors="pt",
        padding="longest",
        max_length=512,
    ).to(device)
    
    targets = to_regress_tokens.input_ids.masked_fill(
        to_regress_tokens.input_ids == llm_tokenizer.pad_token_id, -100
    )
    empty_targets = torch.ones(
        [memory_features.shape[0], memory_features.shape[1]], 
        device=device
    ).fill_(-100)
    
    targets = torch.cat([empty_targets, targets], dim=1)

    to_regress_embeds = actual_llm.model.embed_tokens(to_regress_tokens.input_ids)
    input_embeds = torch.cat([memory_features, to_regress_embeds], dim=1)
    attention_mask = torch.cat([atts_memory, to_regress_tokens.attention_mask], dim=1)

    print(f"Input Embeds device: {input_embeds.device}")
    print(f"Attention Mask device: {attention_mask.device}")
    print(f"Targets device: {targets.device}")
    return {
        'input_embeds': input_embeds,
        'attention_mask': attention_mask,
        'labels': targets
    }