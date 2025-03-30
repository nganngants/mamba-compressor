import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List


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
        # padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=100
    )
    
    # Get the model dtype from the LLM model
    model_dtype = next(llm_model.parameters()).dtype

    # Get memory features from mamba model and ensure it's in the correct dtype
    memory_features = mamba_model(input_ids)
    memory_features = memory_features.to(dtype=model_dtype)
    
    atts_memory = torch.ones(
        (memory_features.size(0), memory_features.size(1)),
        dtype=torch.long,
    )
    
    # Combine system prompt with memory features
    system_encodings = llm_tokenizer(
        [system_prompt] * len(input_texts),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=16
    )
    system_embeds = llm_model.get_input_embeddings(system_encodings['input_ids'].to(device)) # (batch, seq, hidden)
    
    # Ensure consistent dtype
    system_embeds = system_embeds.to(dtype=model_dtype)
    
    memory_features = torch.cat([system_embeds, memory_features], dim=1)
    atts_memory = atts_memory[:, :1].expand(-1, memory_features.size(1))

    # Prepare target texts
    target_texts = [t + end_sym for t in input_texts]
    to_regress_tokens = llm_tokenizer(
        target_texts,
        truncation=True,
        return_tensors="pt",
        max_length=100,
    )
    targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == llm_tokenizer.pad_token_id, -100
            )
    empty_targets = (
                torch.ones([memory_features.shape[0], memory_features.shape[1]],
                        dtype=torch.long).fill_(-100)
            )
    targets = torch.cat([empty_targets, targets], dim=1)

    to_regress_embeds = llm_model.model.embed_tokens(to_regress_tokens.input_ids.to(device))
    
    # Ensure consistent dtype
    to_regress_embeds = to_regress_embeds.to(dtype=model_dtype)
    
    input_embeds = torch.cat([memory_features, to_regress_embeds], dim=1)
    attention_mask = torch.cat([atts_memory, to_regress_tokens.attention_mask], dim=1)

    return {
        'input_embeds': input_embeds,
        'attention_mask': attention_mask,
        'labels': targets,
    }