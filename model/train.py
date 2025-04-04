#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import torch
import logging
import h5py
import tqdm
from pathlib import Path
from typing import Dict, List, Any

# Import the necessary modules
from model import MambaCompressor
from videollama2 import model_init

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to a JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def process_data(
    data: List[Dict[str, Any]],
    mamba_model: MambaCompressor,
    tokenizer,
    output_embeds_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8
) -> List[Dict[str, Any]]:
    """Process data and generate embeddings directly from MambaCompressor"""
    os.makedirs(output_embeds_dir, exist_ok=True)
    mamba_model.eval()
    
    actual_mamba_model = mamba_model.module if hasattr(mamba_model, 'module') else mamba_model
    
    updated_data = []
    
    for i in tqdm.tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data[i:i+batch_size]
        
        history_texts = [sample.get("history_chat_mamba", "") for sample in batch]
        dialogue_ids = [sample.get("Dialogue_ID", i+idx) for idx, sample in enumerate(batch)]
        
        valid_indices = [idx for idx, text in enumerate(history_texts) if text.strip()]
        if not valid_indices:
            updated_data.extend(batch)
            continue
            
        valid_batch = [batch[idx] for idx in valid_indices]
        valid_texts = [history_texts[idx] for idx in valid_indices]
        valid_dialogue_ids = [dialogue_ids[idx] for idx in valid_indices]
        
        with torch.no_grad():
            try:
                input_ids = tokenizer(
                    valid_texts,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                ).to(device)
                
                memory_features = actual_mamba_model(input_ids)
                
                for idx, (sample, dialogue_id) in enumerate(zip(valid_batch, valid_dialogue_ids)):
                    sample_idx = i + valid_indices[idx]
                    embeds_filename = f"hist_embeds_{dialogue_id}_{sample_idx}.h5"
                    embeds_path = os.path.join(output_embeds_dir, embeds_filename)
                    
                    sample_embeds = memory_features[idx].detach().cpu().numpy()
                    
                    with h5py.File(embeds_path, 'w') as f:
                        f.create_dataset('input_embeds', data=sample_embeds)
                    
                    sample['history_embeds_path'] = embeds_path
                    updated_data.append(sample)
                
            except Exception as e:
                logger.error(f"Error processing batch starting at index {i}: {str(e)}")
                for idx, sample in enumerate(batch):
                    if idx in valid_indices:
                        logger.warning(f"Skipping sample {i+idx} due to processing error")
                    updated_data.append(sample)
                continue
        
        for idx, sample in enumerate(batch):
            if idx not in valid_indices:
                updated_data.append(sample)
    
    return updated_data

def main():
    parser = argparse.ArgumentParser(description="Generate and save embeddings from MambaCompressor")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--output_embeds_dir", type=str, required=True, help="Directory to save embedding H5PY files")
    parser.add_argument("--mamba_model_path", type=str, required=True, help="Path to trained MambaCompressor model")
    parser.add_argument("--llm_name", type=str, required=True, help="Name or path of the LLM model (for tokenizer only)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on (cuda/cpu)")
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    os.makedirs(args.output_embeds_dir, exist_ok=True)
    
    # Load LLM and tokenizer (only need tokenizer)
    logger.info(f"Loading tokenizer from: {args.llm_name}")
    _, _, tokenizer = model_init(args.llm_name)
    
    # Add special tokens to the tokenizer
    tokenizer.add_special_tokens(
        {'additional_special_tokens': 
            [
                '<|im_start|>', 
                '<|im_end|>',
                '<history>',
                '<video>',
                '<MEM>'
            ]
        }
    )
    
    # Get memory token ID
    mem_token_id = tokenizer.convert_tokens_to_ids('<MEM>')
    
    # Load MambaCompressor
    logger.info(f"Loading MambaCompressor from: {args.mamba_model_path}")
    mamba_model = MambaCompressor.from_pretrained(
        path=args.mamba_model_path,
        device=args.device,
        tokenizer_len=len(tokenizer),
        mem_token_id=mem_token_id
    ).to(args.device)
    
    # Load input data
    logger.info(f"Loading data from: {args.input_jsonl}")
    data = load_jsonl(args.input_jsonl)
    logger.info(f"Loaded {len(data)} samples")
    
    # Process data
    logger.info("Processing data through MambaCompressor")
    updated_data = process_data(
        data=data,
        mamba_model=mamba_model,
        tokenizer=tokenizer,
        output_embeds_dir=args.output_embeds_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Save updated data
    logger.info(f"Saving processed data to: {args.output_jsonl}")
    save_jsonl(updated_data, args.output_jsonl)
    logger.info(f"Saved {len(updated_data)} samples with embedding paths")
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()