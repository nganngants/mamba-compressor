#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import torch
import logging
import h5py
import tqdm
import traceback
from pathlib import Path
from typing import Dict, List, Any

# Import the necessary modules
from model import MambaCompressor
from videollama2 import model_init

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
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

def process_sample(
    sample: Dict[str, Any],
    sample_idx: int,
    mamba_model: MambaCompressor,
    tokenizer,
    output_embeds_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    debug: bool = False
) -> Dict[str, Any]:
    """Process a single sample to generate embeddings"""
    
    history_text = sample.get("history_chat_mamba", "")
    dialogue_id = sample.get("Dialogue_ID", sample_idx)
    
    # Skip if no valid history text
    if not history_text or not history_text.strip():
        if debug:
            logger.info(f"Sample {sample_idx} has no valid history text")
        return sample
    
    try:
        # Tokenize the input text
        input_ids = tokenizer(
            [history_text],  # Single sample in a list
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(device)
        
        # Process through MambaCompressor
        with torch.no_grad():
            memory_features = mamba_model(input_ids)
        
        # Create a unique filename for this sample
        embeds_filename = f"hist_embeds_{dialogue_id}_{sample_idx}.h5"
        embeds_path = os.path.join(output_embeds_dir, embeds_filename)
        
        # Extract embeddings for this sample
        sample_embeds = memory_features[0].detach().cpu().numpy()  # First (only) item in batch
        
        # Log shape for debugging
        if debug:
            logger.info(f"Sample {sample_idx} - Embedding shape: {sample_embeds.shape}")
        
        # Save embeddings to H5PY file
        with h5py.File(embeds_path, 'w') as f:
            f.create_dataset('input_embeds', data=sample_embeds)
        
        # Update sample with the path to embeddings
        sample['history_embeds_path'] = embeds_path
        return sample
        
    except Exception as e:
        logger.error(f"Error processing sample {sample_idx}: {str(e)}")
        if debug:
            logger.error(traceback.format_exc())
        return sample

def process_data(
    data: List[Dict[str, Any]],
    mamba_model: MambaCompressor,
    tokenizer,
    output_embeds_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    debug: bool = False
) -> List[Dict[str, Any]]:
    """Process data one sample at a time to avoid batch size issues"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_embeds_dir, exist_ok=True)
    
    # Set model to evaluation mode
    mamba_model.eval()
    
    # Get the actual Mamba model if it's wrapped
    actual_mamba_model = mamba_model.module if hasattr(mamba_model, 'module') else mamba_model
    
    # Process each sample individually
    updated_data = []
    success_count = 0
    
    for idx, sample in enumerate(tqdm.tqdm(data, desc="Processing samples")):
        updated_sample = process_sample(
            sample=sample,
            sample_idx=idx,
            mamba_model=actual_mamba_model,
            tokenizer=tokenizer,
            output_embeds_dir=output_embeds_dir,
            device=device,
            debug=debug
        )
        
        # Check if embeddings were successfully generated
        if 'history_embeds_path' in updated_sample:
            success_count += 1
            
        updated_data.append(updated_sample)
    
    # Log final counts
    logger.info(f"Total samples: {len(data)}")
    logger.info(f"Successfully saved embeddings: {success_count}")
    
    # Verify output directory has the expected number of files
    h5_files = [f for f in os.listdir(output_embeds_dir) if f.endswith('.h5')]
    logger.info(f"H5 files in output directory: {len(h5_files)}")
    
    return updated_data

def main():
    parser = argparse.ArgumentParser(description="Generate and save embeddings from MambaCompressor")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--output_embeds_dir", type=str, required=True, help="Directory to save embedding H5PY files")
    parser.add_argument("--mamba_model_path", type=str, required=True, help="Path to trained MambaCompressor model")
    parser.add_argument("--llm_name", type=str, required=True, help="Name or path of the LLM model (for tokenizer only)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    os.makedirs(args.output_embeds_dir, exist_ok=True)
    
    # Check if output directory is writable
    try:
        test_file = os.path.join(args.output_embeds_dir, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        logger.info(f"Output directory {args.output_embeds_dir} is writable")
    except Exception as e:
        logger.error(f"Cannot write to output directory {args.output_embeds_dir}: {str(e)}")
        return
    
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
    
    # Check for history_chat_mamba presence
    has_history = [bool(sample.get("history_chat_mamba", "").strip()) for sample in data]
    logger.info(f"Samples with non-empty history_chat_mamba: {sum(has_history)}/{len(data)}")
    
    if sum(has_history) == 0:
        logger.error("No samples have history_chat_mamba field. Please check your input data.")
        return
    
    # Process data
    logger.info("Processing data through MambaCompressor")
    updated_data = process_data(
        data=data,
        mamba_model=mamba_model,
        tokenizer=tokenizer,
        output_embeds_dir=args.output_embeds_dir,
        device=device,
        debug=args.debug
    )
    
    # Save updated data
    logger.info(f"Saving processed data to: {args.output_jsonl}")
    save_jsonl(updated_data, args.output_jsonl)
    logger.info(f"Saved {len(updated_data)} samples with embedding paths")
    
    # Report mismatch if present
    saved_with_embeds = sum(1 for item in updated_data if 'history_embeds_path' in item)
    logger.info(f"Samples with history_embeds_path field: {saved_with_embeds}/{len(updated_data)}")
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()