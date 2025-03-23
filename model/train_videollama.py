import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import deepspeed
import os
from data import (
    ConversationDataset, 
    prepare_single_utterances_data, 
    prepare_multiple_utterances_data
)
from videollama2 import model_init

from model import MambaCompressor
from model.train import train_single_utterance, train_conversations, TrainingConfig

# Environment settings for better performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

logger = logging.getLogger(__name__)

# Set logger to file
def setup_logging(log_dir: Path, rank=0):
    if rank == 0:  # Only create logs on main process
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
    else:
        # Configure minimal logging for other ranks
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s | RANK {} | %(message)s'.format(rank),
            handlers=[logging.StreamHandler()]
        )

def load_llm_and_tokenizer(config: TrainingConfig):
    """Load LLM and tokenizer with proper quantization settings"""

    model, _, tokenizer = model_init(config.llm_name)
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

    for param in model.parameters():
        param.requires_grad = False
   
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mamba_path", required=False, default="state-spaces/mamba-370m-hf", help="Path to pretrained Mamba model")
    parser.add_argument("--train_data", required=True, help="Path to training jsonl")
    parser.add_argument("--valid_data", required=True, help="Path to validation jsonl")
    parser.add_argument("--model_dir", required=True, help="Directory to save model checkpoints")
    parser.add_argument("--llm_name", required=True, help="Name of the LLM model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size_single", type=int, default=16)
    parser.add_argument("--batch_size_conv", type=int, default=4)
    parser.add_argument("--epochs_single", type=int, default=3)
    parser.add_argument("--epochs_conv", type=int, default=2)
    parser.add_argument("--lr_single", type=float, default=3e-5)
    parser.add_argument("--lr_conv", type=float, default=1.5e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--end_sym", default="\n")
    
    # Step-based validation and early stopping arguments
    parser.add_argument("--eval_steps", type=int, default=100, help="Validate every N steps (0 to disable)")
    parser.add_argument("--patience_steps", type=int, default=300, help="Early stopping after N steps without improvement")
    
    # Epoch-based early stopping (used when eval_steps=0)
    parser.add_argument("--patience", type=int, default=3, help="Epoch-based early stopping patience")
    
    # Learning rate scheduler arguments
    parser.add_argument("--scheduler_type", default="cosine", choices=["reduce_on_plateau", "cosine"])
    parser.add_argument("--scheduler_patience", type=int, default=1)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)
    
    # Other optimization arguments
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1) 
    parser.add_argument("--gradient_clip_value", type=float, default=1.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Mixup alpha parameter (0=disabled)")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    
    # DeepSpeed specific arguments
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--zero_stage", type=int, default=2, help="ZeRO optimization stage")
    parser.add_argument("--offload_optimizer", action="store_true", help="Offload optimizer states to CPU")
    parser.add_argument("--offload_param", action="store_false", help="Do not offload parameters to CPU")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 training")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    
    args = parser.parse_args()
    
    # Initialize distributed training environment for DeepSpeed
    if args.deepspeed:
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(args.local_rank)
        deepspeed.init_distributed()
        args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    else:
        args.local_rank = -1
    
    config = TrainingConfig(**vars(args))
    
    setup_logging(Path(config.model_dir), rank=config.local_rank)
    
    if config.local_rank <= 0:  # Only log on main process
        logger.info(f"Training configuration: {config}")
        logger.info(f"Using DeepSpeed: {config.deepspeed}")
        if config.deepspeed:
            logger.info(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
            logger.info(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    
    # Load models
    llm, tokenizer = load_llm_and_tokenizer(config)
    
    # Create model
    model = None
    if config.checkpoint_path is not None:
        model = MambaCompressor(
            llm_input_size=llm.config.hidden_size,
            device='cuda',
            tokenizer_len=len(tokenizer),
            mem_token_id=tokenizer.convert_tokens_to_ids('<MEM>'),
            mamba_path=config.mamba_path,
            enable_amp=config.amp
        ).to('cuda')
        
        state_dict = torch.load(config.checkpoint_path, map_location=model.device)
        model.load_state_dict(state_dict)
        
        if config.local_rank <= 0:
            logging.info(f"Loaded model from checkpoint {config.checkpoint_path}")
    
    # Train single utterance stage
    model = train_single_utterance(
        config=config,
        llm=llm,
        tokenizer=tokenizer,
        model_dir=f"{config.model_dir}_stage1",
        model=model
    )
    
    # Need to extract the model from DeepSpeed engine for next stage
    if config.deepspeed and hasattr(model, "module"):
        # Get the actual model from DeepSpeed engine
        raw_model = model.module
        
        # Create a new model for the next stage
        model = MambaCompressor(
            llm_input_size=llm.config.hidden_size,
            device='cuda',
            tokenizer_len=len(tokenizer),
            mem_token_id=tokenizer.convert_tokens_to_ids('<MEM>'),
            mamba_path=config.mamba_path,
            enable_amp=config.amp
        ).to('cuda')
        
        # Copy weights from the trained model
        model.load_state_dict(raw_model.state_dict())
        
        if config.local_rank <= 0:
            logging.info("Recreated model for conversation stage")
    
    # Train conversations stage
    train_conversations(
        config=config,
        llm=llm,
        tokenizer=tokenizer,
        model=model,
        model_dir=f"{config.model_dir}_stage2"
    )

if __name__ == "__main__":
    main()