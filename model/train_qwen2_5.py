import argparse
import logging
import torch
import numpy as np
from pathlib import Path
import torch.nn as nn

from model.train import TrainingConfig
import deepspeed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from model import MambaCompressor
import os
from model.train import train_single_utterance, train_conversations
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

# set logger to file
logging.basicConfig(filename='training.log', level=logging.INFO)


def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

def load_llm_and_tokenizer(config: TrainingConfig):
    # First load model in fp16
    model = AutoModelForCausalLM.from_pretrained(
        config.llm_name,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_name,
        truncation_side="left",
        use_fast=True,
    )
    
    # Add LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Convert to LoRA
    model = get_peft_model(model, lora_config)
    
    # Add special tokens
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['<MEM>']
    })

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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size_single", type=int, default=8)
    parser.add_argument("--batch_size_conv", type=int, default=1)
    parser.add_argument("--epochs_single", type=int, default=3)
    parser.add_argument("--epochs_conv", type=int, default=2)
    parser.add_argument("--lr_single", type=float, default=2.5e-5)
    parser.add_argument("--lr_conv", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--end_sym", default="\n")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--compute_dtype", default="float16")
    parser.add_argument("--quant_type", default="nf4")
    parser.add_argument("--use_double_quant", action="store_true")
    
    # Step-based validation and early stopping arguments
    parser.add_argument("--eval_steps", type=int, default=50, help="Validate every N steps (0 to disable)")
    parser.add_argument("--patience_steps", type=int, default=200, help="Early stopping after N steps without improvement")
    
    # Epoch-based early stopping (used when eval_steps=0)
    parser.add_argument("--patience", type=int, default=3, help="Epoch-based early stopping patience")
    
    # Learning rate scheduler arguments
    parser.add_argument("--scheduler_type", default="reduce_on_plateau", choices=["reduce_on_plateau", "cosine"])
    parser.add_argument("--scheduler_patience", type=int, default=1)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)
    
    # Other optimization arguments
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1) 
    parser.add_argument("--gradient_clip_value", type=float, default=5.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Mixup alpha parameter (0=disabled)")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    # DeepSpeed arguments
    parser.add_argument("--local_rank", type=int, default=-1) # 
    parser.add_argument("--deepspeed", type=str, default=None)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16) 
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    args = parser.parse_args()
    config = TrainingConfig(**vars(args))

    logger.info(config)
    
    setup_logging(Path(config.model_dir))
    
    llm, tokenizer = load_llm_and_tokenizer(config)

    model = None
    # TODO: load model from deepspeed checkpoint
    # if config.checkpoint_path is not None:
    #     model = MambaCompressor(
    #         llm_input_size=llm.config.hidden_size,
    #         device=config.device,
    #         tokenizer_len=len(tokenizer),
    #         mem_token_id=tokenizer.convert_tokens_to_ids('<MEM>'),
    #         mamba_path=config.mamba_path,
    #     ).to(config.device)
    #     # Training stages
    #     state_dict = torch.load(config.checkpoint_path, map_location=model.device)
        
    #     model.load_state_dict(state_dict)
    #     print(f"Loaded model from checkpoint {config.checkpoint_path}")

    # model.llm_model = llm.to("cuda").to(dtype=torch.float16)
    # model.llm_tokenizer = tokenizer

    local_rank = int(os.getenv('LOCAL_RANK', -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)


    # model = model.half()
    model = MambaCompressor(
        llm_input_size=llm.config.hidden_size,
        device=config.device,
        tokenizer_len=len(tokenizer),
        mem_token_id=tokenizer.convert_tokens_to_ids('<MEM>'),
        mamba_path=config.mamba_path,
        llm_model=llm.to("cuda").to(dtype=torch.float16),
        llm_tokenizer=tokenizer,
    )

    deepspeed.init_distributed()
    model_engine = deepspeed.initialize(
        model=model,
        config=args.deepspeed,
        model_parameters=model.parameters()
    )[0]

    model = model_engine
    
    model = train_single_utterance(
        config=config,
        llm=llm,
        tokenizer=tokenizer,
        model_dir=f"{config.model_dir}_stage1",
        model=model
    )
    model.to("cuda")

    train_conversations(
        config=config,
        tokenizer=tokenizer,
        model=model,
        model_dir=f"{config.model_dir}_stage2"
    )

if __name__ == "__main__":
    main()