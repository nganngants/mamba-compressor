import argparse
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, MambaModel, BitsAndBytesConfig
import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass
from data import (
    ConversationDataset, 
    prepare_input,
    prepare_single_utterances_data, 
    prepare_multiple_utterances_data
)
from model import MambaCompressor
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    mamba_path: str
    train_data: str
    valid_data: str
    model_dir: str
    llm_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size_single: int = 8
    batch_size_conv: int = 1
    epochs_single: int = 3
    epochs_conv: int = 2
    lr_single: float = 2.5e-5
    lr_conv: float = 1e-4
    max_length: int = 512
    end_sym: str = '\n',
    load_in_4bit: bool = False
    compute_dtype: str = "float16"
    quant_type: str = "nf4"
    use_double_quant: bool = False

def train_epoch(model: MambaCompressor,
                llm: AutoModelForCausalLM,
                tokenizer: AutoTokenizer,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                system_prompt: str,
                config: TrainingConfig) -> float:
    
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader)
    
    for batch in progress_bar:
        optimizer.zero_grad()
        input_data = prepare_input(
            mamba_model=model,
            llm_model=llm,
            llm_tokenizer=tokenizer,
            system_prompt=system_prompt,
            input_texts=batch['input_text'],
            device=config.device,
            end_sym=config.end_sym
        )
        
        logger.info(f"input embeds: {input_data['input_embeds'].shape}")
        logger.info(f"attention mask: {input_data['attention_mask'].shape}")
        logger.info(f"labels: {input_data['labels'].shape}")
        
        llm_outputs = llm(
            inputs_embeds=input_data['input_embeds'],
            attention_mask=input_data['attention_mask'],
            labels=input_data['labels'],
            return_dict=True
        )
        
        loss = llm_outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
        del input_data, llm_outputs
        torch.cuda.empty_cache()
        
    return total_loss / len(train_loader)

def validate(model: MambaCompressor,
            llm: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            val_loader: DataLoader,
            system_prompt: str,
            config: TrainingConfig) -> float:
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_data = prepare_input(
                mamba_model=model,
                llm_model=llm,
                llm_tokenizer=tokenizer,
                system_prompt=system_prompt,
                input_texts=batch['input_text'],
                device=config.device,
                end_sym=config.end_sym
            )
            
            llm_outputs = llm(
                inputs_embeds=input_data['input_embeds'],
                attention_mask=input_data['attention_mask'],
                labels=input_data['labels'],
                return_dict=True
            )
            
            total_loss += llm_outputs.loss.item()
            
    return total_loss / len(val_loader)

def train_single_utterance(config: TrainingConfig,
                         llm: AutoModelForCausalLM,
                         tokenizer: AutoTokenizer,
                         model_dir: str,
                         model: Optional[MambaCompressor] = None) -> MambaCompressor:
    
    train_data = prepare_single_utterances_data(config.train_data)
    valid_data = prepare_single_utterances_data(config.valid_data)
    
    if model is None:
        model = MambaCompressor(
            llm_input_size=llm.config.hidden_size,
            device=config.device,
            tokenizer_len=len(tokenizer),
            mem_token_id=tokenizer.convert_tokens_to_ids('<MEM>'),
            mamba_path=config.mamba_path,
        ).to(config.device)
    
    train_dataset = ConversationDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_single, shuffle=True)
    val_dataset = ConversationDataset(valid_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size_single)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_single)
    system_prompt = "Please reconstruct the conversation in a natural way."
    
    for epoch in range(config.epochs_single):
        train_loss = train_epoch(model, llm, tokenizer, train_loader, optimizer, system_prompt, config)
        val_loss = validate(model, llm, tokenizer, val_loader, system_prompt, config)
        logging.info(f'Epoch {epoch+1} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')
    
    model.save_pretrained(model_dir)
    return model

def train_conversations(config: TrainingConfig,
                       llm: AutoModelForCausalLM,
                       tokenizer: AutoTokenizer,
                       model: MambaCompressor,
                       model_dir: str):
    
    train_data = prepare_multiple_utterances_data(config.train_data)
    valid_data = prepare_multiple_utterances_data(config.valid_data)
    
    train_dataset = ConversationDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_conv, shuffle=True)
    val_dataset = ConversationDataset(valid_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size_conv)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_conv)
    system_prompt = "You are a helpful assistant. Provided the compressed embeddings, please reconstruct the conversation."
    
    for epoch in range(config.epochs_conv):
        train_loss = train_epoch(model, llm, tokenizer, train_loader, optimizer, system_prompt, config)
        val_loss = validate(model, llm, tokenizer, val_loader, system_prompt, config)
        logging.info(f'Epoch {epoch+1} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')
    
    model.save_pretrained(model_dir)

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
    """Load LLM and tokenizer with proper quantization settings"""
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, config.compute_dtype),
            bnb_4bit_quant_type=config.quant_type,
            bnb_4bit_use_double_quant=config.use_double_quant,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(config.llm_name, add_bos_token=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # add <MEM> token to tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': ['<MEM>']})
        
    model = AutoModelForCausalLM.from_pretrained(
        config.llm_name,
        device_map=config.device,
        quantization_config=quantization_config,
        torch_dtype=getattr(torch, config.compute_dtype)
    )
    
    # Move model to device explicitly if not using device_map
    if not config.load_in_4bit and config.device != "auto":
        model = model.to(config.device)
        
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mamba_path", required=False, default="state-spaces/mamba-370m-hf", help="Path to pretrained Mamba model")
    parser.add_argument("--train_data", required=True, help="Path to training jsonl")
    parser.add_argument("--valid_data", required=True, help="Path to validation jsonl")
    parser.add_argument("--model_dir", required=True, help="Directory to save model checkpoints")
    parser.add_argument("--llm_name", required=True, help="Name of the LLM model")
    parser.add_argument("--device", default="cpu")
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
    
    args = parser.parse_args()
    config = TrainingConfig(**vars(args))
    
    setup_logging(Path(config.model_dir))
    
    llm, tokenizer = load_llm_and_tokenizer(config)
    
    # Training stages
    model = train_single_utterance(
        config=config,
        llm=llm,
        tokenizer=tokenizer,
        model_dir=f"{config.model_dir}_stage1"
    )
    
    train_conversations(
        config=config,
        llm=llm,
        tokenizer=tokenizer,
        model=model,
        model_dir=f"{config.model_dir}_stage2"
    )

if __name__ == "__main__":
    main()
