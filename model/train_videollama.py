import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from data import (
    ConversationDataset, 
    prepare_single_utterances_data, 
    prepare_multiple_utterances_data
)
from videollama2 import model_init

from model import MambaCompressor
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"  # Try disabling cuDNN v8 API
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For better error reporting

# After imports, add these PyTorch settings
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

logger = logging.getLogger(__name__)

# set logger to file
logging.basicConfig(filename='training.log', level=logging.INFO)

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
    end_sym: str = '\n'
    load_in_4bit: bool = False
    compute_dtype: str = "float16"
    quant_type: str = "nf4"
    use_double_quant: bool = False
    checkpoint_path: Optional[str] = None
    
    # Step-based validation and early stopping
    eval_steps: int = 50  # Validate every N steps
    patience_steps: int = 200  # Stop if no improvement after N steps
    
    # Traditional epoch-based early stopping (not used with step-based approach)
    patience: int = 3  
    
    # LR scheduler
    scheduler_type: str = "reduce_on_plateau"  # or "cosine"
    scheduler_patience: int = 1
    scheduler_factor: float = 0.5  # Reduce LR by half when plateau is detected
    scheduler_min_lr: float = 1e-6
    
    # Optimization parameters
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    gradient_clip_value: float = 5.0
    mixup_alpha: float = 0.0  # Mixup regularization, 0 = disabled
    warmup_steps: int = 0

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0, restore_best_weights=True, step_based=False):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = np.inf
        self.best_weights = None
        self.should_stop = False
        self.step_based = step_based  # Whether this is step-based or epoch-based
        self.steps_without_improvement = 0

    def __call__(self, model, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.steps_without_improvement = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            if self.step_based:
                self.steps_without_improvement += 1
                if self.steps_without_improvement >= self.patience:
                    self.should_stop = True
                    if self.restore_best_weights and self.best_weights is not None:
                        model.load_state_dict(self.best_weights)
                        logger.info("Restoring best weights after step-based early stopping triggered")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
                    if self.restore_best_weights and self.best_weights is not None:
                        model.load_state_dict(self.best_weights)
                        logger.info("Restoring best weights after epoch-based early stopping triggered")
            
        return self.should_stop

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
        max_length=1024
    ).to(device)

    memory_features = mamba_model(input_ids).to(torch.float16)
    atts_memory = torch.ones(
        (memory_features.size(0), memory_features.size(1)),
        dtype=torch.long,
    ).to(device)
    
    # Combine system prompt with memory features
    system_encodings = llm_tokenizer(
        [system_prompt] * len(input_texts),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=64
    ).to(device)
    system_embeds = llm_model.model.embed_tokens(system_encodings['input_ids']) # (batch, seq, hidden)
    
    memory_features = torch.cat([system_embeds, memory_features], dim=1)
    atts_memory = atts_memory[:, :1].expand(-1, memory_features.size(1))

    # Prepare target texts
    target_texts = [t + end_sym for t in input_texts]
    to_regress_tokens = llm_tokenizer(
        target_texts,
        truncation=True,
        return_tensors="pt",
        max_length=1024,
    ).to(device)
    targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == llm_tokenizer.pad_token_id, -100
            )
    empty_targets = (
                torch.ones([memory_features.shape[0], memory_features.shape[1]],
                        dtype=torch.long).to(device).fill_(-100)
            )
    targets = torch.cat([empty_targets, targets], dim=1)

    to_regress_embeds = llm_model.model.embed_tokens(to_regress_tokens.input_ids)
    input_embeds = torch.cat([memory_features, to_regress_embeds], dim=1)
    attention_mask = torch.cat([atts_memory, to_regress_tokens.attention_mask], dim=1)

    return {
        'input_embeds': input_embeds,
        'attention_mask': attention_mask,
        'labels': targets
    }

def mixup_data(input_embeds, labels, alpha=1.0, device='cuda'):
    """Applies mixup augmentation to the embeddings"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = input_embeds.size(0)
    index = torch.randperm(batch_size).to(device)
    
    mixed_input_embeds = lam * input_embeds + (1 - lam) * input_embeds[index, :]
    return mixed_input_embeds, labels, labels[index], lam

def train_epoch(model: MambaCompressor,
                llm: AutoModelForCausalLM,
                tokenizer: AutoTokenizer,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler,
                system_prompt: str,
                config: TrainingConfig,
                early_stopping=None) -> Tuple[float, bool]:
    
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    # Disable cuDNN optimizations for debugging
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    
    optimizer.zero_grad()
    global_step = 0
    stop_training = False
    best_val_loss = float('inf')
    
    for i, batch in enumerate(progress_bar):
        try:
            input_data = prepare_input(
                mamba_model=model,
                llm_model=llm,
                llm_tokenizer=tokenizer,
                system_prompt=system_prompt,
                input_texts=batch['input_text'],
                device=config.device,
                end_sym=config.end_sym
            )
            
            # Optional mixup regularization
            if config.mixup_alpha > 0:
                mixed_embeds, labels_a, labels_b, lam = mixup_data(
                    input_data['input_embeds'], 
                    input_data['labels'],
                    alpha=config.mixup_alpha,
                    device=config.device
                )
                input_data['input_embeds'] = mixed_embeds
            
            llm_outputs = llm(
                inputs_embeds=input_data['input_embeds'],
                attention_mask=input_data['attention_mask'],
                labels=input_data['labels'],
                return_dict=True
            )
            
            loss = llm_outputs.loss
            
            # Apply loss scaling for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (i + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_value)
                optimizer.step()
                optimizer.zero_grad()
                
                # Step the scheduler if it's not ReduceLROnPlateau
                if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step()
                
                global_step += 1
                
                # Step-based validation and early stopping
                if config.eval_steps > 0 and global_step % config.eval_steps == 0:
                    val_loss = validate(model, llm, tokenizer, val_loader, system_prompt, config)
                    model.train()  # Switch back to train mode
                    
                    logger.info(f'Step {global_step} - Validation loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]}')
                    
                    # For ReduceLROnPlateau scheduler
                    if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    
                    # Step-based early stopping check
                    if early_stopping is not None and early_stopping.step_based:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            # Save the best model
                            torch.save(model.state_dict(), f"{config.model_dir}_step_best.pt")
                            logger.info(f"Saved best model at step {global_step} with validation loss: {val_loss:.4f}")
                        
                        if early_stopping(model, val_loss):
                            logger.info(f"Early stopping triggered at step {global_step}")
                            stop_training = True
                            break
            
            # Log the actual loss value (not the scaled one)
            batch_loss = loss.item() * config.gradient_accumulation_steps
            total_loss += batch_loss
            progress_bar.set_postfix({'loss': batch_loss, 'lr': optimizer.param_groups[0]['lr'], 'step': global_step})
            
            del input_data, llm_outputs
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            logger.error(f"Error during forward pass: {e}")
            torch.cuda.empty_cache()
            continue
        
        if stop_training:
            break
    
    avg_loss = total_loss / len(train_loader)
    
    # For ReduceLROnPlateau scheduler - only if not using step-based validation
    if config.eval_steps <= 0 and scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(avg_loss)
        
    return avg_loss, stop_training

def validate(model: MambaCompressor,
            llm: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            val_loader: DataLoader,
            system_prompt: str,
            config: TrainingConfig) -> float:
    
    model.eval()
    total_loss = 0
    
    progress_bar = tqdm(val_loader, desc="Validating")
    
    with torch.no_grad():
        for batch in progress_bar:
            try:
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
                
                batch_loss = llm_outputs.loss.item()
                total_loss += batch_loss
                progress_bar.set_postfix({'val_loss': batch_loss})
                
                del input_data, llm_outputs
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                logger.error(f"Error during validation: {e}")
                torch.cuda.empty_cache()
                continue
    
    return total_loss / len(val_loader)

def get_scheduler(scheduler_type, optimizer, patience, factor, min_lr, t_max=None):
    """Returns the appropriate learning rate scheduler based on config"""
    if scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=factor,
            patience=patience, 
            verbose=True,
            min_lr=min_lr
        )
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=t_max if t_max is not None else 10,
            eta_min=min_lr
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}. No scheduler will be used.")
        return None

def train_single_utterance(config: TrainingConfig,
                         llm: AutoModelForCausalLM,
                         tokenizer: AutoTokenizer,
                         model_dir: str,
                         model: Optional[MambaCompressor] = None) -> MambaCompressor:
    
    print("Training single utterances")
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
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr_single,
        weight_decay=config.weight_decay
    )
    
    # Get the appropriate scheduler
    scheduler = get_scheduler(
        config.scheduler_type,
        optimizer,
        config.scheduler_patience,
        config.scheduler_factor,
        config.scheduler_min_lr,
        t_max=config.epochs_single * len(train_loader)
    )
    
    system_prompt = "You are a helpful assistant. Provided the compressed embeddings, please reconstruct the conversation."
    
    # Initialize early stopping
    # If we have step-based early stopping configured, use that
    if config.eval_steps > 0 and config.patience_steps > 0:
        early_stopping = EarlyStopping(
            patience=config.patience_steps, 
            restore_best_weights=True,
            step_based=True
        )
        logger.info(f"Using step-based early stopping with patience of {config.patience_steps} steps")
    else:
        # Otherwise fall back to epoch-based early stopping
        early_stopping = EarlyStopping(
            patience=config.patience, 
            restore_best_weights=True,
            step_based=False
        )
        logger.info(f"Using epoch-based early stopping with patience of {config.patience} epochs")
    
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs_single):
        # If using step-based validation, we pass the validation loader and early stopping to train_epoch
        if config.eval_steps > 0:
            train_loss, stop_training = train_epoch(
                model, llm, tokenizer, train_loader, val_loader, optimizer, scheduler, system_prompt, config, early_stopping
            )
            logging.info(f'Epoch {epoch+1} completed - Train loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]}')
            
            if stop_training:
                logging.info(f"Early stopping triggered during epoch {epoch+1}")
                break
        else:
            # Traditional epoch-based approach
            train_loss, _ = train_epoch(
                model, llm, tokenizer, train_loader, None, optimizer, scheduler, system_prompt, config
            )
            val_loss = validate(model, llm, tokenizer, val_loader, system_prompt, config)
            
            logging.info(f"Epoch {epoch+1} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            
            # Save checkpoint if it's the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(f"{model_dir}_best")
                logging.info(f"Saved best model with validation loss: {val_loss:.4f}")
            
            # Early stopping check for epoch-based
            if early_stopping(model, val_loss):
                logging.info(f"Early stopping triggered after epoch {epoch+1}")
                break
    
    # Final save
    model.save_pretrained(model_dir)
    return model

def train_conversations(config: TrainingConfig,
                       llm: AutoModelForCausalLM,
                       tokenizer: AutoTokenizer,
                       model: MambaCompressor,
                       model_dir: str):
    print("Training multiple utterances")
    train_data = prepare_multiple_utterances_data(config.train_data)
    valid_data = prepare_multiple_utterances_data(config.valid_data)
    
    train_dataset = ConversationDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_conv, shuffle=True)
    val_dataset = ConversationDataset(valid_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size_conv)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr_conv,
        weight_decay=config.weight_decay
    )
    
    # Get the appropriate scheduler
    scheduler = get_scheduler(
        config.scheduler_type,
        optimizer,
        config.scheduler_patience,
        config.scheduler_factor,
        config.scheduler_min_lr,
        t_max=config.epochs_conv * len(train_loader)
    )
    
    system_prompt = "You are a helpful assistant. Provided the compressed embeddings, please reconstruct the conversation."
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=config.patience)
    
    best_val_loss = float('inf')

    for epoch in range(config.epochs_conv):
        train_loss = train_epoch(
            model, llm, tokenizer, train_loader, val_loader, optimizer, scheduler, system_prompt, config
        )
        val_loss = validate(model, llm, tokenizer, val_loader, system_prompt, config)
        
        logging.info(f'Epoch {epoch+1} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')
        
        # Save checkpoint if it's the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(f"{model_dir}_best")
            logging.info(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Early stopping check
        if early_stopping(model, val_loss):
            logging.info(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Final save
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
    
    args = parser.parse_args()
    config = TrainingConfig(**vars(args))

    logger.info(config)
    
    setup_logging(Path(config.model_dir))
    
    llm, tokenizer = load_llm_and_tokenizer(config)
    
    model = None
    if config.checkpoint_path is not None:
        model = MambaCompressor(
            llm_input_size=llm.config.hidden_size,
            device=config.device,
            tokenizer_len=len(tokenizer),
            mem_token_id=tokenizer.convert_tokens_to_ids('<MEM>'),
            mamba_path=config.mamba_path,
        ).to(config.device)
        # Training stages
        state_dict = torch.load(config.checkpoint_path, map_location=model.device)
        
        model.load_state_dict(state_dict)
        print(f"Loaded model from checkpoint {config.checkpoint_path}")


    # model = train_single_utterance(
    #     config=config,
    #     llm=llm,
    #     tokenizer=tokenizer,
    #     model_dir=f"{config.model_dir}_stage1",
    #     model=model
    # )
    model = MambaCompressor.from_pretrained("./mamba_compressor_videollama_stage1", device="cuda", tokenizer_len=len(tokenizer))

    model.to("cuda")
    
    train_conversations(
        config=config,
        llm=llm,
        tokenizer=tokenizer,
        model=model,
        model_dir=f"{config.model_dir}_stage2"
    )

if __name__ == "__main__":
    main()