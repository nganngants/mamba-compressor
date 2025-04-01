import logging
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from model.inputs import prepare_input
from data import ConversationDataset, prepare_single_utterances_data, prepare_multiple_utterances_data
from dataclasses import dataclass

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
    scheduler_min_lr: float = 1e-7
    
    # Optimization parameters
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    gradient_clip_value: float = 5.0
    mixup_alpha: float = 0.0  # Mixup regularization, 0 = disabled
    warmup_steps: int = 0

    local_rank: int = -1
    deepspeed: str = "ds_config.json"

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05


def mixup_data(input_embeds, labels, alpha=1.0, device='cuda'):
    """Applies mixup augmentation to the embeddings"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = input_embeds.size(0)
    index = torch.randperm(batch_size)
    
    mixed_input_embeds = lam * input_embeds + (1 - lam) * input_embeds[index, :]
    return mixed_input_embeds, labels, labels[index], lam



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

def train_epoch(model,
                tokenizer: AutoTokenizer,
                train_loader: DataLoader,
                val_loader: DataLoader,
                config: TrainingConfig,
                optimizer,
                scheduler=None,
                early_stopping=None) -> Tuple[float, bool]:
    
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    global_step = 0
    stop_training = False
    best_val_loss = float('inf')
    
    for i, batch in enumerate(progress_bar):
        input_ids = tokenizer(
            batch["input_text"],
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=config.max_length
        )
        
        input_ids["input_ids"] = input_ids["input_ids"].to("cuda")
        input_ids["attention_mask"] = input_ids["attention_mask"].to("cuda")
        llm_outputs = model(input_ids)
        
        loss = llm_outputs.loss
        model.backward(loss)
        model.step()
        
        # optimizer.zero_grad()
        # loss.backward()

        # if config.gradient_clip_value > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_value)
        # optimizer.step()
        
        # if scheduler is not None:
        #     scheduler.step()

        # Log the actual loss value (not the scaled one)
        batch_loss = loss.item()
        total_loss += batch_loss
        progress_bar.set_postfix({'loss': batch_loss, 'step': i})

        if config.eval_steps > 0 and i % config.eval_steps == 0 and i > 0:
            val_loss = validate(model, tokenizer, val_loader, config)
            model.train()
            
            if early_stopping is not None and early_stopping(model, val_loss):
                stop_training = True
                break
        
        del input_ids, llm_outputs
        torch.cuda.empty_cache()
            
        
        if stop_training:
            break
    
    avg_loss = total_loss / len(train_loader)
        
    return avg_loss, stop_training

def validate(model,
            tokenizer: AutoTokenizer,
            val_loader: DataLoader,
            config: TrainingConfig) -> float:
    
    model.eval()
    total_loss = 0
    
    progress_bar = tqdm(val_loader, desc="Validating")
    
    with torch.no_grad():
        for batch in progress_bar:
            try:
                input_ids = tokenizer(
                    batch["input_text"],
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=config.max_length
                )
                
                input_ids["input_ids"] = input_ids["input_ids"].to("cuda")
                input_ids["attention_mask"] = input_ids["attention_mask"].to("cuda")
                llm_outputs = model(input_ids)
                
                batch_loss = llm_outputs.loss.item()
                total_loss += batch_loss
                progress_bar.set_postfix({'val_loss': batch_loss})
                
                del input_ids, llm_outputs
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
                         model):
    
    print("Training single utterances")
    train_data = prepare_single_utterances_data(config.train_data)
    valid_data = prepare_single_utterances_data(config.valid_data)
    
    if model is None:
        raise ValueError("Model is None")
    
    train_dataset = ConversationDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_single, shuffle=True)
    val_dataset = ConversationDataset(valid_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size_single)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr_single,
        weight_decay=config.weight_decay
    )

    scheduler = get_scheduler(
        config.scheduler_type,
        optimizer,
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
        min_lr=config.scheduler_min_lr
    )
        
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
        train_loss = train_epoch(
            model, tokenizer, train_loader, val_loader, config, optimizer, scheduler, early_stopping
        )
        val_loss = validate(model, tokenizer, val_loader, config)
        
        logging.info(f'Epoch {epoch+1} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')
        
        # Save checkpoint if it's the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.module.save_pretrained(f"{model_dir}_best")
            model.save_checkpoint(f"{model_dir}_engine_best", tag="best_model")
            logging.info(f"Saved best model with validation loss: {val_loss:.4f}")
    
    # Final save
    model.save_pretrained(model_dir)
    return model

def train_conversations(config: TrainingConfig,
                       tokenizer,
                       model,
                       model_dir: str):
    print("Training multiple utterances")
    train_data = prepare_multiple_utterances_data(config.train_data)
    valid_data = prepare_multiple_utterances_data(config.valid_data)
    
    train_dataset = ConversationDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_conv, shuffle=True)
    val_dataset = ConversationDataset(valid_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size_conv)
        
    # Initialize early stopping
    
    best_val_loss = float('inf')
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr_single,
        weight_decay=config.weight_decay
    )

    scheduler = get_scheduler(
        config.scheduler_type,
        optimizer,
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
        min_lr=config.scheduler_min_lr
    )

    for epoch in range(config.epochs_conv):
        train_loss = train_epoch(
            model, tokenizer, train_loader, val_loader, config, optimizer, scheduler, None
        )
        val_loss = validate(model, tokenizer, val_loader, config)
        
        logging.info(f'Epoch {epoch+1} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')
        
        # Save checkpoint if it's the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.module.save_pretrained(f"{model_dir}_best")
            model.save_checkpoint(f"{model_dir}_engine_best", tag="best_model")
            logging.info(f"Saved best model with validation loss: {val_loss:.4f}")
    
    # Final save
    model.save_pretrained(model_dir)