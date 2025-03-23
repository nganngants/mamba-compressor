# import libraries
from dataclasses import dataclass
import logging
import numpy as np
import os
from typing import Optional, Tuple, Dict, Any, Callable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
import argparse
from tqdm import tqdm
from data import ConversationDataset, prepare_single_utterances_data, prepare_multiple_utterances_data
from model import MambaCompressor
from model.inputs import prepare_input
import json 
from .mamba_compressor import setup_deepspeed_model

@dataclass
class TrainingConfig:
    mamba_path: str
    train_data: str
    valid_data: str
    model_dir: str
    llm_name: str
    device: str = "cuda"
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

    # DeepSpeed parameters
    deepspeed: bool = True  # Set to True by default
    deepspeed_config: Optional[str] = "ds_config.json"
    local_rank: int = -1
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = True
    fp16: bool = True
    bf16: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    amp: bool = False


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
                # For DeepSpeed compatibility, save the model checkpoint instead of storing weights in memory
                if hasattr(model, 'module'):
                    is_ds_model = hasattr(model.module, 'save_checkpoint')
                    if is_ds_model:
                        # Save DeepSpeed checkpoint
                        model.save_checkpoint("best_model_checkpoint")
                        self.best_weights = "best_model_checkpoint"
                    else:
                        self.best_weights = {k: v.detach().cpu().clone() for k, v in model.module.state_dict().items()}
                else:
                    self.best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            if self.step_based:
                self.steps_without_improvement += 1
                if self.steps_without_improvement >= self.patience:
                    self.should_stop = True
                    if self.restore_best_weights and self.best_weights is not None:
                        if isinstance(self.best_weights, str):
                            # Load DeepSpeed checkpoint
                            model.load_checkpoint(self.best_weights)
                            logging.info("Restoring best weights from checkpoint after step-based early stopping triggered")
                        else:
                            if hasattr(model, 'module'):
                                model.module.load_state_dict(self.best_weights)
                            else:
                                model.load_state_dict(self.best_weights)
                            logging.info("Restoring best weights after step-based early stopping triggered")
            else:
                raise NotImplementedError("Epoch-based early stopping is not implemented yet")
            
        return self.should_stop

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

def train_epoch(model,
                llm: AutoModelForCausalLM,
                tokenizer: AutoTokenizer,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer,
                scheduler,
                system_prompt: str,
                config: TrainingConfig,
                early_stopping=None) -> Tuple[float, bool]:
    
    # Set the model to train mode (handled by DeepSpeed if using it)
    if hasattr(model, 'train'):
        model.train()
    
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", disable=config.local_rank != 0)
    
    # Disable cuDNN optimizations for debugging
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    
    # No need to manually zero gradients when using DeepSpeed
    global_step = 0
    stop_training = False
    best_val_loss = float('inf')
    
    actual_mamba_model = model.module if hasattr(model, 'module') else model
    
    for i, batch in enumerate(progress_bar):
        try:
            logging.info(f'{i}')
            logging.info(f'device: {actual_mamba_model.device}')
            input_data = prepare_input(
                mamba_model=actual_mamba_model,
                llm_model=llm,
                llm_tokenizer=tokenizer,
                system_prompt=system_prompt,
                input_texts=batch['input_text'],
                device='cuda',
                end_sym=config.end_sym
            )
            
            logging.info('check devices')
            logging.info(input_data['input_embeds'].device)
            logging.info(input_data['attention_mask'].device)
            logging.info(input_data['labels'].device)

            # Optional mixup regularization
            if config.mixup_alpha > 0:
                mixed_embeds, labels_a, labels_b, lam = mixup_data(
                    input_data['input_embeds'], 
                    input_data['labels'],
                    alpha=config.mixup_alpha,
                    device='cuda'
                )
                input_data['input_embeds'] = mixed_embeds
            
            llm_outputs = llm(
                inputs_embeds=input_data['input_embeds'],
                attention_mask=input_data['attention_mask'],
                labels=input_data['labels'],
                return_dict=True
            )
            
            loss = llm_outputs.loss
            
            # Using DeepSpeed's backward
            if hasattr(model, 'backward'):
                model.backward(loss)
                model.step()
            else:
                # Apply loss scaling for gradient accumulation if not using DeepSpeed
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
                
                # Gradient accumulation
                if (i + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(actual_mamba_model.parameters(), config.gradient_clip_value)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Step the scheduler if it's not ReduceLROnPlateau
            if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau) and (i + 1) % config.gradient_accumulation_steps == 0:
                scheduler.step()
            
            global_step += 1
            
            # Step-based validation and early stopping
            if config.eval_steps > 0 and global_step % config.eval_steps == 0:
                val_loss = validate(model, llm, tokenizer, val_loader, system_prompt, config)
                
                # Set the model back to train mode
                if hasattr(model, 'train'):
                    model.train()
                
                # Only log on rank 0 if using distributed training
                if config.local_rank <= 0:
                    current_lr = optimizer.param_groups[0]["lr"] if not hasattr(model, 'lr') else model.get_lr()[0]
                    logging.info(f'Step {global_step} - Validation loss: {val_loss:.4f}, LR: {current_lr}')
                
                # For ReduceLROnPlateau scheduler
                if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                
                # Step-based early stopping check
                if early_stopping is not None and early_stopping.step_based:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # Save the best model
                        if hasattr(model, 'save_checkpoint'):
                            model.save_checkpoint(f"{config.model_dir}_step_best")
                        else:
                            torch.save(actual_mamba_model.state_dict(), f"{config.model_dir}_step_best.pt")
                        
                        if config.local_rank <= 0:
                            logging.info(f"Saved best model at step {global_step} with validation loss: {val_loss:.4f}")
                    
                    if early_stopping(model, val_loss):
                        if config.local_rank <= 0:
                            logging.info(f"Early stopping triggered at step {global_step}")
                        stop_training = True
                        break
            
            # Log the actual loss value (not the scaled one)
            batch_loss = loss.item() * config.gradient_accumulation_steps if not hasattr(model, 'backward') else loss.item()
            total_loss += batch_loss
            print(batch_loss)
            
            # Only update progress bar on rank 0
            if config.local_rank <= 0:
                current_lr = optimizer.param_groups[0]["lr"] if not hasattr(model, 'lr') else model.get_lr()[0]
                progress_bar.set_postfix({'loss': batch_loss, 'lr': current_lr, 'step': global_step})
            
            del input_data, llm_outputs
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            logging.error(f"Error during forward pass: {e}")
            torch.cuda.empty_cache()
            continue
        
        if stop_training:
            break
    
    avg_loss = total_loss / len(train_loader)
    
    # For ReduceLROnPlateau scheduler - only if not using step-based validation
    if config.eval_steps <= 0 and scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(avg_loss)
        
    return avg_loss, stop_training

def validate(model,
            llm: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            val_loader: DataLoader,
            system_prompt: str,
            config: TrainingConfig) -> float:
    
    # Set model to eval mode
    if hasattr(model, 'eval'):
        model.eval()
    
    total_loss = 0
    progress_bar = tqdm(val_loader, desc="Validating", disable=config.local_rank != 0)
    
    actual_mamba_model = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        for batch in progress_bar:
            try:
                input_data = prepare_input(
                    mamba_model=actual_mamba_model,
                    llm_model=llm,
                    llm_tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    input_texts=batch['input_text'],
                    device='cuda',
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
                
                if config.local_rank <= 0:
                    progress_bar.set_postfix({'val_loss': batch_loss})
                
                del input_data, llm_outputs
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                logging.error(f"Error during validation: {e}")
                torch.cuda.empty_cache()
                continue
    
    # Synchronize loss across all GPUs when using DeepSpeed
    if config.deepspeed and torch.distributed.is_initialized():
        # Sum up the total loss
        torch.distributed.all_reduce(torch.tensor([total_loss]), op=torch.distributed.ReduceOp.SUM)
        # Get the correct divisor (total number of batches across all GPUs)
        num_batches = torch.tensor([len(val_loader)])
        torch.distributed.all_reduce(num_batches, op=torch.distributed.ReduceOp.SUM)
        total_loss /= num_batches.item()
    else:
        total_loss /= len(val_loader)
    
    return total_loss

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
        logging.warning(f"Unknown scheduler type: {scheduler_type}. No scheduler will be used.")
        return None

def train_model(config: TrainingConfig,
                llm: AutoModelForCausalLM,
                tokenizer: AutoTokenizer,
                model_dir: str,
                model=None,
                data_preparation_func: Callable=None,
                learning_rate: float=None,
                batch_size: int=None,
                num_epochs: int=None,
                is_step_based: bool=True,
                model_save_suffix: str="") -> Any:
    
    # Initialize distributed training if using DeepSpeed
    if config.deepspeed and not torch.distributed.is_initialized():
        deepspeed.init_distributed()
        config.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        device = f'cuda:{config.local_rank}'
    else:
        device = config.device
        
    if config.local_rank <= 0:
        logging.info(f"Training on device: {device}")
    
    # Prepare the data
    train_data = data_preparation_func(config.train_data)
    valid_data = data_preparation_func(config.valid_data)
    
    # Initialize the model if not provided
    if model is None:
        model = MambaCompressor(
            llm_input_size=llm.config.hidden_size,
            device=device,  # Pass device but don't move model
            tokenizer_len=len(tokenizer),
            mem_token_id=tokenizer.convert_tokens_to_ids('<MEM>'),
            mamba_path=config.mamba_path,
            enable_amp=config.amp
        )
    
    # Create datasets and data loaders
    train_dataset = ConversationDataset(train_data, tokenizer)
    val_dataset = ConversationDataset(valid_data, tokenizer)
    
    # Create data loaders with appropriate samplers for distributed training
    if config.deepspeed and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=config.local_rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=config.local_rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=4
    )
    
    # Set up DeepSpeed configuration
    ds_config = {
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 4,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": config.weight_decay
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": 1000,
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": 100,
            }
        },
        
        "fp16": {
            "enabled": config.fp16,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        "zero_optimization": {
            "stage": config.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": int(2e8),
            "allgather_bucket_size": int(2e8)
        },
        
        "gradient_clipping": config.gradient_clip_value,
        "steps_per_print": 50,
        "wall_clock_breakdown": False
    }
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters(),
    )
    
    # Set up scheduler if not handled by DeepSpeed
    if scheduler is None and not hasattr(model, 'get_lr'):
        scheduler = get_scheduler(
            config.scheduler_type,
            optimizer,
            config.scheduler_patience,
            config.scheduler_factor,
            config.scheduler_min_lr,
            t_max=num_epochs * len(train_loader)
        )
    
    system_prompt = "You are a helpful assistant. Provided the compressed embeddings, please reconstruct the conversation."
    
    # Initialize early stopping
    if is_step_based:
        early_stopping = EarlyStopping(
            patience=config.patience_steps, 
            restore_best_weights=True,
            step_based=True
        )
        if config.local_rank <= 0:
            logging.info(f"Using step-based early stopping with patience of {config.patience_steps} steps")
    else:
        raise NotImplementedError("Epoch-based early stopping is not implemented yet")
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Set epoch for DistributedSampler if using deepspeed
        if config.deepspeed and torch.distributed.is_initialized():
            train_loader.sampler.set_epoch(epoch)
        
        # If using step-based validation, we pass the validation loader and early stopping to train_epoch
        if is_step_based:
            train_loss, stop_training = train_epoch(
                model, llm, tokenizer, train_loader, val_loader, optimizer, scheduler, 
                system_prompt, config, early_stopping
            )
            if config.local_rank <= 0:
                current_lr = optimizer.param_groups[0]["lr"] if not hasattr(model, 'get_lr') else model.get_lr()[0]
                logging.info(f'Epoch {epoch+1} completed - Train loss: {train_loss:.4f}, LR: {current_lr}')
            
            if stop_training:
                if config.local_rank <= 0:
                    logging.info(f"Early stopping triggered during epoch {epoch+1}")
                break
        else:
            raise NotImplementedError("Epoch-based early stopping is not implemented yet")
    
    # Final save (only on rank 0)
    if config.local_rank <= 0:
        if hasattr(model, 'save_checkpoint'):
            model.save_checkpoint(f"{model_dir}{model_save_suffix}")
        else:
            if hasattr(model, 'module'):
                model.module.save_pretrained(f"{model_dir}{model_save_suffix}")
            else:
                model.save_pretrained(f"{model_dir}{model_save_suffix}")
    
    return model

def train_single_utterance(config: TrainingConfig,
                         llm: AutoModelForCausalLM,
                         tokenizer: AutoTokenizer,
                         model_dir: str,
                         model: Optional[MambaCompressor] = None) -> MambaCompressor:
    """Train on single utterance data"""
    if config.local_rank <= 0:
        logging.info("Training single utterances")
    
    return train_model(
        config=config,
        llm=llm,
        tokenizer=tokenizer,
        model_dir=model_dir,
        model=model,
        data_preparation_func=prepare_single_utterances_data,
        learning_rate=config.lr_single,
        batch_size=config.batch_size_single,
        num_epochs=config.epochs_single,
        is_step_based=(config.eval_steps > 0),
        model_save_suffix="_single"
    )

def train_conversations(config: TrainingConfig,
                       llm: AutoModelForCausalLM,
                       tokenizer: AutoTokenizer,
                       model,
                       model_dir: str):
    """Train on conversation data"""
    if config.local_rank <= 0:
        logging.info("Training multiple utterances")
    
    return train_model(
        config=config,
        llm=llm,
        tokenizer=tokenizer,
        model_dir=model_dir,
        model=model,
        data_preparation_func=prepare_multiple_utterances_data,
        learning_rate=config.lr_conv,
        batch_size=config.batch_size_conv,
        num_epochs=config.epochs_conv,
        is_step_based=(config.eval_steps > 0),
        model_save_suffix="_conv"
    )
