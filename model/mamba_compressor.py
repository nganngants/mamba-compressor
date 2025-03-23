# model.py
from dataclasses import dataclass
import os
import json
import torch
import torch.nn as nn
from transformers import (
    MambaModel,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoTokenizer
)
from torch.cuda.amp import autocast
import deepspeed
import logging
logger = logging.getLogger(__name__)

class MambaCompressor(nn.Module):
    def __init__(self, 
             llm_input_size: int, 
             device: str, 
             tokenizer_len: int,
             mem_token_id: int,
             mamba_path: str = "state-spaces/mamba-370m-hf",
             use_cache: bool = True,
             enable_amp: bool = True):
        super().__init__()
        
        # Store config values
        self.device = device
        self.use_cache = use_cache
        self.enable_amp = enable_amp
        self.mem_token_id = mem_token_id
        
        # Initialize model components without moving them
        self.mamba = MambaModel.from_pretrained(mamba_path)
        self.mamba.resize_token_embeddings(tokenizer_len)
        
        self.memory_projection = nn.Linear(self.mamba.config.hidden_size, llm_input_size)
        # Initialize with better weight initialization for linear layer
        nn.init.xavier_uniform_(self.memory_projection.weight)
        nn.init.zeros_(self.memory_projection.bias)
        
        # Add layer norm before projection
        self.pre_projection_norm = nn.LayerNorm(self.mamba.config.hidden_size)
        
        logger.info(
            f"Initialized Mamba model from {mamba_path} "
            f"with hidden size {self.mamba.config.hidden_size} and "
            f"memory projection to size {llm_input_size}. "
            f"Memory token ID: {mem_token_id}"
        )

    def forward(self, input_ids):
        """Process input tokens and extract memory features.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length) 
                                    containing token IDs with special memory tokens

        Returns:
            torch.Tensor: Projected memory features for each batch item
        """
        # Get hidden states from Mamba model - use autocast for mixed precision
        with torch.amp.autocast('cuda'):
            logging.info(f"self.mamba: {self.mamba.device}")
            outputs = self.mamba(
                input_ids['input_ids'].to('cuda'),
                use_cache=self.use_cache
            ).last_hidden_state

            mem_token_mask = input_ids['input_ids'] == self.mem_token_id
           
            # More efficient indexing for memory tokens
            # Find positions of memory tokens
            mem_positions = torch.nonzero(mem_token_mask, as_tuple=True)
            batch_indices = mem_positions[0]
            seq_positions = mem_positions[1]
            
            # Group memory features by batch using more efficient approach
            # Get the count of memory tokens per batch
            batch_size = input_ids['input_ids'].size(0)
            mem_per_batch = torch.bincount(batch_indices, minlength=batch_size)
            max_mem_tokens = mem_per_batch.max().item()
            
            # Create padded tensor for memory features
            memory_features = torch.zeros(
                batch_size, 
                max_mem_tokens, 
                outputs.size(-1), 
            )
            
            # Optimized memory feature extraction
            if mem_token_mask.any():
                # Apply layer norm before projection for more stable gradients
                extracted_features = self.pre_projection_norm(
                    outputs[batch_indices, seq_positions]
                )
                
                # Create offset indices for each batch
                batch_offsets = torch.zeros_like(mem_per_batch)
                batch_offsets[1:] = torch.cumsum(mem_per_batch[:-1], dim=0)
                
                # Create memory feature index mapping
                memory_indices = torch.arange(batch_indices.size(0))
                batch_position = memory_indices - batch_offsets[batch_indices]
                
                # Populate memory features tensor
                memory_features[batch_indices, batch_position] = extracted_features
                
                # Apply memory projection
                outputs = self.memory_projection(
                    memory_features[:, :mem_per_batch.max()]
                )
            else:
                # Handle case where no memory tokens are present
                logger.warning("No memory tokens found in batch")
                outputs = torch.zeros(
                    batch_size, 
                    1,  # At least one memory token
                    self.memory_projection.out_features, 
                )
        
        return outputs

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.mamba.save_pretrained(os.path.join(path, "mamba"))
        torch.save(self.memory_projection.state_dict(), 
                  os.path.join(path, "memory_projection.pt"))
        torch.save(self.pre_projection_norm.state_dict(),
                  os.path.join(path, "pre_projection_norm.pt"))
        
        config = {
            "llm_input_size": self.memory_projection.out_features,
            "mamba_hidden_size": self.mamba.config.hidden_size,
            "mem_token_id": self.mem_token_id,
            "use_cache": self.use_cache,
            "enable_amp": self.enable_amp
        }
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, path: str, device: str, tokenizer_len: int, mem_token_id: int):
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        
        model = cls(
            llm_input_size=config["llm_input_size"],
            device=device,
            tokenizer_len=tokenizer_len,
            mem_token_id=mem_token_id,
            mamba_path=os.path.join(path, "mamba"),
            use_cache=config.get("use_cache", True),
            enable_amp=config.get("enable_amp", True)
        )
        
        model.memory_projection.load_state_dict(
            torch.load(os.path.join(path, "memory_projection.pt"))
        )
        
        # Load pre-projection norm if it exists
        norm_path = os.path.join(path, "pre_projection_norm.pt")
        if os.path.exists(norm_path):
            model.pre_projection_norm.load_state_dict(
                torch.load(norm_path)
            )
        
        return model

@dataclass 
class LLMConfig:
   model_name: str
   device: str = "cuda"
   load_in_4bit: bool = True
   compute_dtype: torch.dtype = torch.float16
   quant_type: str = "nf4"
   use_double_quant: bool = True
def setup_deepspeed_model(model, config, learning_rate):
    """Set up DeepSpeed model, optimizer and scheduler"""
    if config.deepspeed:
        # Create a programmatic DeepSpeed config instead of loading from file
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        # Create PyTorch optimizer first
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=config.weight_decay
        )
        
        ds_config = {
            "train_batch_size": config.batch_size_single * world_size,
            "train_micro_batch_size_per_gpu": config.batch_size_single,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            
            "zero_optimization": {
                "stage": config.zero_stage if hasattr(config, "zero_stage") else 1,  # Lower to stage 1
                "offload_optimizer": {
                    "device": "cpu" if hasattr(config, "offload_optimizer") and config.offload_optimizer else "none",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu" if hasattr(config, "offload_param") and config.offload_param else "none",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": int(2e8),
                "allgather_bucket_size": int(2e8)
            },
            
            "fp16": {
                "enabled": config.fp16 if hasattr(config, "fp16") else True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            
            # Explicitly disable fused kernels
            "zero_allow_untested_optimizer": True,
            "amp": {
                "enabled": config.amp if hasattr(config, "amp") else False,
                "opt_level": "O1"
            },
            
            # Don't define optimizer in config, pass it directly to initialize
            "scheduler": {
                "type": "WarmupDecayLR" if config.scheduler_type == "cosine" else "WarmupLR",
                "params": {
                    "warmup_min_lr": config.scheduler_min_lr,
                    "warmup_max_lr": learning_rate,
                    "warmup_num_steps": config.warmup_steps,
                    "total_num_steps": 1000  # This is an estimate, adjust based on your dataset size
                }
            },
            
            "gradient_clipping": config.gradient_clip_value,
            "steps_per_print": 50
        }
        
        # Print the config for debugging
        if config.local_rank <= 0:
            print("DeepSpeed Config:", ds_config)
        
        # Initialize DeepSpeed with our optimizer
        model_engine, _, _, scheduler = deepspeed.initialize(
            model=model,
            config_params=ds_config,
            optimizer=optimizer,  # Pass our pre-created optimizer
            model_parameters=None  # No need when optimizer is provided
        )
        
        model = model_engine
    else:
        # Traditional PyTorch setup
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Get the appropriate scheduler
        scheduler = get_scheduler(
            config.scheduler_type,
            optimizer,
            config.scheduler_patience,
            config.scheduler_factor,
            config.scheduler_min_lr,
            t_max=1000  # Adjust based on your dataset size
        )
    
    return model, optimizer, scheduler