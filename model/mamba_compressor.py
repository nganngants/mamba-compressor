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

import logging
logger = logging.getLogger(__name__)

class MambaCompressor(nn.Module):
    def __init__(self, 
                 llm_input_size: int, 
                 device: str, 
                 tokenizer_len: int,
                 mem_token_id: int,
                 mamba_path: str = "state-spaces/mamba-370m-hf"):
        super().__init__()
        self.mamba = MambaModel.from_pretrained(mamba_path)
        self.mamba.resize_token_embeddings(tokenizer_len)
        self.mem_token_id = mem_token_id
        self.memory_projection = nn.Linear(self.mamba.config.hidden_size, llm_input_size)
        self.device = device
        logger.info(
            f"Loaded Mamba model from {mamba_path} "
            f"with hidden size {self.mamba.config.hidden_size} and "
            f"memory projection to size {llm_input_size}"
            f"Memory token ID: {mem_token_id}"
        )

    def forward(self, input_ids):
        """Process input tokens and extract memory features.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length) 
                                    containing token IDs with special memory tokens

        Returns:
            torch.Tensor: Projected memory features for each batch item

        The function performs these steps:
        1. Passes input through Mamba model to get hidden states
        2. Locates all <MEM> tokens in the input
        3. Extracts features at <MEM> token positions
        4. Projects features to target LLM dimension
        """
        # Get hidden states from Mamba model
        outputs = self.mamba(input_ids['input_ids'].to(self.device)).last_hidden_state

        mem_token_mask = input_ids['input_ids'] == self.mem_token_id
       
        batch_indices = torch.arange(outputs.size(0), device=outputs.device)[:, None]
        mem_positions = mem_token_mask.nonzero()
        batch_nums = mem_positions[:, 0]
        seq_positions = mem_positions[:, 1]
        
        # Group memory features by batch
        memory_features = []
        for batch_idx in range(outputs.size(0)):
            batch_mask = batch_nums == batch_idx
            batch_positions = seq_positions[batch_mask]
            batch_features = outputs[batch_idx, batch_positions]
            memory_features.append(batch_features)
        
        memory_features = torch.stack(memory_features)

        outputs = self.memory_projection(memory_features)
        
        return outputs

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.mamba.save_pretrained(os.path.join(path, "mamba"))
        torch.save(self.memory_projection.state_dict(), 
                  os.path.join(path, "memory_projection.pt"))
        
        config = {
            "llm_input_size": self.memory_projection.out_features,
            "mamba_hidden_size": self.mamba.config.hidden_size,
            "device": self.device,
            "mem_token_id": self.mem_token_id
        }
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, path: str, device: str, tokenizer_len: int):
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        
        model = cls(
            llm_input_size=config["llm_input_size"],
            device=device,
            tokenizer_len=tokenizer_len,
            mem_token_id=config["mem_token_id"],
            mamba_path=os.path.join(path, "mamba")
        )
        
        model.memory_projection.load_state_dict(
            torch.load(os.path.join(path, "memory_projection.pt"), 
                      map_location=device)
        )
        return model

@dataclass 
class LLMConfig:
   model_name: str
   device: str = "cuda" if torch.cuda.is_available() else "cpu"
   load_in_4bit: bool = True
   compute_dtype: torch.dtype = torch.float16
   quant_type: str = "nf4"
   use_double_quant: bool = True
