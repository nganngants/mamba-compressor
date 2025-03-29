# model.py
from dataclasses import dataclass
import os
import json
from typing import List
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
    def __init__(self, llm_input_size: int, device: str, tokenizer_len, mem_token_id, mamba_path: str = "state-spaces/mamba-370m-hf"):
        """
        Initialize MambaCompressor with a frozen Qwen LLM for reconstruction training
        Args:
            mamba_path: Path to pretrained Mamba model
            llm_path: Path to Qwen LLM
            torch_dtype: Data type for model loading (helps with DeepSpeed compatibility)
        """
        super().__init__()
        # Add DeepSpeed-specific parameters for model loading
        self.mamba = MambaModel.from_pretrained(
            mamba_path
        )
        
        self.mamba.resize_token_embeddings(tokenizer_len)
        self.mem_token_id = mem_token_id
        
        
        mamba_hidden = self.mamba.config.hidden_size
        self.memory_projection = nn.Linear(mamba_hidden, llm_input_size)
        self.device = device

    def forward(self, input_ids):
        outputs = self.mamba(input_ids["input_ids"].to(self.device)).last_hidden_state
     
        mem_token_mask = input_ids["input_ids"] == self.mem_token_id
       
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

        # Handle different in sequence length
        max_seq_len = max([len(mem) for mem in memory_features])
        for i, mem in enumerate(memory_features):
            if len(mem) < max_seq_len:
                pad_len = max_seq_len - len(mem)
                memory_features[i] = torch.cat([mem, torch.zeros(pad_len, mem.size(-1), device=mem.device)], dim=0)
        
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
        try:
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
        except:
            raise ValueError("Cannot load model")
        return model
