#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import logging
import os
from pathlib import Path
from typing import List
from transformers import AutoTokenizer
import torch.nn as nn

# Import the necessary modules
from model import MambaCompressor
from model import prepare_input
from videollama2 import model_init

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Environment settings for better performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

def generate_text(llm, input_embeds, attention_mask, tokenizer, max_length=50):
    """Generate text from the embeddings"""
    # Prepare generation config
    generation_config = {
        "max_length": max_length,
        "num_beams": 5,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    with torch.no_grad():
        generated_ids = llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            **generation_config
        )
    
    # Decode the generated output
    generated_text = tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )
    
    return generated_text

def run_test(
    mamba_model_path: str,
    llm_name: str,
    test_inputs: List[str],
    system_prompt: str = "You are a helpful assistant. Provided the compressed embeddings, please reconstruct the conversation.",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Run a test with the MambaCompressor and LLM"""
    logger.info(f"Loading LLM: {llm_name}")
    llm, _, tokenizer = model_init(llm_name)
    
    logger.info("Adding special tokens to tokenizer")
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
    
    logger.info(f"Loading MambaCompressor from: {mamba_model_path}")
    mamba_model = MambaCompressor.from_pretrained(
        path=mamba_model_path,
        device=device,
        tokenizer_len=len(tokenizer),
    ).to(device)
    
    # Set models to evaluation mode
    mamba_model.eval()
    llm.eval()
    
    logger.info("Starting test with the following inputs:")
    for i, text in enumerate(test_inputs):
        logger.info(f"Input {i+1}: {text}")
    
    # Tokenize the input texts
    input_ids = tokenizer(
        test_inputs,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512
    ).to(device)
    
    logger.info("Processing through MambaCompressor...")
    with torch.no_grad():
        # Get the memory features from Mamba
        memory_features = mamba_model(input_ids)
    
    logger.info("Preparing input for LLM...")
    # Prepare inputs for the LLM
    input_data = prepare_input(
        mamba_model=mamba_model,
        llm_model=llm,
        llm_tokenizer=tokenizer,
        system_prompt=system_prompt,
        input_texts=test_inputs,
        device=device
    )
    
    logger.info("Generating text from LLM...")
    # Generate reconstructed text from the embeddings
    generated_texts = generate_text(
        llm=llm,
        input_embeds=input_data['input_embeds'],
        attention_mask=input_data['attention_mask'],
        tokenizer=tokenizer,
        max_length=512
    )
    
    logger.info("\n\n===== RESULTS =====")
    for i, (original, generated) in enumerate(zip(test_inputs, generated_texts)):
        logger.info(f"\nTest {i+1}:")
        logger.info(f"Original: {original}")
        logger.info(f"Generated: {generated}")
        logger.info("=" * 50)
    
    return generated_texts

def main():
    parser = argparse.ArgumentParser(description="Test MambaCompressor with VideoLLaMA2")
    parser.add_argument("--mamba_model_path", type=str, required=True, help="Path to trained MambaCompressor model")
    parser.add_argument("--llm_name", type=str, required=True, help="Name or path of the LLM model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (cuda/cpu)")
    args = parser.parse_args()
    
    # Hardcoded test inputs
    test_inputs = ["""
    user ( user Emotion : anger , user Expresstion : I mean, shit, don't you know that men are the new women? The speaker seems to be in a calm and relaxed state. She is sitting on a couch and looking at the camera with a smile on her face.The speaker's emotional expression in this picture could be due to a variety of reasons, including her personal life, relationships, or work-related stress. However, without further information, it is difficult to determine the specific cause of her emotional state. . Obsessed with weddings and children. The speaker seems to be in a state of contemplation or deep thought, as he is looking down and talking on the phone while sitting at a table.The speaker's emotional expression could be due to a personal or professional problem that he is trying to solve. . And lately he's lost touch with reality. The speaker seems to be in a calm and relaxed state, as she is sitting on a couch and talking to someone.The speaker's emotional expression in this picture could be a sign of sadness or depression, as she is sitting on a couch and talking to someone in a calm and relaxed state. . He he thinks I'm seeing someone. That's how this whole thing started. The speaker appears to be calm and relaxed throughout the video.The speaker's emotional expression could be due to a variety of reasons, such as stress, anxiety, or frustration. However, without more context, it is difficult to determine the specific cause of the speaker's emotional state. . It's unbelievable. The speaker appears to be in a state of contemplation, as he is sitting in a chair and looking down. He seems to be deep in thought, possibly pondering a difficult decision or problem.The speaker's emotional expression could be a sign of stress or frustration, as he is sitting in a chair and looking down. It is possible that he is dealing with a difficult situation or problem that is causing him to feel overwhelmed or upset.)  : I mean, shit, don't you know that men are the new women? Obsessed with weddings and children. And lately he's lost touch with reality. He he thinks I'm seeing someone. That's how this whole thing started. It's unbelievable. <MEM>  sys ( sys Emotion : neutral , sys Strategy : Restatement ) : But I thought you said it came out of nowhere. You were watching tv, then suddenly... <MEM> 
    """
    # "user ( user Emotion : anger ): I mean, shit, don't you know that men are the new women? Obsessed with weddings and children. And lately he's lost touch with reality. He he thinks I'm seeing someone. That's how this whole thing started. It's unbelievable. <MEM>"
    ]
    
    # System prompt
    system_prompt = "You are a helpful assistant. Provided the compressed embeddings, please reconstruct the conversation."
    
    # Run the test
    run_test(
        mamba_model_path=args.mamba_model_path,
        llm_name=args.llm_name,
        test_inputs=test_inputs,
        system_prompt=system_prompt,
        device=args.device
    )

if __name__ == "__main__":
    main()