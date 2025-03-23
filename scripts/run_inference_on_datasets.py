#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import subprocess
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('inference.log')]
)
logger = logging.getLogger(__name__)

def run_inference(
    inference_script: str,
    input_jsonl: str,
    output_jsonl: str,
    output_embeds_dir: str,
    mamba_model_path: str,
    llm_name: str,
    batch_size: int,
    device: str
):
    """Run inference on a single JSONL file"""
    cmd = [
        "python", inference_script,
        "--input_jsonl", input_jsonl,
        "--output_jsonl", output_jsonl,
        "--output_embeds_dir", output_embeds_dir,
        "--mamba_model_path", mamba_model_path,
        "--llm_name", llm_name,
        "--batch_size", str(batch_size),
        "--device", device
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            logger.info(line.strip())
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Inference failed for {input_jsonl} with return code {process.returncode}")
            return False
        else:
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully processed {input_jsonl} in {elapsed_time:.2f} seconds")
            return True
            
    except Exception as e:
        logger.error(f"Error running inference on {input_jsonl}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run inference on multiple JSONL datasets")
    parser.add_argument("--inference_script", type=str, required=True, 
                        help="Path to the inference script")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing train.jsonl, val.jsonl, and test.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save output JSONL files")
    parser.add_argument("--embeds_dir", type=str, required=True, 
                        help="Directory to save embedding H5PY files")
    parser.add_argument("--mamba_model_path", type=str, required=True, 
                        help="Path to trained MambaCompressor model")
    parser.add_argument("--llm_name", type=str, required=True, 
                        help="Name or path of the LLM model")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run on (cuda/cpu)")
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define dataset files
    datasets = ["train", "val", "test"]
    
    for dataset in datasets:
        input_jsonl = os.path.join(args.data_dir, f"{dataset}.jsonl")
        
        # Skip if file doesn't exist
        if not os.path.isfile(input_jsonl):
            logger.warning(f"Dataset file {input_jsonl} not found, skipping.")
            continue
            
        # Create dataset-specific output directory for embeddings
        dataset_embeds_dir = os.path.join(args.embeds_dir, dataset)
        os.makedirs(dataset_embeds_dir, exist_ok=True)
        
        output_jsonl = os.path.join(args.output_dir, f"{dataset}_with_embeds.jsonl")
        
        logger.info(f"Processing {dataset} dataset")
        success = run_inference(
            inference_script=args.inference_script,
            input_jsonl=input_jsonl,
            output_jsonl=output_jsonl,
            output_embeds_dir=dataset_embeds_dir,
            mamba_model_path=args.mamba_model_path,
            llm_name=args.llm_name,
            batch_size=args.batch_size,
            device=args.device
        )
        
        if success:
            logger.info(f"Successfully processed {dataset} dataset")
        else:
            logger.error(f"Failed to process {dataset} dataset")
    
    logger.info("All datasets processed!")

if __name__ == "__main__":
    main()