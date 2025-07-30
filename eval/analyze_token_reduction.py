#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from tqdm import tqdm
from typing import Dict, List, Any
from videollama2 import model_init

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser(description="Analyze token reduction by MambaCompressor on test set")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to input JSONL file (with both original and compressed fields)")
    parser.add_argument("--llm_name", type=str, required=True, help="Name or path of the LLM model (for tokenizer)")
    parser.add_argument("--original_field", type=str, default="history_chat_original", help="Field name for original input text")
    parser.add_argument("--compressed_field", type=str, default="history_chat_mamba", help="Field name for compressed input text")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from: {args.llm_name}")
    _, _, tokenizer = model_init(args.llm_name)
    # Add special tokens if needed (same as in inference)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': [
            '<|im_start|>', '<|im_end|>', '<history>', '<video>', '<MEM>'
        ]}
    )

    # Load data
    data = load_jsonl(args.input_jsonl)
    print(f"Loaded {len(data)} samples")

    total_tokens_before = 0
    total_tokens_after = 0
    count = 0

    for sample in tqdm(data, desc="Analyzing samples"):
        orig_text = sample.get(args.original_field, "")
        comp_text = sample.get(args.compressed_field, "")
        if not orig_text.strip() or not comp_text.strip():
            continue
        tokens_before = len(tokenizer(orig_text, truncation=False, return_tensors=None)["input_ids"])
        tokens_after = len(tokenizer(comp_text, truncation=False, return_tensors=None)["input_ids"])
        total_tokens_before += tokens_before
        total_tokens_after += tokens_after
        count += 1

    if count == 0 or total_tokens_before == 0:
        print("No valid samples with both original and compressed text.")
        return

    reduction = total_tokens_before - total_tokens_after
    percent_reduction = 100.0 * reduction / total_tokens_before

    print(f"\nTotal samples analyzed: {count}")
    print(f"Total tokens before compression: {total_tokens_before}")
    print(f"Total tokens after compression: {total_tokens_after}")
    print(f"Total tokens reduced: {reduction}")
    print(f"Percentage reduction: {percent_reduction:.2f}%")

if __name__ == "__main__":
    main()