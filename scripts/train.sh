#!/bin/bash

python model/train.py \
    --mamba_path "state-spaces/mamba-370m-hf" \
    --train_data "MESC/train.jsonl" \
    --valid_data "MESC/val.jsonl" \
    --model_dir "./mamba_compressor_log" \
    --llm_name "qwen/Qwen2-7B-Instruct" \
    --batch_size_single 8 \
    --batch_size_conv 1 \
    --epochs_single 3 \
    --epochs_conv 2 \
    --lr_single 2.5e-5 \
    --lr_conv 1e-4