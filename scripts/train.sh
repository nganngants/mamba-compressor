#!/bin/bash

python model/train.py \
    --mamba_path "state-spaces/mamba-370m-hf" \
    --train_data "jsonl_train_test_val/train.jsonl" \
    --valid_data "jsonl_train_test_val/val.jsonl" \
    --model_dir "./mamba_compressor_log" \
    --llm_name "qwen/Qwen2-7B-Instruct" \
    --batch_size_single 4 \
    --batch_size_conv 2 \
    --epochs_single 3 \
    --epochs_conv 2 \
    --lr_single 2.5e-5 \
    --lr_conv 1e-4 \
    --load_in_4bit \
    --compute_dtype float16 \
    --quant_type nf4 \
    --use_double_quant \
    --device cuda