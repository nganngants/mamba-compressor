#!/bin/bash

deepspeed --num_gpus=4 \
    model/train_videollama.py \
    --deepspeed ds_config.json \
    --mamba_path "state-spaces/mamba-370m-hf" \
    --train_data "jsonl_TrainTesTval_noVidDesc_latest/train.jsonl" \
    --valid_data "jsonl_TrainTesTval_noVidDesc_latest/val.jsonl" \
    --model_dir "./mamba_compressor_log" \
    --llm_name "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV" \
    --batch_size_single 1 \
    --batch_size_conv 1 \
    --epochs_single 3 \
    --epochs_conv 1 \
    --lr_single 2.5e-5 \
    --lr_conv 1e-4 \
    --eval_steps 100 \
    --patience_steps 2 \
    --gradient_accumulation_steps 1 \
    --scheduler_type reduce_on_plateau \
    --lora_r 4 \
    --lora_alpha 8 