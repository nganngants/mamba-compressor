#!/bin/bash

deepspeed --num_gpus=2 \
    model/train_qwen2_5.py \
    --deepspeed ds_config.json \
    --mamba_path "state-spaces/mamba-370m-hf" \
    --train_data "jsonl_TrainValMerge_noVidDesc_onlyUtterance_max_emotion_max_strategy/train.jsonl" \
    --valid_data "jsonl_TrainValMerge_noVidDesc_onlyUtterance_max_emotion_max_strategy/val.jsonl" \
    --model_dir "./mamba_compressor_log" \
    --llm_name "Qwen/Qwen2.5-7B-Instruct" \
    --max_length 1024 \
    --batch_size_single 1 \
    --batch_size_conv 1 \
    --epochs_single 3 \
    --epochs_conv 1 \
    --lr_single 2.5e-5 \
    --lr_conv 1e-4 \
    --eval_steps 100 \
    --patience_steps 2 \
    --gradient_accumulation_steps 16 \
    --scheduler_type reduce_on_plateau \
    --lora_r 64 \
    --lora_alpha 128 