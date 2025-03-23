#!/bin/bash

# Get the number of GPUs available
NUM_GPUS=2

# Basic CUDA setup
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# DeepSpeed specific settings
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Performance optimization
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

deepspeed \
    --num_gpus=$NUM_GPUS \
    model/train_videollama.py \
    --deepspeed \
    --zero_stage 1 \
    --train_data jsonl_TrainTesTval_VidDescHis_5_latest/train.jsonl \
    --valid_data jsonl_TrainTesTval_VidDescHis_5_latest/val.jsonl \
    --model_dir ./mamba_compressor_videollama \
    --llm_name DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
    --mamba_path state-spaces/mamba-370m-hf \
    --batch_size_single 1 \
    --batch_size_conv 1 \
    --epochs_single 1 \
    --epochs_conv 1 \
    --eval_steps 50 \
    --patience_steps 4 \
    --gradient_clip_value 1.0 \
    --gradient_accumulation_steps 4 \
    --fp16 \
    --warmup_steps 100 \
    --scheduler_type cosine \
    --scheduler_min_lr 1e-6 \
    --weight_decay 0.01