#!/bin/bash

# Get the number of GPUs available
NUM_GPUS=2

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH

export CUDA_VISIBLE_DEVICES=0,1  # Explicitly set which GPUs to use
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=lo  # Use loopback interface for local multi-GPU
export TORCH_DISTRIBUTED_DEBUG=INFO  # Enable distributed debug info

# Performance settings
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Command to run the training script
deepspeed --num_gpus=$NUM_GPUS \
    model/train_videollama.py \
    --deepspeed \
    --zero_stage 2 \
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
    --weight_decay 0.01 \
    --offload_optimizer