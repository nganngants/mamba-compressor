#!/bin/bash

python model/train_videollama.py \
    --mamba_path "state-spaces/mamba-370m-hf" \
    --train_data "jsonl_TrainTesTval_VidDescHis_5_latest/train.jsonl" \
    --valid_data "jsonl_TrainTesTval_VidDescHis_5_latest/val.jsonl" \
    --model_dir "./mamba_compressor_videollama" \
    --llm_name "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV" \
    --batch_size_single 1 \
    --batch_size_conv 1 \
    --epochs_single 1 \
    --epochs_conv 1 \
    --lr_single 5e-5 \
    --lr_conv 1e-4 \
    --device cuda \
    --load_in_4bit \
    --compute_dtype float16 \
    --quant_type nf4 \
    --use_double_quant \
    --eval_steps 50 \
    --patience_steps 4 \
    --scheduler_type reduce_on_plateau \
    --scheduler_factor 0.5 \
    --scheduler_min_lr 1e-6 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 4 \
    --gradient_clip_value 5.0 \
    --checkpoint_path mamba_compressor_log_step900_best.pt