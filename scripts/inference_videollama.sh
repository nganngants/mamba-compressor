#!/bin/bash

# Directory setup - modify these paths as needed
DATA_DIR="./data"
OUTPUT_DIR="./hist_jsonl"
EMBEDS_DIR="./hist_embeddings"
MAMBA_MODEL_PATH=""
LLM_NAME="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
INFERENCE_SCRIPT="./eval/inference_videollama.py"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $EMBEDS_DIR

# Run the inference script
python run_inference_on_datasets.py \
  --inference_script $INFERENCE_SCRIPT \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --embeds_dir $EMBEDS_DIR \
  --mamba_model_path $MAMBA_MODEL_PATH \
  --llm_name $LLM_NAME \
  --batch_size 8 \
  --device cuda