#!/bin/bash

# Ensure the Hugging Face CLI is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli not found. Installing..."
    pip install huggingface_hub
fi

# Set the model repository and destination directory
MODEL_REPO="deepseek-ai/deepseek-moe-16b-base"
DEST_DIR="/workspace/deepseek-moe-16b-base"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Download the model files using the Hugging Face CLI
huggingface-cli download "$MODEL_REPO" --local-dir "$DEST_DIR" --local-dir-use-symlinks False

# Confirm completion
echo "Model downloaded to $DEST_DIR"

