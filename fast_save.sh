# 1. Identify the snapshot directory
SNAPSHOT_DIR="/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-MoE-16B-Base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580"

# 2. Create destination directory
DEST_DIR="/workspace/DeepSeek-MoE-16B-pretrained"
mkdir -p $DEST_DIR

# 3. Copy all necessary files
cp $SNAPSHOT_DIR/* $DEST_DIR/
cp $SNAPSHOT_DIR/../*.json $DEST_DIR/  # Copy config/tokenizer files

# 4. Verify
ls -lh $DEST_DIR
