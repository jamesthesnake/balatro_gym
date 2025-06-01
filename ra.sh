# Set paths
SRC="/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-MoE-16B-Base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
DEST="/workspace/DeepSeek-MoE-16B-complete"

# Copy ALL files including Python modules
mkdir -p $DEST
cp -r $SRC/* $DEST/
cp $SRC/../*.py $DEST/ 2>/dev/null || true
cp $SRC/../*.json $DEST/
