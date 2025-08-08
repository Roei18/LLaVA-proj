# pick a local, per-node fast path
export SCRATCH=/home/ai_center/ai_users/roeibenzion/VLM-FGA/LLaVA_converter/cache

# Put all HF caches on the fast disk
export HF_HOME=$SCRATCH/hf
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$SCRATCH/torch

mkdir -p "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME"
