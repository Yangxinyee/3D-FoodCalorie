#!/bin/bash

# Stop if any command fails
set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate food37

# Check GPU status
nvidia-smi

# Navigate to your project directory
cd /home/3D-FoodCalorie/DepthEstimator

# Check PyTorch and CUDA details
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'NA'); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'NA')"

# Run training script
python train.py \
  --gpus 0,1 \
  --batch_size 256 \
  --num_epochs 30 \
  --dataset_root /home/dataset/FoodSeg103 \
  --save_path /home/checkpoints/MaskRCNN \

echo "Training complete!"
