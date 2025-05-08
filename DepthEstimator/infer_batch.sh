#!/bin/bash

# Set error exit
set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate food37

# Check GPU status
nvidia-smi

# Navigate to your project directory
cd /home/zhd/3D-FoodCalorie/DepthEstimator

# Check PyTorch and CUDA availability
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'NA'); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'NA')"

python infer.py \
  --config_file /home/3D-FoodCalorie/DepthEstimator/config/nyu.yaml \
  --gpu 0 \
  --task batch \
  --pretrained_model /home/checkpoints/Nutrition5K_depth/last.pth \
  --min_depth 0.01 \
  --max_depth 1.2 \
  --root_dir /home/tests | tee infer_demo.log

# /home/checkpoints/Nutrition5K_depth/depth_epoch_3.pth
# /home/checkpoints/NYU/depth/best.pth
