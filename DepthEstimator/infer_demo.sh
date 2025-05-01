#!/bin/bash

# Set error exit
set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate food37

# Check GPU status
nvidia-smi

# Navigate to your project directory
cd /home/3D-FoodCalorie/DepthEstimator

# Check PyTorch and CUDA availability
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'NA'); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'NA')"

python infer.py \
  --config_file /home/3D-FoodCalorie/DepthEstimator/config/kitti_3stage.yaml \
  --gpu 0 \
  --task demo \
  --pretrained_model /home/checkpoints/KITTI/depth_pose/best.pth \
  --result_dir /home/results \
  --indices 0,5,10,80,100 | tee infer_demo.log