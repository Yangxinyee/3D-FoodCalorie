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

# Run the testing script
python test.py \
  --config_file /home/3D-FoodCalorie/DepthEstimator/config/kitti.yaml \
  --gpu 0 \
  --mode depth \
  --task kitti_depth \
  --pretrained_model /home/checkpoints/KITTI/depth/last.pth \
  --result_dir /home/results | tee test_depth_K.log

echo "Testing complete!"
