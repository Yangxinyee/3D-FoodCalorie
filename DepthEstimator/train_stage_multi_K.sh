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

python train.py \
  --config_file /home/3D-FoodCalorie/DepthEstimator/config/kitti_3stage.yaml \
  --gpu 0,1,2,3 \
  --multi_gpu \
  --mode depth_pose \
  --prepared_save_dir KITTI_prepared \
  --model_dir /home/checkpoints/KITTI \
  --depth_pretrained_model /home/checkpoints/KITTI/depth/last.pth \
  --batch_size 8 \
  --num_workers 4 \
  --no_test \
  --lr 0.0002

echo "Training complete!"