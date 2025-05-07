#!/bin/bash

# Stop if any command fails
set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate food37

# 显式绑定你想用的GPU（非常关键！）
export CUDA_VISIBLE_DEVICES=0,1

# 查看GPU状态
nvidia-smi

# 进入代码目录
cd /home/3D-FoodCalorie/FoodSeg

# 检查PyTorch环境
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'NA'); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'NA')"

# 运行训练（使用rank 0和1对应的两个设备）
torchrun \
  --nproc_per_node=2 \
  --master_port=29500 \
  finetune.py \
  --batch_size 8 \
  --num_epochs 30 \
  --dataset_root /home/dataset/FoodSeg103 \
  --save_path /home/checkpoints/MaskRCNN

echo "Training complete!"
