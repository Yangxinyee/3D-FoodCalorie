#!/bin/bash
set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate food37

export CUDA_VISIBLE_DEVICES=0,1 

nvidia-smi

cd /home/3D-FoodCalorie/DepthEstimator

torchrun \
  --nproc_per_node=2 \
  --master_port=29500 \
  finetune.py \
  --batch_size 128 \
  --num_epochs 100 \
  --dataset_root /home/dataset/nutrition5k_dataset/imagery/realsense_overhead \
  --save_path /home/checkpoints/Nutrition5K_depth \
  --pretrained_model /home/checkpoints/NYU/depth/best.pth \
  --config_file /home/3D-FoodCalorie/DepthEstimator/config/nyu.yaml \
  --encoder_decoder 2 \
  --few_shot -1
