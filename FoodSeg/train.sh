#!/bin/bash
set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate food37

export CUDA_VISIBLE_DEVICES=0,1 

nvidia-smi

cd /home/3D-FoodCalorie/FoodSeg

torchrun \
  --nproc_per_node=2 \
  --master_port=29500 \
  finetune.py \
  --batch_size 8 \
  --num_epochs 30 \
  --dataset_root /home/dataset/FoodSeg103 \
  --save_path /home/checkpoints/MaskRCNN