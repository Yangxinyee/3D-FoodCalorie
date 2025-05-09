#!/bin/bash
set -e

# Activate conda env
source $(conda info --base)/etc/profile.d/conda.sh
conda activate food37

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0

# Show GPU usage
nvidia-smi

# Change to project directory
cd /home/3D-FoodCalorie/FoodSeg

# Run inference
python inference.py \
  --model_path /home/mrcnn_foodseg103_best.pth \
  --input_dir /home/dataset/nutrition5k_dataset/imagery/realsense_overhead \
  --output_dir /home/dataset/nutrition5k_dataset/imagery/mrcnn_output \
  --category_path /home/dataset/FoodSeg103/category_id.txt