#!/bin/bash
#SBATCH -J flownet_train           # Job name
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu --gres=gpu:2 --constraint=geforce3090  
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8          
#SBATCH -o slurm-out/slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haodong_zhang@brown.edu

# Load required modules
module load miniconda3/23.11.0s

# Activate conda environment
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate food37
module load cuda/11.8.0-lpttyok

# uncomment it if using python3.6
# module load cuda/10.2.89-xnfjmrt
# conda activate foodestimator

nvidia-smi

# Navigate to your project directory
cd /users/hzhan351/projects/3D-FoodCalorie/DepthEstimator

python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'NA'); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'NA')"

# Run the training script
python train.py \
  --config_file /users/hzhan351/projects/3D-FoodCalorie/DepthEstimator/config/nyu.yaml \
  --gpu 0,1 \                      
  --multi_gpu \                    
  --mode flow \
  --prepared_save_dir nyuv2_prepared \
  --model_dir /users/hzhan351/scratch/checkpoints/NYUV2 \
  --batch_size 16 \               
  --num_workers 8 \                
  --no_test \
  --lr 0.0002                      

echo "Training complete!"
