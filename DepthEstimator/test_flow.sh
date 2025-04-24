#!/bin/bash
#SBATCH -J flownet_train           # Job name
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu --gres=gpu:1 --constraint=geforce3090
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
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
python test.py \
  --config_file /users/hzhan351/projects/3D-FoodCalorie/DepthEstimator/config/kitti.yaml \
  --gpu 0 \
  --mode flow \
  --task kitti_flow \
  --pretrained_model /users/hzhan351/scratch/checkpoints/flow_checkpoints/flow/last.pth \
  --result_dir /users/hzhan351/scratch/results

echo "Testing complete!"
