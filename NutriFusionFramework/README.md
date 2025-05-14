# NutriFusion Framework

## Dataset Structure

```
|-nutrition5k_dataset
    |---imagery
        |---realsense_overhead
            |---Dish1
                |---depth_color.png
                |---rgb.png
            |---Dish2
                |---depth_color.png
                |---rgb.png
            ......
            |---DishM
                |---depth_color.png
                |---rgb.png
        |---rgbd_train_processed.txt  
        |---rgb_in_overhead_train_processed.txt
        |---rgbd_test_processed.txt
        |---rgb_in_overhead_test_processed.txt
```

## Prerequisites

### Environment Setup

1. Create a new conda environment (recommended):
```bash
conda create -n nutrifusion python=3.11
conda activate nutrifusion
```

2. Install PyTorch and CUDA:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Install other required packages:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- Python 3.8+
- PyTorch 2.7.0
- CUDA 12.1
- OpenCV 4.11.0
- NumPy 2.2.5
- Pandas 2.2.3
- Matplotlib 3.10.1
- timm 1.0.15
- scipy 1.15.2

### Hardware Requirements
- NVIDIA GPU with CUDA support
- Minimum 8GB GPU memory recommended
- 16GB+ RAM recommended

## Checkpoints

The `CHECKPOINTS` directory contains the pre-trained ResNet101 weights on Food2K dataset. These weights are used as initialization for the nutritional value prediction model.

## Usage

### Training

Before training the model, you need to:

1. Load the pre-trained ResNet101 weights from Food2K:
```python
resnet101_food2k = torch.load("path/to/weight/file")
```

2. Modify the following paths in the code:
   - Path for predicting nutritional values
   - Training process paths
   - Log file paths
   ```python
   log_file_path = "path/to/your/logs"
   ```

3. Run the training script:
```bash
python train_RGBD_multi_fusion.py --model resnet101 --dataset nutrition_rgbd --data_root path/to/dataset --rgbd --direct_prediction
```

Training parameters:
- `--model`: Model architecture (default: resnet101)
- `--dataset`: Dataset name (default: nutrition_rgbd)
- `--data_root`: Path to the dataset
- `--rgbd`: Enable RGB-D mode (default: RGB and D modes)
- `--direct_prediction`: Enable direct prediction mode

### Testing

Before testing:

1. Update the dataset path in the code:
```python
data_root = "path/to/dataset"
```

2. Run the testing script:
```bash
python test_RGBD_multi_fusion.py
```

Optional testing parameters:
- `--lr`: Learning rate
- `--wd`: Weight decay
- `--b`: Batch size
- `--resume`: Resume from checkpoint
- `--print_freq`: Print frequency
- `--bn_momentum`: BatchNorm momentum

## Dataset Configuration

The dataset metadata is stored in the `nutrition5k_dataset` directory. The dataset organization follows the structure shown in the Project Structure section above.

The dataset includes the following label files:
- Training set:
  - `rgbd_train_processed.txt`
  - `rgb_in_overhead_train_processed.txt`
- Testing set:
  - `rgbd_test_processed.txt`
  - `rgb_in_overhead_test_processed.txt`

## Model Checkpoints

Trained models are saved in the `saved` directory. You can use these checkpoints to resume training or for inference.