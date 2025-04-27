#!/bin/bash

# 1. Set environment name
ENV_NAME=food37

# 2. Create a new conda environment with Python 3.7
echo "Creating conda environment: $ENV_NAME with Python 3.7..."
conda create -y -n $ENV_NAME python=3.7

# 3. Activate the environment
echo "Activating environment: $ENV_NAME"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 4. Install required packages from requirements.txt
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

# 5. Done
echo "Environment $ENV_NAME setup completed!"
echo "To activate it manually later, run: conda activate $ENV_NAME"
