#!/bin/bash

echo "Creating Conda environment..."

conda env create -f environment.yml

echo "Activating environment..."

source $(conda info --base)/etc/profile.d/conda.sh
conda activate calories-prediction

echo "Environment successfully created."

python --version

echo "Installed packages:"
pip list
