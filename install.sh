#!/bin/bash

# Update pip to the latest version
python3 -m pip install -U pip

# Install dependencies for reading and saving data
pip install openpyxl
pip install pandas
pip install pyarrow
pip install fastparquet
# Install dependencies for processing drug data
pip install pubchempy

# Install deep learning dependencies
pip install torch torchvision torchaudio
pip install torch_geometric

# Install machine learning models
pip install -U scikit-learn

# Install visualization tools
pip install -U matplotlib
pip install seaborn

# Install Jupyter Notebook
pip install jupyter

# Print completion message
echo "Environment setup complete."