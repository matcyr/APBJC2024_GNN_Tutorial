#!/bin/bash

# Update pip to the latest version
python -m pip install -U pip

# Install dependencies for reading and saving data
pip install openpyxl
pip install pandas
pip install pyarrow
pip install fastparquet
# Install dependencies for processing drug data
pip install pubchempy
pip install rdkit
# Install deep learning dependencies
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc -l) -eq 1 ]]; then
    echo "Python version is $PYTHON_VERSION, installing numpy==1.26.4..."
    pip install numpy==1.26.4
else
    echo "Python version is less than 3.10. Current version is $PYTHON_VERSION. Skipping numpy installation."
fi
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