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
# Get Python version in major.minor format
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

# Function to compare Python versions
version_compare() {
    # Compare two versions passed as arguments
    if [[ $1 == $2 ]]; then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ ${ver1[i]} -lt ${ver2[i]} ]]; then
            return 1
        elif [[ ${ver1[i]} -gt ${ver2[i]} ]]; then
            return 0
        fi
    done
    return 0
}

# Python version threshold for numpy installation (3.10)
REQUIRED_VERSION="3.10"

# Compare the current Python version with the required version
if version_compare "$PYTHON_VERSION" "$REQUIRED_VERSION"; then
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