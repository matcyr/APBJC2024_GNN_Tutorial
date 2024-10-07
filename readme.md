# A Tutorial for Drug Response Prediction and Graph Neural Network at APBJC 2024

This repository provides a tutorial on building a Graph Neural Network (GNN) for drug response prediction from scratch. The project includes data processing, model layers, and a Jupyter Notebook to run the full pipeline.

To start, in your terminal, and run:
```bash
git clone https://github.com/matcyr/APBJC2024_GNN_Tutorial.git
```

## Setting Up the Environment
The tutorial is based on Python.
### Key Dependencies
- `pandas`
- `numpy`
- `torch`
- `torch_geometric`

### 1. If You Have Conda Installed
To install the environment from the provided `environment.yml` file, run the following commands in your terminal:

```bash
conda env create -f environment.yml -n <env_name>
conda activate <env_name>
```

### 2. Without Conda (Using pip)
If you prefer to set up the environment using pip, run the following commands in your terminal:

```bash
python3 -m venv <env_name>
source <env_name>/bin/activate
sh install.sh
```

This will create a virtual environment named `<env_name>` and install the required dependencies using the `install.sh` script.

## To start
Run
```bash
jupyter notebook notebooks/tutorial.ipynb
```