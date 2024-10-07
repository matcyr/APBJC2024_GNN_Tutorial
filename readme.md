# A Tutorial for Drug Response Prediction and Graph Neural Network at APBJC 2024

This repository provides a tutorial on building a Graph Neural Network (GNN) for drug response prediction from scratch. The project includes data processing, model layers, and a Jupyter Notebook to run the full pipeline.

To start, in your terminal, run:
```bash
git clone https://github.com/matcyr/APBJC2024_GNN_Tutorial.git
```

## Setting Up the Environment
The tutorial is based on Python. The key dependencies include:
- `pandas`
- `numpy`
- `torch`
- `torch_geometric`

### 1. Conda + Linux/Windows
To install the environment from the provided `environment.yml` file, run the following commands in your terminal:

```bash
conda env create -f environment.yml -n <env_name>
conda activate <env_name>
```

### 2. Pip
If you do not have Conda installed, or if you are using a Mac. You can set up the environment using pip by following these instructions:

#### Step 1: Create a Virtual Environment
Run the following command in your terminal to create a virtual environment:

```bash
python -m venv <env_name>
```

#### Step 2: Activate the Virtual Environment
- **macOS/Linux**: Run the following command in your terminal:
  
  ```bash
  source <env_name>/bin/activate
  ```

- **Windows**: Run the following command in your **Command Prompt** (cmd):
  
  ```
  <env_name>\Scripts\activate
  ```

#### Step 3: Install Dependencies
After activating the virtual environment, run the following command to install the required dependencies:

```bash
sh install.sh
```

This will create a virtual environment named `<env_name>` and install the required dependencies using the `install.sh` script.

## To Start
Run the following command to open the tutorial notebook:
```bash
jupyter notebook notebooks/tutorial.ipynb
```
