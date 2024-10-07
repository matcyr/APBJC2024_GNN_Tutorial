# A Tutorial for Drug Response Prediction and Graph Neural Network at APBJC 2024

This repository provides a tutorial on building a Graph Neural Network (GNN) for drug response prediction from scratch. The project includes data processing, model layers, and a Jupyter Notebook to run the full pipeline.

## Setting Up the Environment
1. With Conda:
To install the environment from the provided `environment.yml` file, use the following command:

```bash
conda env create -f environment.yml -n <env_name>
conda activate <env_name>
```

2. With pip:
   In your terminal, run
   ```bash
        python3 -m venv my_env
        source my_env/bin/activate
        python3 -m pip install --upgrade pip
    ```
## To start
Run
```bash
jupyter notebook notebooks/tutorial.ipynb
```