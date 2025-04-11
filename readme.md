# A Tutorial for Drug Response Prediction and Graph Neural Network at APBJC 2024

Contact: **Yurui Chen**, [chenyr@sics.a-star.edu.sg](mailto:chenyr@sics.a-star.edu.sg)

This repository provides a tutorial on building a Graph Neural Network (GNN) for drug response prediction from scratch. The project includes data processing, model layers, and a Jupyter Notebook to run the full pipeline.

- Drug response prediction plays a crucial role in precision medicine, aiming to determine how various drugs will impact specific cell lines or patients, ultimately assisting in creating personalized treatment strategies.

- The GNN approach is highly effective for learning representations from small molecule graphs, making it well-suited for this type of task.

We will cover data processing, building the model, and running predictions step by step. You are expected to:
1. Get familiar with the drug response prediction problem.
2. Gain basic knowledge of the Graph Neural Network (GCN and GIN).
3. Understand the pipeline to train and evaluate machine learning and deep learning models for drug responses.

We will use the [GDSC](https://www.cancerrxgene.org/) project as the drug response dataset.


The github repo is:
https://github.com/matcyr/APBJC2024_GNN_Tutorial
short link to the repo:

## *bit.ly/3AaFxDO*

We will use **colab** to run the tutorial. Please click on <a href="https://colab.research.google.com/github/matcyr/APBJC2024_GNN_Tutorial/blob/master/notebooks/APBJC_tutorial_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

or scan:

<img src="qr-code_colab.png" alt="Open In Colab" width="600px"/>





## For experienced python users:
To start, in your terminal, run:
```bash
git clone https://github.com/matcyr/APBJC2024_GNN_Tutorial.git
```
Then go to the project directory, by:
```bash
cd APBJC2024_GNN_Tutorial
```

## Setting Up the Environment
The tutorial is based on Python. The key dependencies include:
- `pandas`
- `numpy`
- `torch`
- `torch_geometric`


**Note**: The provided dependencies are configured for CPU usage only. If you want to train the model using a GPU, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/) and [PyTorch Geometric website](https://pytorch-geometric.readthedocs.io/en/latest/) for instructions on setting up a GPU-backed environment.

### 1. Conda + Linux
If you have a linux platform with conda installed, you can simply install the environment from the provided `environment.yml` file. Run the following commands in your terminal:

```bash
conda env create -f environment.yml -n <env_name>
conda activate <env_name>
```

### 2. Pip
If you do not have Conda installed, or if you are using a Mac/Windows laptop. You can set up the environment using pip by following these instructions:

#### Step 1: Create a Virtual Environment
Run the following command in your terminal to create a virtual environment:

```bash
python -m venv <env_name>
```
or:
```bash
python3 -m venv <env_name>
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

## The Tutorial Notebook
Run the following command to open the tutorial notebook:
```bash
jupyter notebook notebooks/APBJC_tutorial.ipynb
```
