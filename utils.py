import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
import zipfile
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def process_molecule(smiles):
    # Convert SMILES to a molecule object
    molecule = Chem.MolFromSmiles(smiles)

    # Get the number of atoms in the molecule
    num_atoms = molecule.GetNumAtoms()

    # Initialize the adjacency matrix (num_atoms x num_atoms)
    adj_matrix = rdmolops.GetAdjacencyMatrix(molecule)

    # Get atom types as node features (atomic numbers and symbols)
    atom_types = []
    atom_symbols = []
    atomic_num_symbol_map = {}
    
    for atom in molecule.GetAtoms():
        atom_num = atom.GetAtomicNum()
        atom_symbol = atom.GetSymbol()
        
        atom_types.append(atom_num)  # Use atomic numbers as node features
        atom_symbols.append(atom_symbol)  # Use atomic symbols as labels
        
        # Map atomic number to symbol if not already in the map
        if atom_num not in atomic_num_symbol_map:
            atomic_num_symbol_map[atom_num] = atom_symbol

    # Get the edge list (undirected)
    edge_list = []
    for bond in molecule.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        edge_list.append((atom1, atom2))
        edge_list.append((atom2, atom1))

    return atom_types, adj_matrix, edge_list, atomic_num_symbol_map

def generate_colors(atomic_num_symbol_map):
    num_atom_types = len(atomic_num_symbol_map)
    
    # Use the updated colormap call
    cmap = plt.get_cmap('tab10', num_atom_types)  # Get a colormap with distinct colors
    color_map = {}
    
    for i, atomic_num in enumerate(atomic_num_symbol_map.keys()):
        color_map[atomic_num] = cmap(i)  # Assign a color for each unique atom type
    
    return color_map
def visualize_molecule_graph(edge_list, atom_types, atomic_num_symbol_map):
    # Dynamically generate a color map based on atom types
    atom_type_color_map = generate_colors(atomic_num_symbol_map)

    # Create the graph using NetworkX
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # Create a list of node colors based on atom types
    node_colors = [atom_type_color_map[atom_type] for atom_type in atom_types]

    # Plot the graph
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G, seed=42)  # Positioning of the graph nodes
    
    # Draw nodes with corresponding colors (use node index as labels)
    nx.draw(G, pos, with_labels=True, labels={i: i for i in G.nodes()}, 
            node_color=node_colors, node_size=500, font_size=10)
    
    # Create the legend manually for the atom types based on atomic_num_symbol_map
    legend_labels = {atomic_num_symbol_map[num]: plt.Line2D([0], [0], marker='o', color='w', 
                     markerfacecolor=atom_type_color_map[num], markersize=10)
                     for num in atomic_num_symbol_map}
    
    plt.legend(legend_labels.values(), legend_labels.keys(), title="Atom Types", loc="upper left")
    plt.title("Molecular Graph with Node Indices and Atom Type Colors")
    plt.show()



class GDSCProcessor:
    """
    The GDSCProcessor class automates the download, processing, and preparation of drug response, drug metadata, and gene expression data
    from the Genomics of Drug Sensitivity in Cancer (GDSC) project. This class organizes the data into a final dataset that can be used 
    for further analysis, particularly in drug response predictions.
    
    Functions:
    - download_gdsc_data: Downloads the GDSC2 IC50 data if not already downloaded. 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx'
    - process_drug_meta: Downloads and processes drug metadata, retrieves SMILES using PubChem API, and saves results. https://www.cancerrxgene.org/compounds
    - process_gene_expression: Downloads and processes gene expression data, and matches it with the drug response data. https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources
    - process_final_dataframe: Combines drug response data with drug metadata and gene expression data to produce a final dataset.
    - run: Executes all the steps to process and produce the final dataset.
    """
    def __init__(self, gdsc_link, drug_meta_link, exp_data_link, data_path = './Data/', verbose = True):
        """
        Initializes the GDSCProcessor class with URLs for downloading data and setting up the data directory.
        
        Parameters:
        - gdsc_link (str): URL to download the GDSC2 fitted dose response data.
        - drug_meta_link (str): URL to download the drug metadata (PubChem, SMILES, etc.).
        - exp_data_link (str): URL to download gene expression data.
        - data_path (str): Directory to save the processed data (default: '../Data/').
        """
        self.gdsc_link = gdsc_link
        self.drug_meta_link = drug_meta_link
        self.exp_data_link = exp_data_link
        self.data_path = data_path
        self.verbose = verbose
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

    def download_gdsc_data(self):
        gdsc_file = os.path.join(self.data_path, 'GDSC2_fitted_dose_response_27Oct23.xlsx')
        if not os.path.exists(gdsc_file):
            print(f"Downloading GDSC data from {self.gdsc_link}")
            os.system(f"wget -O {gdsc_file} {self.gdsc_link}")
        self.df = pd.read_excel(gdsc_file)
        if self.verbose:
            print(self.df.head())
    
    def process_drug_meta(self):
        """
        Downloads and processes the drug metadata, retrieving SMILES from the PubChem API if necessary.
        The processed drug metadata is saved as 'drug_meta.csv'.
        If the file already exists, it loads it from the disk.
        """
        drug_meta_file = os.path.join(self.data_path, 'drug_meta.csv')
        if not os.path.exists(drug_meta_file):
            print('-----------------')
            print(f"Downloading Drug SMILES data")
            self.drug_meta = pd.read_csv(self.drug_meta_link)
            self.drug_meta.columns = self.drug_meta.columns.str.strip()
            ## Keep only the GDSC2 drugs
            self.drug_meta = self.drug_meta[self.drug_meta['Datasets'] == 'GDSC2']
            ## Keep only drugs without missing values in 'PubCHEM'
            self.drug_meta = self.drug_meta[~self.drug_meta['PubCHEM'].isna()]
            ## Keep the unique drug by the name
            self.drug_meta.drop_duplicates(subset='Name', inplace=True, keep='first')
            ## Drop the PubCHEM with value 'several'
            self.drug_meta = self.drug_meta[self.drug_meta['PubCHEM'] != 'several']
            ## Drop the PubCHEM with value 'none'
            self.drug_meta = self.drug_meta[self.drug_meta['PubCHEM'] != 'none']
            self.drug_meta['PubCHEM'] = self.drug_meta['PubCHEM'].apply(lambda x: x.split(',')[0])

            def get_smiles(pubchem_id):
                try:
                    compound = pcp.Compound.from_cid(pubchem_id)
                    print(f"SMILES for {pubchem_id} is {compound.isomeric_smiles}")
                    return compound.isomeric_smiles
                except Exception as e:
                    return f"Error: {e}"

            self.drug_meta['SMILES'] = self.drug_meta['PubCHEM'].apply(get_smiles)
            self.drug_meta.to_csv(drug_meta_file, index=False)
        else:
            print(f"Drug Meta data already exists")
            self.drug_meta = pd.read_csv(drug_meta_file)
        if self.verbose:
            print(self.drug_meta.head())

    def process_gene_expression(self):
        """
        Downloads and processes gene expression data. It matches the COSMIC IDs from the drug response data 
        with the gene expression data. The processed gene expression data is saved as 'rnaseq_df.csv'.
        """
        exp_df_file = os.path.join(self.data_path, 'rnaseq_df.parquet')
        if not os.path.exists(exp_df_file):
            print('-----------------')
            print(f"Downloading Gene Expression data")
            self.exp_df = pd.read_csv(self.exp_data_link, compression="zip", sep="\t")
            self.exp_df = self.exp_df.set_index("GENE_SYMBOLS").iloc[:, 1:].T
            self.exp_df.index = self.exp_df.index.str.extract("DATA.([0-9]+)").to_numpy().squeeze()
            self.exp_df.reset_index(drop=False).groupby("index").first()
            self.exp_df.index = self.exp_df.index.astype(int)
            common_cosmic_ids = self.df['COSMIC_ID'][self.df['COSMIC_ID'].isin(self.exp_df.index)].unique()
            self.df = self.df[self.df['COSMIC_ID'].isin(common_cosmic_ids)]
            self.exp_df = self.exp_df.loc[common_cosmic_ids]
            self.exp_df = self.exp_df.loc[~self.exp_df.index.duplicated(keep='first')]
            self.exp_df.columns = self.exp_df.columns.str.strip()
            self.exp_df.columns = self.exp_df.columns.str.upper()
            nan_columns_idx = np.where(self.exp_df.columns.isna())[0].tolist()
            for i, col_idx in enumerate(nan_columns_idx):
                new_name = f"unknown_gene_{i + 1}"  # Generate unique names
                self.exp_df.columns.values[col_idx] = new_name
            self.exp_df = self.exp_df.astype('float32')
            self.exp_df.columns = self.exp_df.columns.str.strip()
            self.exp_df.to_parquet(exp_df_file)
        else:
            print(f"Gene Expression data already exists")
            self.exp_df = pd.read_parquet(exp_df_file)

    def process_final_dataframe(self):
        # Filter the df for matching COSMIC_IDs in exp_df
        if not os.path.exists(os.path.join(self.data_path, 'GDSC2_df.csv')):
            self.df = self.df[self.df.COSMIC_ID.isin(self.exp_df.index)]

            # Merge with drug_meta to get PubCHEM
            GDSC2_df = self.df.merge(self.drug_meta[['Drug Id', 'PubCHEM']], left_on='DRUG_ID', right_on='Drug Id')[['PubCHEM', 'COSMIC_ID', 'LN_IC50']]
            self.GDSC2_df = GDSC2_df[GDSC2_df['COSMIC_ID'].isin(self.exp_df.index)]

            # Pivot the GDSC2_df to a table with PubCHEM as columns, COSMIC_ID as index, and LN_IC50 as values
            # GDSC2_df = GDSC2_df.pivot(index='COSMIC_ID', columns='PubCHEM', values='LN_IC50')
            GDSC2_df.to_csv(os.path.join(self.data_path, 'GDSC2_df.csv'))
            if self.verbose:
                print(GDSC2_df.head())
        else:
            print(f"Final DataFrame already exists")
            self.GDSC2_df = pd.read_csv(os.path.join(self.data_path, 'GDSC2_df.csv'), index_col=0)
            if self.verbose:
                print(GDSC2_df.head())

    def run(self):
        # Step 1: Download GDSC Data
        self.download_gdsc_data()

        # Step 2: Process Drug Meta Data
        self.process_drug_meta()

        # Step 3: Process Gene Expression Data
        self.process_gene_expression()

        # Step 4: Process the final DataFrame
        self.process_final_dataframe()



def train(model, train_loader, criterion, optimizer, device):
    '''
    Trains the given model for one epoch.
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        device (torch.device): Device on which to perform training (e.g., 'cpu' or 'cuda').
    Returns:
        float: The loss over the training dataset for the epoch.
    '''
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        cell_feat, drug_feat, IC50 = batch[0], batch[1], batch[2]
        cell_feat, drug_feat, IC50 = cell_feat.to(device), drug_feat.to(device), IC50.to(device)
        optimizer.zero_grad()
        outputs = model(cell_feat, drug_feat)
        loss = criterion(outputs, IC50.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * cell_feat.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def test(model, test_loader, device):
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for batch in test_loader:
            cell_feat, drug_feat, IC50 = batch[0], batch[1], batch[2]
            cell_feat, drug_feat, IC50 = cell_feat.to(device), drug_feat.to(device), IC50.to(device)
            outputs = model(cell_feat, drug_feat)
            predictions.extend(outputs.view(-1).detach().cpu().numpy())
            true_values.extend(IC50.detach().cpu().numpy())
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    return predictions, true_values

def test_metric(true_values, predictions):
    rmse = root_mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    pcc, _ = pearsonr(true_values, predictions)
    return rmse, mae, r2, pcc