import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch

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
    pos = nx.spring_layout(G)  # Positioning of the graph nodes
    
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

class GraphProcessor:
    def __init__(self, edge_list):
        """
        Initialize the GraphProcessor with an edge list and number of nodes.
        Parameters:
        - edge_list: List of edges in the graph (undirected).
        """
        self.edge_list = edge_list
        self.num_nodes = max(torch.tensor(edge_list).flatten()) + 1
        self.adj_matrix = self.build_adj_matrix()
        
    def build_adj_matrix(self):
        """
        Converts the edge list to an adjacency matrix.
        Returns:
        - Adjacency matrix (num_nodes x num_nodes).
        """
        adj_matrix = torch.zeros((self.num_nodes, self.num_nodes))
        for edge in self.edge_list:
            adj_matrix[edge[0], edge[1]] = 1
        return adj_matrix

    def add_self_loops(self):
        """
        Adds self-loops to the adjacency matrix.
        """
        self.adj_matrix += torch.eye(self.num_nodes)

    def normalize_adj_matrix(self):
        """
        Normalizes the adjacency matrix using D^(-1/2) * A * D^(-1/2).
        Returns:
        - Normalized adjacency matrix.
        """
        # Compute degree matrix (sum of connections for each node)
        degree_matrix = torch.diag(torch.sum(self.adj_matrix, dim=1))
        
        # Compute D^(-1/2) (invert the square root of degree matrix)
        degree_matrix_inv_sqrt = torch.pow(degree_matrix, -0.5)
        degree_matrix_inv_sqrt[torch.isinf(degree_matrix_inv_sqrt)] = 0  # Set inf to 0 (for isolated nodes)
        
        # Return normalized adjacency matrix: D^(-1/2) * A * D^(-1/2)
        adj_matrix_normalized = degree_matrix_inv_sqrt @ self.adj_matrix @ degree_matrix_inv_sqrt
        
        return adj_matrix_normalized