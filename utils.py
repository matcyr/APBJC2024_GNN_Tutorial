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
