"""
Data utilities for Relational GraphVAE.

Handles conversion from NetworkX graphs to PyTorch Geometric format,
managing mixed-type node and edge attributes (continuous + categorical).
"""

import sys
import os
import pickle
import numpy as np
import networkx as nx
from enum import Enum
from typing import Optional, Tuple, List, Dict
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add archgen module to path for loading pickled objects
_archgen_path = os.path.join(os.path.dirname(__file__), "..", "archgen", "src")
if os.path.exists(_archgen_path):
    sys.path.insert(0, _archgen_path)


class ArchitectureType(Enum):
    """Node types in architecture graphs."""

    WALL = "Wall"
    WINDOW = "Window"
    DOOR = "Door"
    COLUMN = "Column"
    OTHER = "Other"


class EdgeType(Enum):
    """Edge types in architecture graphs."""

    WALL_INT = "wall"
    ADJACENCY = "adjacency"


def load_graphs(path: str) -> List[nx.Graph]:
    """Load graphs from pickle file."""
    with open(path, "rb") as f:
        graphs = pickle.load(f)
    return graphs


def extract_edge_type_mapping(graphs: List[nx.Graph]) -> Dict[str, int]:
    """Extract unique edge types and create integer mapping."""
    edge_types = set()
    for G in graphs:
        for _, _, attrs in G.edges(data=True):
            if "type" in attrs:
                edge_type = attrs["type"]
                # Handle both string and EdgeType enum
                if isinstance(edge_type, EdgeType):
                    edge_types.add(edge_type.value)
                else:
                    edge_types.add(str(edge_type))

    # Create mapping
    edge_type_map = {et: i for i, et in enumerate(sorted(edge_types))}
    return edge_type_map


def extract_node_type_mapping(graphs: List[nx.Graph]) -> Dict[str, int]:
    """Extract unique node types (arch_type) and create integer mapping."""
    node_types = set()
    for G in graphs:
        for _, attrs in G.nodes(data=True):
            if "arch_type" in attrs:
                node_type = attrs["arch_type"]
                node_types.add(str(node_type))

    # Create mapping
    node_type_map = {nt: i for i, nt in enumerate(sorted(node_types))}
    return node_type_map


def networkx_to_pyg(
    G: nx.Graph,
    node_type_map: Dict[str, int],
    edge_type_map: Dict[str, int],
    node_scaler=None,
    edge_scaler=None,
) -> Data:
    """
    Convert a NetworkX graph to PyTorch Geometric Data object.

    Node features:
    - x_cont: [N, 2] containing (pos_x, pos_y) and angle
    - x_cat: [N] containing arch_type

    Edge features:
    - edge_index: [2, E] edge connectivity
    - edge_attr_cat: [E] edge type
    - edge_attr_cont: [E, 1] edge length
    """

    # Map node indices (NetworkX uses tuples as node ids, convert to sequential)
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    N = len(node_list)

    # Extract node features
    x_cont_list = []
    x_cat_list = []

    for node in node_list:
        attrs = G.nodes[node]

        # Continuous features: pos (2D) + angle (1D)
        pos = attrs.get("pos", (0.0, 0.0))
        angle = attrs.get("angle", 0.0)
        x_cont = [pos[0], pos[1], angle]
        x_cont_list.append(x_cont)

        # Categorical feature: arch_type
        arch_type = attrs.get("arch_type", "Other")
        arch_type_idx = node_type_map.get(arch_type, 0)
        x_cat_list.append(arch_type_idx)

    x_cont = torch.tensor(x_cont_list, dtype=torch.float32)
    x_cat = torch.tensor(x_cat_list, dtype=torch.long)

    # Normalize continuous features if scaler provided
    if node_scaler is not None:
        x_cont_np = node_scaler.transform(x_cont.numpy())
        x_cont = torch.tensor(x_cont_np, dtype=torch.float32)

    # Extract edge features
    edge_index_list = []
    edge_attr_cat_list = []
    edge_attr_cont_list = []

    for u, v, attrs in G.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]

        # Add edge in both directions (undirected)
        edge_index_list.append([u_idx, v_idx])
        edge_index_list.append([v_idx, u_idx])

        # Edge type
        edge_type = attrs.get("type", "unknown")
        if isinstance(edge_type, EdgeType):
            edge_type = edge_type.value
        else:
            edge_type = str(edge_type)
        edge_type_idx = edge_type_map.get(edge_type, 0)
        edge_attr_cat_list.append(edge_type_idx)
        edge_attr_cat_list.append(edge_type_idx)  # Both directions

        # Edge continuous features: length
        length = attrs.get("length", 0.0)
        edge_attr_cont_list.append([length])
        edge_attr_cont_list.append([length])  # Both directions

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr_cat = torch.tensor(edge_attr_cat_list, dtype=torch.long)
        edge_attr_cont = torch.tensor(edge_attr_cont_list, dtype=torch.float32)
    else:
        # Empty graph
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_cat = torch.zeros((0,), dtype=torch.long)
        edge_attr_cont = torch.zeros((0, 1), dtype=torch.float32)

    # Normalize edge continuous features if scaler provided
    if edge_scaler is not None and edge_attr_cont.shape[0] > 0:
        edge_attr_cont_np = edge_scaler.transform(edge_attr_cont.numpy())
        edge_attr_cont = torch.tensor(edge_attr_cont_np, dtype=torch.float32)

    # Create PyG Data object
    data = Data(
        x_cont=x_cont,
        x_cat=x_cat,
        edge_index=edge_index,
        edge_attr_cat=edge_attr_cat,
        edge_attr_cont=edge_attr_cont,
        num_nodes=N,
    )

    return data


def create_dataset(
    graphs: List[nx.Graph],
    train_split: float = 0.8,
    val_split: float = 0.1,
    normalize: bool = True,
) -> Tuple[List[Data], List[Data], List[Data], Dict, Dict]:
    """
    Convert NetworkX graphs to PyG dataset and split into train/val/test.

    Returns:
        train_data, val_data, test_data, node_type_map, edge_type_map
    """

    # Extract type mappings
    node_type_map = extract_node_type_mapping(graphs)
    edge_type_map = extract_edge_type_mapping(graphs)

    print(f"Node types: {node_type_map}")
    print(f"Edge types: {edge_type_map}")

    # Fit scalers on all data if normalizing
    if normalize:
        all_x_cont = []
        all_edge_cont = []
        for G in graphs:
            for _, attrs in G.nodes(data=True):
                pos = attrs.get("pos", (0.0, 0.0))
                angle = attrs.get("angle", 0.0)
                all_x_cont.append([pos[0], pos[1], angle])

            for _, _, attrs in G.edges(data=True):
                length = attrs.get("length", 0.0)
                all_edge_cont.append([length])

        # Fit scalers
        from sklearn.preprocessing import StandardScaler

        if all_x_cont:
            node_scaler = StandardScaler()
            node_scaler.fit(np.array(all_x_cont))
        else:
            node_scaler = None

        if all_edge_cont:
            edge_scaler = StandardScaler()
            edge_scaler.fit(np.array(all_edge_cont))
        else:
            edge_scaler = None
    else:
        node_scaler = None
        edge_scaler = None

    # Convert all graphs
    pyg_data = []
    for G in graphs:
        data = networkx_to_pyg(
            G, node_type_map, edge_type_map, node_scaler, edge_scaler
        )
        pyg_data.append(data)

    # Split data
    n_total = len(pyg_data)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    train_data = [pyg_data[i] for i in train_indices]
    val_data = [pyg_data[i] for i in val_indices]
    test_data = [pyg_data[i] for i in test_indices]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")

    return train_data, val_data, test_data, node_type_map, edge_type_map


class GraphDataLoader(DataLoader):
    """Custom DataLoader for graph data with mixed attributes."""

    def __init__(self, data_list: List[Data], batch_size: int = 1, **kwargs):
        super().__init__(data_list, batch_size=batch_size, **kwargs)


# Data field conventions
"""
PyTorch Geometric Data fields used:
- x_cont: FloatTensor [N, F_x_cont] — continuous node features (pos_x, pos_y, angle)
- x_cat: LongTensor [N] — categorical node features (arch_type)
- edge_index: LongTensor [2, E] — edge connectivity
- edge_attr_cat: LongTensor [E] — categorical edge types
- edge_attr_cont: FloatTensor [E, F_e_cont] — continuous edge features (length)
- num_nodes: int — number of nodes (automatically set)

Masking conventions:
- If a node has no continuous features, x_cont can be None or zero-filled
- If a node has no categorical features, x_cat can be None or -1 (invalid index)
- If an edge has no continuous features, edge_attr_cont can be None or zero-filled
- Node/edge masks can be added for handling variable-length attributes
"""
