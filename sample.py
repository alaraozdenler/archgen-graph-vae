#!/usr/bin/env python3
"""
Graph sampling and generation from Relational GraphVAE.

Implements generation API:
- Sample new graphs from prior (random latent codes)
- Reconstruct graphs from data
- Conditional generation
- Visualization utilities
"""

import os
import sys
import argparse
from typing import Dict, Tuple, Optional, List
from pathlib import Path

import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "archgen", "src"))

from config import (
    DEVICE,
    LATENT_DIM,
    HIDDEN_DIM,
    ENCODER_NUM_LAYERS,
    NODE_DECODER_NUM_MP_LAYERS,
    DROPOUT_RATE,
    USE_BATCH_NORM,
    USE_CUDA,
    OUT_NODE_CONT_DIM,
    OUT_NODE_CAT_DIM,
    DATA_PATH,
)
from data_utils import load_graphs, create_dataset
from models import RelationalGraphVAE


class GraphSampler:
    """Sampler for generating graphs from RelationalGraphVAE."""

    def __init__(
        self,
        model: RelationalGraphVAE,
        device: str = "cpu",
        edge_threshold: float = 0.5,
        node_type_map: Optional[Dict[int, str]] = None,
        edge_type_map: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize sampler.

        Args:
            model: Trained RelationalGraphVAE
            device: Device to sample on
            edge_threshold: Threshold for edge existence probability
            node_type_map: Mapping from type indices to names
            edge_type_map: Mapping from type indices to names
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.edge_threshold = edge_threshold

        self.node_type_map = node_type_map or {
            0: "Column",
            1: "Door",
            2: "Wall",
            3: "Window",
        }
        self.edge_type_map = edge_type_map or {
            0: "EdgeType.ADJACENCY",
            1: "EdgeType.WALL_INT",
            2: "EdgeType.WINDOW_INT",
        }

    @torch.no_grad()
    def sample(
        self,
        num_nodes: int,
        latent_codes: Optional[torch.Tensor] = None,
        num_samples: int = 1,
    ) -> List[Data]:
        """
        Generate graphs by sampling from prior or given latent codes.

        Args:
            num_nodes: Number of nodes in generated graph
            latent_codes: Pre-computed latent codes [N, latent_dim], or None to sample from N(0,I)
            num_samples: Number of graphs to generate

        Returns:
            List of torch_geometric.Data objects
        """
        generated_graphs = []

        for sample_idx in range(num_samples):
            # Sample or use provided latent codes
            if latent_codes is None:
                z = torch.randn(num_nodes, self.model.latent_dim, device=self.device)
            else:
                z = latent_codes.to(self.device)

            # Decode graph
            data = self._decode_graph(z)
            generated_graphs.append(data)

        return generated_graphs

    @torch.no_grad()
    def reconstruct(self, data: Data) -> Data:
        """
        Reconstruct graph: encode then decode.

        Args:
            data: Input torch_geometric.Data object

        Returns:
            Reconstructed torch_geometric.Data object
        """
        # Move to device
        data = data.to(self.device)

        # Encode
        mu, logvar = self.model.encode(
            data.x_cont,
            data.x_cat,
            data.edge_index,
            data.edge_attr_cat,
            data.edge_attr_cont,
        )

        # Use mean (deterministic reconstruction)
        z = mu

        # Decode
        reconstructed = self._decode_graph(z, data.edge_index)

        return reconstructed

    @torch.no_grad()
    def _decode_graph(
        self,
        z: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Data:
        """
        Decode latent codes into graph.

        Args:
            z: Latent codes [N, latent_dim]
            edge_index: Optional true edge_index for teacher forcing

        Returns:
            torch_geometric.Data object
        """
        num_nodes = z.shape[0]

        # Decode edges
        edge_exist_logits, edge_type_logits, edge_cont_mu, edge_cont_logvar = (
            self.model.edge_decoder(z)
        )

        # Sample edge existence
        edge_probs = torch.sigmoid(edge_exist_logits)
        edge_samples = torch.bernoulli(edge_probs)  # [P]

        # Convert to edge indices
        node_pairs = []
        edge_indices = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                pair_idx = i * num_nodes + j - (i + 1) * (i + 2) // 2
                if edge_samples[pair_idx].item() > 0.5:
                    node_pairs.append([i, j])
                    edge_indices.append(pair_idx)

        if len(node_pairs) == 0:
            # No edges - create isolated graph
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            edge_attr_cat = torch.zeros((0, 1), dtype=torch.long, device=self.device)
            edge_attr_cont = torch.zeros(
                (0, 1), dtype=torch.float32, device=self.device
            )
        else:
            # Build edge index (undirected - add both directions)
            node_pairs = np.array(node_pairs)
            edge_index_list = []
            edge_index_list.append(node_pairs)  # i -> j
            edge_index_list.append(node_pairs[:, [1, 0]])  # j -> i
            edge_index = (
                torch.from_numpy(np.concatenate(edge_index_list, axis=0).T)
                .to(torch.long)
                .to(self.device)
            )

            # Sample edge types
            edge_indices = torch.tensor(edge_indices, device=self.device)
            edge_type_logits_subset = edge_type_logits[edge_indices]  # [E, 3]
            edge_type_samples = torch.argmax(edge_type_logits_subset, dim=1)  # [E]

            # Duplicate for undirected edges
            edge_attr_cat = (
                torch.cat([edge_type_samples, edge_type_samples], dim=0)
                .unsqueeze(1)
                .to(torch.long)
            )  # [2E, 1]

            # Sample edge continuous features (length)
            edge_cont_mu_subset = edge_cont_mu[edge_indices]  # [E, 1]
            edge_cont_logvar_subset = edge_cont_logvar[edge_indices]  # [E, 1]
            edge_cont_std = torch.sqrt(torch.exp(edge_cont_logvar_subset))
            edge_cont_samples = torch.normal(
                edge_cont_mu_subset, edge_cont_std
            )  # [E, 1]

            # Duplicate for undirected edges
            edge_attr_cont = torch.cat(
                [edge_cont_samples, edge_cont_samples], dim=0
            ).to(
                torch.float32
            )  # [2E, 1]

        # Decode nodes (using sampled edges)
        node_cont_mu, node_cont_logvar, node_cat_logits = self.model.node_decoder(
            z, edge_index, edge_attr_cat, edge_attr_cont
        )
        # node_cont_mu: [N, 3]
        # node_cont_logvar: [N, 3]
        # node_cat_logits: [N, 4]

        # Sample node continuous features
        node_cont_std = torch.sqrt(torch.exp(node_cont_logvar))
        x_cont = torch.normal(node_cont_mu, node_cont_std)  # [N, 3]

        # Sample node categorical features
        node_cat_samples = torch.argmax(node_cat_logits, dim=1).unsqueeze(1)  # [N, 1]

        # Create Data object
        data = Data(
            x_cont=x_cont.cpu(),
            x_cat=node_cat_samples.cpu().to(torch.long),
            edge_index=edge_index.cpu(),
            edge_attr_cat=edge_attr_cat.cpu(),
            edge_attr_cont=edge_attr_cont.cpu(),
        )

        return data

    def to_networkx(
        self,
        data: Data,
        remove_isolated: bool = False,
    ) -> nx.Graph:
        """
        Convert torch_geometric.Data to NetworkX graph.

        Args:
            data: torch_geometric.Data object
            remove_isolated: Whether to remove isolated nodes

        Returns:
            NetworkX Graph
        """
        G = nx.Graph()

        # Add nodes with attributes
        for i in range(data.x_cont.shape[0]):
            node_type = self.node_type_map.get(data.x_cat[i, 0].item(), "Unknown")
            G.add_node(
                i,
                pos_x=float(data.x_cont[i, 0].item()),
                pos_y=float(data.x_cont[i, 1].item()),
                angle=float(data.x_cont[i, 2].item()),
                arch_type=node_type,
            )

        # Add edges with attributes
        if data.edge_index is not None and data.edge_index.shape[1] > 0:
            edges = data.edge_index.t().numpy()
            for idx, (i, j) in enumerate(edges):
                if i < j:  # Only add once (undirected)
                    edge_type = self.edge_type_map.get(
                        data.edge_attr_cat[idx, 0].item(), "Unknown"
                    )
                    length = float(data.edge_attr_cont[idx, 0].item())
                    G.add_edge(i, j, edge_type=edge_type, length=length)

        # Remove isolated nodes if requested
        if remove_isolated:
            G.remove_nodes_from(list(nx.isolates(G)))

        return G


def main():
    """Main generation script."""
    parser = argparse.ArgumentParser(
        description="Generate graphs from Relational GraphVAE"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_graphs", type=int, default=10, help="Number of graphs to generate"
    )
    parser.add_argument(
        "--num_nodes", type=int, default=50, help="Number of nodes per graph"
    )
    parser.add_argument(
        "--edge_threshold", type=float, default=0.5, help="Threshold for edge existence"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Device to sample on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_graphs",
        help="Directory to save generated graphs",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample", "reconstruct"],
        default="sample",
        help="Generation mode: sample from prior or reconstruct",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help="Path to graph pickle (for reconstruct mode)",
    )

    args = parser.parse_args()

    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Create model
    print(f"Creating model...")
    model = RelationalGraphVAE(
        in_node_cont_dim=3,
        in_node_cat_dim=4,  # 4 node types
        in_edge_cat_dim=3,  # 3 edge types
        in_edge_cont_dim=1,
        out_node_cont_dim=OUT_NODE_CONT_DIM,
        out_node_cat_dim=OUT_NODE_CAT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        encoder_num_layers=ENCODER_NUM_LAYERS,
        node_decoder_num_layers=NODE_DECODER_NUM_MP_LAYERS,
        use_batch_norm=USE_BATCH_NORM,
        dropout=DROPOUT_RATE,
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Create sampler
    sampler = GraphSampler(model, device=device, edge_threshold=args.edge_threshold)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating graphs in {args.mode} mode...")

    if args.mode == "sample":
        # Generate new graphs
        graphs = sampler.sample(num_nodes=args.num_nodes, num_samples=args.num_graphs)

        for i, data in enumerate(graphs):
            G = sampler.to_networkx(data, remove_isolated=True)
            save_path = output_dir / f"generated_{i}.graphml"
            nx.write_graphml(G, save_path)
            print(
                f"  Generated graph {i}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
            )

    elif args.mode == "reconstruct":
        # Reconstruct from data
        print(f"Loading data from {args.data_path}...")
        graphs = load_graphs(args.data_path)
        train_data, _, _, _, _ = create_dataset(graphs)

        for i in range(min(args.num_graphs, len(train_data))):
            data = train_data[i]

            # Reconstruct
            reconstructed = sampler.reconstruct(data)

            # Convert to NetworkX
            G_orig = sampler.to_networkx(data, remove_isolated=False)
            G_recon = sampler.to_networkx(reconstructed, remove_isolated=False)

            # Save
            save_orig = output_dir / f"original_{i}.graphml"
            save_recon = output_dir / f"reconstructed_{i}.graphml"
            nx.write_graphml(G_orig, save_orig)
            nx.write_graphml(G_recon, save_recon)

            print(f"  Graph {i}:")
            print(
                f"    Original:      {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges"
            )
            print(
                f"    Reconstructed: {G_recon.number_of_nodes()} nodes, {G_recon.number_of_edges()} edges"
            )

    print(f"\nGraphs saved to {output_dir}")


if __name__ == "__main__":
    main()
