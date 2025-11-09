"""
Edge Decoder for Relational GraphVAE.

Decodes edge existence, types, and continuous features from latent node codes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeDecoder(nn.Module):
    """
    Decoder for edges (adjacency, edge types, and continuous edge features).

    For each pair of nodes (i, j), predicts:
    - Edge existence probability: P(a_ij = 1)
    - Edge type distribution: P(e_type_ij | a_ij = 1)
    - Edge continuous features: μ_e, σ_e (if a_ij = 1)

    Args:
        latent_dim: dimension of node latent codes
        hidden_dim: hidden layer dimension
        edge_type_count: number of edge types
        edge_cont_dim: dimension of continuous edge features
        use_batch_norm: whether to use batch normalization
        dropout: dropout rate
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        edge_type_count: int,
        edge_cont_dim: int,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super(EdgeDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.edge_type_count = edge_type_count
        self.edge_cont_dim = edge_cont_dim

        # Input dimension: [z_i, z_j, |z_i - z_j|, z_i * z_j] for undirected graphs
        # This gives us 4 * latent_dim features per edge
        edge_mlp_input_dim = latent_dim * 4

        # Shared MLP for edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.Dropout(dropout),
        )

        # Head for edge existence (binary classification)
        self.exist_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Head for edge type (multi-class classification, only when edge exists)
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, edge_type_count),
        )

        # Head for continuous edge features (Gaussian parameters)
        if edge_cont_dim > 0:
            self.cont_head_mu = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, edge_cont_dim),
            )

            self.cont_head_logvar = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, edge_cont_dim),
            )
        else:
            self.cont_head_mu = None
            self.cont_head_logvar = None

    def forward(self, z, node_pairs=None):
        """
        Decode edges from latent node codes.

        Args:
            z: [N, latent_dim] latent codes for N nodes
            node_pairs: Optional [P, 2] tensor of node pairs to decode.
                       If None, decodes all pairs (upper triangle for undirected)

        Returns:
            exist_logits: [P] logits for edge existence
            type_logits: [P, edge_type_count] logits for edge types
            cont_mu: [P, edge_cont_dim] mean of continuous edge features
            cont_logvar: [P, edge_cont_dim] log-variance of continuous edge features
        """
        N = z.shape[0]

        if node_pairs is None:
            # Generate all pairs (upper triangle for undirected graphs)
            i_indices = []
            j_indices = []
            for i in range(N):
                for j in range(i + 1, N):
                    i_indices.append(i)
                    j_indices.append(j)

            if not i_indices:
                # Handle empty graph
                device = z.device
                P = 0
                exist_logits = torch.zeros((0,), device=device)
                type_logits = torch.zeros((0, self.edge_type_count), device=device)
                if self.cont_head_mu is not None:
                    cont_mu = torch.zeros((0, self.edge_cont_dim), device=device)
                    cont_logvar = torch.zeros((0, self.edge_cont_dim), device=device)
                else:
                    cont_mu = None
                    cont_logvar = None
                return exist_logits, type_logits, cont_mu, cont_logvar

            i_indices = torch.tensor(i_indices, device=z.device, dtype=torch.long)
            j_indices = torch.tensor(j_indices, device=z.device, dtype=torch.long)
        else:
            # Use provided pairs
            i_indices = node_pairs[:, 0]
            j_indices = node_pairs[:, 1]

        P = i_indices.shape[0]

        # Construct symmetric edge features
        z_i = z[i_indices]  # [P, latent_dim]
        z_j = z[j_indices]  # [P, latent_dim]

        # Concatenate: [z_i, z_j, |z_i - z_j|, z_i * z_j]
        edge_features = torch.cat(
            [
                z_i,  # [P, latent_dim]
                z_j,  # [P, latent_dim]
                torch.abs(z_i - z_j),  # [P, latent_dim]
                z_i * z_j,  # [P, latent_dim]
            ],
            dim=-1,
        )  # [P, 4 * latent_dim]

        # Pass through shared MLP
        edge_embeddings = self.edge_mlp(edge_features)  # [P, hidden_dim]

        # Decode edge existence
        exist_logits = self.exist_head(edge_embeddings).squeeze(-1)  # [P]

        # Decode edge types
        type_logits = self.type_head(edge_embeddings)  # [P, edge_type_count]

        # Decode continuous features (if applicable)
        if self.cont_head_mu is not None:
            cont_mu = self.cont_head_mu(edge_embeddings)  # [P, edge_cont_dim]
            cont_logvar = self.cont_head_logvar(edge_embeddings)  # [P, edge_cont_dim]

            # Clamp log-variance for stability
            cont_logvar = torch.clamp(cont_logvar, min=-10.0, max=10.0)
        else:
            cont_mu = None
            cont_logvar = None

        return exist_logits, type_logits, cont_mu, cont_logvar

    def get_edge_probs(self, z, node_pairs=None):
        """
        Get edge existence probabilities.

        Args:
            z: [N, latent_dim] latent codes
            node_pairs: Optional [P, 2] node pairs

        Returns:
            exist_probs: [P] edge existence probabilities in [0, 1]
        """
        exist_logits, _, _, _ = self.forward(z, node_pairs=node_pairs)
        exist_probs = torch.sigmoid(exist_logits)
        return exist_probs

    def sample_edges(self, z, threshold=0.5, sample=False, temperature=1.0):
        """
        Sample edges from the decoder.

        Args:
            z: [N, latent_dim] latent codes
            threshold: existence threshold for hard threshold sampling
            sample: if True, sample from Bernoulli; if False, use threshold
            temperature: temperature for scaling logits (only if sample=True)

        Returns:
            edge_index: [2, E] edge indices of sampled edges
            edge_attr_cat: [E] sampled edge types
            edge_attr_cont: [E, edge_cont_dim] sampled continuous edge features
        """
        N = z.shape[0]
        device = z.device

        exist_logits, type_logits, cont_mu, cont_logvar = self.forward(z)

        if exist_logits.shape[0] == 0:
            # Empty graph
            edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
            edge_attr_cat = torch.zeros((0,), device=device, dtype=torch.long)
            if cont_mu is not None:
                edge_attr_cont = torch.zeros((0, cont_mu.shape[-1]), device=device)
            else:
                edge_attr_cont = torch.zeros((0, 1), device=device)
            return edge_index, edge_attr_cat, edge_attr_cont

        # Sample/threshold edge existence
        if sample:
            exist_probs = torch.sigmoid(exist_logits / temperature)
            edge_exists = torch.bernoulli(exist_probs).bool()
        else:
            exist_probs = torch.sigmoid(exist_logits)
            edge_exists = exist_probs > threshold

        # Get indices of existing edges
        # Reconstruct pair indices from upper triangle ordering
        pair_idx = 0
        existing_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if edge_exists[pair_idx]:
                    existing_pairs.append((i, j))
                pair_idx += 1

        if not existing_pairs:
            # No edges sampled
            edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
            edge_attr_cat = torch.zeros((0,), device=device, dtype=torch.long)
            if cont_mu is not None:
                edge_attr_cont = torch.zeros((0, cont_mu.shape[-1]), device=device)
            else:
                edge_attr_cont = torch.zeros((0, 1), device=device)
            return edge_index, edge_attr_cat, edge_attr_cont

        # Build edge index with both directions (undirected graph)
        existing_pairs = torch.tensor(existing_pairs, device=device, dtype=torch.long)
        edge_index_fwd = existing_pairs.t()  # [2, E/2]
        edge_index_bwd = torch.stack(
            [existing_pairs[:, 1], existing_pairs[:, 0]]
        )  # [2, E/2]
        edge_index = torch.cat([edge_index_fwd, edge_index_bwd], dim=1)  # [2, E]

        # Sample edge types
        type_probs = F.softmax(type_logits[edge_exists] / temperature, dim=-1)
        edge_attr_cat_fwd = torch.multinomial(type_probs, num_samples=1).squeeze(-1)
        edge_attr_cat = torch.cat(
            [edge_attr_cat_fwd, edge_attr_cat_fwd]
        )  # Repeat for both directions

        # Sample continuous edge features
        if cont_mu is not None:
            cont_mu_sampled = cont_mu[edge_exists]
            cont_std = torch.exp(0.5 * cont_logvar[edge_exists])
            edge_cont_fwd = cont_mu_sampled + cont_std * torch.randn_like(
                cont_mu_sampled
            )
            edge_attr_cont = torch.cat(
                [edge_cont_fwd, edge_cont_fwd], dim=0
            )  # Repeat for both directions
        else:
            edge_attr_cont = torch.zeros((edge_index.shape[1], 1), device=device)

        return edge_index, edge_attr_cat, edge_attr_cont
