"""
Node Decoder for Relational GraphVAE.

Conditionally reconstructs node features given the decoded graph structure and latent codes.
Uses message passing on the decoded adjacency and edge features.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool
import torch.nn.functional as F


class RelationalNodeDecoder(MessagePassing):
    """
    Conditional node feature decoder using message passing.

    Given:
    - Latent node codes z_i
    - Decoded edge structure (from edge decoder)
    - Decoded edge types and features

    Performs message passing to integrate local neighborhood information,
    then reconstructs node continuous and categorical features.

    Args:
        latent_dim: dimension of node latent codes
        edge_type_count: number of edge types
        edge_cont_dim: dimension of continuous edge features
        node_cont_out_dim: dimension of output continuous node features
        node_cat_out_dim: number of output node categories
        hidden_dim: hidden layer dimension
        num_mp_layers: number of message passing layers
        use_batch_norm: whether to use batch normalization
        dropout: dropout rate
    """

    def __init__(
        self,
        latent_dim: int,
        edge_type_count: int,
        edge_cont_dim: int,
        node_cont_out_dim: int,
        node_cat_out_dim: int,
        hidden_dim: int,
        num_mp_layers: int = 2,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super(RelationalNodeDecoder, self).__init__(aggr="mean")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.edge_type_count = edge_type_count
        self.edge_cont_dim = edge_cont_dim
        self.node_cont_out_dim = node_cont_out_dim
        self.node_cat_out_dim = node_cat_out_dim
        self.num_mp_layers = num_mp_layers

        # Edge feature embeddings
        self.edge_type_embed = nn.Embedding(edge_type_count, hidden_dim)
        self.edge_cont_embed = (
            nn.Linear(edge_cont_dim, hidden_dim) if edge_cont_dim > 0 else None
        )

        # Initial node state from latent code
        self.node_proj = nn.Linear(latent_dim, hidden_dim)

        # Message passing layers
        self.message_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),  # node + neighbor + edge
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_mp_layers)
            ]
        )

        # Node update MLPs
        self.node_update_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),  # node + aggregated messages
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_mp_layers)
            ]
        )

        # Output heads
        # Continuous node features
        if node_cont_out_dim > 0:
            self.cont_mu_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, node_cont_out_dim),
            )

            self.cont_logvar_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, node_cont_out_dim),
            )
        else:
            self.cont_mu_head = None
            self.cont_logvar_head = None

        # Categorical node features
        if node_cat_out_dim > 0:
            self.cat_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, node_cat_out_dim),
            )
        else:
            self.cat_head = None

    def forward(self, z, edge_index, edge_attr_cat, edge_attr_cont):
        """
        Forward pass of node decoder.

        Args:
            z: [N, latent_dim] latent node codes
            edge_index: [2, E] edge connectivity (from decoded graph)
            edge_attr_cat: [E] categorical edge attributes
            edge_attr_cont: [E, edge_cont_dim] continuous edge attributes

        Returns:
            cont_mu: [N, node_cont_out_dim] or None
            cont_logvar: [N, node_cont_out_dim] or None
            cat_logits: [N, node_cat_out_dim] or None
        """
        N = z.shape[0]

        # Initial node state from latent code
        h = self.node_proj(z)  # [N, hidden_dim]

        # Embed edge features
        edge_embeds = []
        if edge_attr_cat is not None and edge_attr_cat.shape[0] > 0:
            # Squeeze if needed - edge_attr_cat might be [E, 1] or [E]
            edge_cat = (
                edge_attr_cat.squeeze(-1) if edge_attr_cat.dim() > 1 else edge_attr_cat
            )
            edge_embeds.append(self.edge_type_embed(edge_cat))
        if (
            self.edge_cont_embed is not None
            and edge_attr_cont is not None
            and edge_attr_cont.shape[0] > 0
        ):
            edge_embeds.append(self.edge_cont_embed(edge_attr_cont))

        if edge_embeds:
            edge_h = (
                torch.cat(edge_embeds, dim=-1)
                if len(edge_embeds) > 1
                else edge_embeds[0]
            )
            # Project to hidden_dim if needed
            if edge_h.shape[-1] != self.hidden_dim:
                edge_h = nn.Linear(
                    edge_h.shape[-1],
                    self.hidden_dim,
                    device=edge_h.device,
                    dtype=edge_h.dtype,
                )(edge_h)
        else:
            edge_h = None

        # Message passing layers
        for layer_idx in range(self.num_mp_layers):
            # Aggregate messages from neighbors
            if edge_index.shape[1] > 0 and edge_h is not None:
                m = self.propagate(
                    edge_index, x=h, edge_attr=edge_h, mlp=self.message_mlps[layer_idx]
                )
            else:
                m = torch.zeros_like(h)

            # Update node state with aggregated messages
            h_new = self.node_update_mlps[layer_idx](torch.cat([h, m], dim=-1))
            h = F.relu(h_new)

        # Output heads
        cont_mu = None
        cont_logvar = None
        cat_logits = None

        if self.cont_mu_head is not None:
            cont_mu = self.cont_mu_head(h)
            cont_logvar = self.cont_logvar_head(h)
            # Clamp log-variance for stability
            cont_logvar = torch.clamp(cont_logvar, min=-10.0, max=10.0)

        if self.cat_head is not None:
            cat_logits = self.cat_head(h)

        return cont_mu, cont_logvar, cat_logits

    def message(self, x_i, x_j, edge_attr, mlp):
        """
        Compute messages from source to target nodes.

        Args:
            x_i: [E, hidden_dim] target node features
            x_j: [E, hidden_dim] source node features
            edge_attr: [E, hidden_dim] edge features
            mlp: MLP for message computation

        Returns:
            messages: [E, hidden_dim]
        """
        # Concatenate source, target, and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        messages = mlp(msg_input)
        return messages

    def aggregate(self, inputs, index, dim_size=None):
        """
        Aggregate messages at target nodes using scatter-add and count.
        Overrides MessagePassing to use mean aggregation.

        Args:
            inputs: [E, hidden_dim] messages
            index: [E] target node indices
            dim_size: number of nodes

        Returns:
            aggregated: [N, hidden_dim]
        """
        if dim_size is None:
            dim_size = index.max().item() + 1

        # Initialize output
        aggregated = torch.zeros(
            dim_size, inputs.shape[-1], device=inputs.device, dtype=inputs.dtype
        )

        # Scatter-add messages
        aggregated.scatter_add_(
            0, index.unsqueeze(-1).expand(-1, inputs.shape[-1]), inputs
        )

        # Count messages per node for averaging
        counts = torch.zeros(dim_size, 1, device=inputs.device, dtype=inputs.dtype)
        counts.scatter_add_(
            0,
            index.unsqueeze(-1),
            torch.ones_like(index.unsqueeze(-1), dtype=inputs.dtype),
        )

        # Avoid division by zero
        counts = torch.clamp(counts, min=1.0)

        # Average
        aggregated = aggregated / counts

        return aggregated
