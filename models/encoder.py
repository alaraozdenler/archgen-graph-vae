import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class RelationalEncoder(nn.Module):
    """
    Graph encoder for Relational GraphVAE.

    Performs message passing to aggregate node and edge information,
    then outputs per-node latent distributions (mu, logvar).

    Args:
        in_node_cont_dim: dimension of continuous node features
        in_node_cat_dim: number of categorical node feature classes
        in_edge_cat_dim: number of edge type classes
        in_edge_cont_dim: dimension of continuous edge features
        hidden_dim: hidden layer dimension
        latent_dim: latent variable dimension
        num_layers: number of message passing layers
        use_batch_norm: whether to use batch normalization
        dropout: dropout rate
    """

    def __init__(
        self,
        in_node_cont_dim: int,
        in_node_cat_dim: int,
        in_edge_cat_dim: int,
        in_edge_cont_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super(RelationalEncoder, self).__init__()

        self.in_node_cont_dim = in_node_cont_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Node feature embedding layers
        self.node_cont_embed = (
            nn.Linear(in_node_cont_dim, hidden_dim) if in_node_cont_dim > 0 else None
        )
        self.node_cat_embed = (
            nn.Embedding(in_node_cat_dim, hidden_dim) if in_node_cat_dim > 0 else None
        )

        # Edge feature embedding layers
        self.edge_cont_embed = (
            nn.Linear(in_edge_cont_dim, hidden_dim) if in_edge_cont_dim > 0 else None
        )
        self.edge_cat_embed = (
            nn.Embedding(in_edge_cat_dim, hidden_dim) if in_edge_cat_dim > 0 else None
        )

        # Message passing layers (GCN-style)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Batch normalization
        if use_batch_norm:
            self.norms = nn.ModuleList(
                [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
            )
        else:
            self.norms = None

        # Node MLP for edge context integration
        self.edge_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output heads for latent distribution (per-node)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_cont, x_cat, edge_index, edge_attr_cat, edge_attr_cont):
        """
        Forward pass of encoder.

        Args:
            x_cont: [N, in_node_cont_dim] continuous node features
            x_cat: [N] categorical node features
            edge_index: [2, E] edge connectivity
            edge_attr_cat: [E] categorical edge attributes
            edge_attr_cont: [E, in_edge_cont_dim] continuous edge attributes

        Returns:
            mu: [N, latent_dim] mean of posterior
            logvar: [N, latent_dim] log-variance of posterior
        """
        N = x_cont.shape[0] if x_cont is not None else x_cat.shape[0]

        # Embed and concatenate node features
        node_embeds = []
        if self.node_cont_embed is not None and x_cont is not None:
            node_embeds.append(self.node_cont_embed(x_cont))
        if self.node_cat_embed is not None and x_cat is not None:
            node_embeds.append(self.node_cat_embed(x_cat))

        if not node_embeds:
            raise ValueError("No node features provided")

        # Combine node embeddings
        h = torch.cat(node_embeds, dim=-1) if len(node_embeds) > 1 else node_embeds[0]

        # Project to hidden_dim if concatenation changed dimension
        if h.shape[-1] != self.hidden_dim:
            h = nn.Linear(h.shape[-1], self.hidden_dim, device=h.device, dtype=h.dtype)(
                h
            )

        # Embed edge features
        edge_embeds = []
        if (
            self.edge_cont_embed is not None
            and edge_attr_cont is not None
            and edge_attr_cont.shape[0] > 0
        ):
            edge_embeds.append(self.edge_cont_embed(edge_attr_cont))
        if (
            self.edge_cat_embed is not None
            and edge_attr_cat is not None
            and edge_attr_cat.shape[0] > 0
        ):
            edge_embeds.append(self.edge_cat_embed(edge_attr_cat))

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
        for layer_idx in range(self.num_layers):
            # Graph convolution
            h_gnn = self.convs[layer_idx](h, edge_index)

            # Incorporate edge information if available
            if edge_h is not None and edge_index.shape[1] > 0:
                # Aggregate edge features to nodes (mean pooling)
                edge_pooled = torch.zeros_like(h)
                src, dst = edge_index

                # Pool edges to destination nodes
                for i in range(N):
                    edge_mask = dst == i
                    if edge_mask.sum() > 0:
                        edge_pooled[i] = edge_h[edge_mask].mean(dim=0)

                # Combine GNN output with edge features
                h_combined = torch.cat([h_gnn, edge_pooled], dim=-1)
                h = self.edge_mlps[layer_idx](h_combined)
            else:
                h = h_gnn

            # Normalization
            if self.norms is not None:
                h = self.norms[layer_idx](h)

            h = F.relu(h)
            h = self.dropout(h)

        # Output latent parameters (per-node)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        return mu, logvar
