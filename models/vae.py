"""
Top-level Relational GraphVAE model.

Integrates encoder, edge decoder, and node decoder into a unified VAE framework.
"""

import torch
import torch.nn as nn
from models.encoder import RelationalEncoder
from models.edge_decoder import EdgeDecoder
from models.node_decoder import RelationalNodeDecoder


class RelationalGraphVAE(nn.Module):
    """
    Relational Graph Variational Autoencoder.

    Combines:
    - Encoder: infers latent per-node codes from graph structure and features
    - Edge decoder: reconstructs adjacency and edge features
    - Node decoder: conditionally reconstructs node features given decoded graph

    Args:
        in_node_cont_dim: dimension of continuous input node features
        in_node_cat_dim: number of categorical input node feature classes
        in_edge_cat_dim: number of edge type classes
        in_edge_cont_dim: dimension of continuous input edge features
        out_node_cont_dim: dimension of output continuous node features
        out_node_cat_dim: number of output categorical node feature classes
        hidden_dim: hidden layer dimension for all modules
        latent_dim: dimension of per-node latent codes
        encoder_num_layers: number of layers in encoder
        node_decoder_num_layers: number of message passing layers in node decoder
        beta: weight for KL divergence term (default: 1.0)
        use_batch_norm: whether to use batch normalization
        dropout: dropout rate
    """

    def __init__(
        self,
        in_node_cont_dim: int,
        in_node_cat_dim: int,
        in_edge_cat_dim: int,
        in_edge_cont_dim: int,
        out_node_cont_dim: int,
        out_node_cat_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        encoder_num_layers: int = 2,
        node_decoder_num_layers: int = 2,
        beta: float = 1.0,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super(RelationalGraphVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder = RelationalEncoder(
            in_node_cont_dim=in_node_cont_dim,
            in_node_cat_dim=in_node_cat_dim,
            in_edge_cat_dim=in_edge_cat_dim,
            in_edge_cont_dim=in_edge_cont_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=encoder_num_layers,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        # Edge decoder
        self.edge_decoder = EdgeDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            edge_type_count=in_edge_cat_dim,
            edge_cont_dim=in_edge_cont_dim,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        # Node decoder
        self.node_decoder = RelationalNodeDecoder(
            latent_dim=latent_dim,
            edge_type_count=in_edge_cat_dim,
            edge_cont_dim=in_edge_cont_dim,
            node_cont_out_dim=out_node_cont_dim,
            node_cat_out_dim=out_node_cat_dim,
            hidden_dim=hidden_dim,
            num_mp_layers=node_decoder_num_layers,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

    def encode(self, x_cont, x_cat, edge_index, edge_attr_cat, edge_attr_cont):
        """
        Encode graph to latent distribution.

        Returns:
            mu: [N, latent_dim] mean
            logvar: [N, latent_dim] log-variance
        """
        mu, logvar = self.encoder(
            x_cont, x_cat, edge_index, edge_attr_cat, edge_attr_cont
        )
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Sample from latent distribution.

        Args:
            mu: [N, latent_dim] mean
            logvar: [N, latent_dim] log-variance

        Returns:
            z: [N, latent_dim] latent samples
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(
        self,
        z,
        edge_index_true=None,
        edge_attr_cat_true=None,
        edge_attr_cont_true=None,
        teacher_forcing=True,
    ):
        """
        Decode latent codes to graph.

        Args:
            z: [N, latent_dim] latent codes
            edge_index_true: [2, E] true edge indices (for teacher forcing)
            edge_attr_cat_true: [E] true edge types (for teacher forcing)
            edge_attr_cont_true: [E, F_e_cont] true edge features (for teacher forcing)
            teacher_forcing: if True, use true edges for node decoder; if False, sample edges

        Returns:
            Tuple of:
            - exist_logits: [P] edge existence logits
            - type_logits: [P, n_types] edge type logits
            - cont_mu_e: [P, F_e_cont] edge continuous feature means
            - cont_logvar_e: [P, F_e_cont] edge continuous feature log-variances
            - cont_mu_x: [N, F_x_cont] node continuous feature means
            - cont_logvar_x: [N, F_x_cont] node continuous feature log-variances
            - cat_logits_x: [N, n_categories] node categorical feature logits
        """
        # Edge decoding
        exist_logits, type_logits, cont_mu_e, cont_logvar_e = self.edge_decoder(z)

        # Node decoding: use teacher forcing during training
        if teacher_forcing and edge_index_true is not None:
            # Use true edges
            cont_mu_x, cont_logvar_x, cat_logits_x = self.node_decoder(
                z, edge_index_true, edge_attr_cat_true, edge_attr_cont_true
            )
        else:
            # Sample edges first
            edge_index_sampled, edge_attr_cat_sampled, edge_attr_cont_sampled = (
                self.edge_decoder.sample_edges(z)
            )

            # Then decode nodes on sampled graph
            cont_mu_x, cont_logvar_x, cat_logits_x = self.node_decoder(
                z, edge_index_sampled, edge_attr_cat_sampled, edge_attr_cont_sampled
            )

        return (
            exist_logits,
            type_logits,
            cont_mu_e,
            cont_logvar_e,
            cont_mu_x,
            cont_logvar_x,
            cat_logits_x,
        )

    def forward(self, x_cont, x_cat, edge_index, edge_attr_cat, edge_attr_cont):
        """
        Full forward pass (training mode with teacher forcing).

        Returns:
            outputs: dict containing all model outputs
            mu: [N, latent_dim]
            logvar: [N, latent_dim]
        """
        # Encode
        mu, logvar = self.encode(
            x_cont, x_cat, edge_index, edge_attr_cat, edge_attr_cont
        )

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode (with teacher forcing)
        (
            exist_logits,
            type_logits,
            cont_mu_e,
            cont_logvar_e,
            cont_mu_x,
            cont_logvar_x,
            cat_logits_x,
        ) = self.decode(
            z,
            edge_index_true=edge_index,
            edge_attr_cat_true=edge_attr_cat,
            edge_attr_cont_true=edge_attr_cont,
            teacher_forcing=True,
        )

        outputs = {
            "exist_logits": exist_logits,
            "type_logits": type_logits,
            "cont_mu_e": cont_mu_e,
            "cont_logvar_e": cont_logvar_e,
            "cont_mu_x": cont_mu_x,
            "cont_logvar_x": cont_logvar_x,
            "cat_logits_x": cat_logits_x,
        }

        return outputs, mu, logvar

    def sample(self, num_nodes, device=None):
        """
        Generate a new graph by sampling from prior.

        Args:
            num_nodes: number of nodes to generate
            device: device to use

        Returns:
            data: dict with generated graph
        """
        if device is None:
            device = next(self.parameters()).device

        # Sample from prior
        z = torch.randn(num_nodes, self.latent_dim, device=device)

        # Decode (without teacher forcing, sample edges)
        (
            exist_logits,
            type_logits,
            cont_mu_e,
            cont_logvar_e,
            cont_mu_x,
            cont_logvar_x,
            cat_logits_x,
        ) = self.decode(z, teacher_forcing=False)

        # Sample from reconstructed distributions
        # Sample node continuous features
        if cont_mu_x is not None:
            node_cont = cont_mu_x + torch.exp(0.5 * cont_logvar_x) * torch.randn_like(
                cont_mu_x
            )
        else:
            node_cont = None

        # Sample node categories
        if cat_logits_x is not None:
            node_cat = torch.argmax(cat_logits_x, dim=-1)
        else:
            node_cat = None

        # Edge structure is already sampled in edge_decoder.sample_edges()
        # (called via decode with teacher_forcing=False)

        data = {
            "z": z,
            "node_cont": node_cont,
            "node_cat": node_cat,
            "exist_logits": exist_logits,
            "type_logits": type_logits,
        }

        return data
