"""
Loss functions for Relational GraphVAE training.

Includes:
- KL divergence
- Edge existence (binary cross-entropy)
- Edge type (categorical cross-entropy, masked for existing edges only)
- Edge continuous features (Gaussian NLL, masked for existing edges only)
- Node continuous features (Gaussian NLL)
- Node categorical features (categorical cross-entropy)
"""

import torch
import torch.nn.functional as F


def kl_divergence(mu, logvar):
    """
    KL divergence between posterior q(z|x) and prior p(z) = N(0, I).

    Args:
        mu: [N, latent_dim] posterior mean
        logvar: [N, latent_dim] posterior log-variance

    Returns:
        kl: [N] KL divergence per node, shape [N]
        Or: scalar if reduced
    """
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl  # [N]


def edge_existence_loss(
    exist_logits, edge_index, num_nodes, weight_pos=None, weight_neg=None
):
    """
    Binary cross-entropy loss for edge existence.

    Args:
        exist_logits: [P] logits for edge pairs
        edge_index: [2, E] true edges
        num_nodes: number of nodes
        weight_pos: weight for positive examples (default: None)
        weight_neg: weight for negative examples (default: None)

    Returns:
        loss: scalar
    """
    # Create target labels (1 for true edges, 0 for sampled non-edges)
    P = exist_logits.shape[0]

    # Reconstruct which pairs are true edges
    # Map edge_index to pair indices
    true_edges_set = set()
    for u, v in edge_index.t():
        u_idx, v_idx = u.item(), v.item()
        # Store as upper triangle (i < j)
        if u_idx > v_idx:
            u_idx, v_idx = v_idx, u_idx
        true_edges_set.add((u_idx, v_idx))

    # Create labels: 1 if edge exists in true_edges_set, 0 otherwise
    labels = torch.zeros(P, device=exist_logits.device, dtype=torch.float)

    pair_idx = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) in true_edges_set:
                labels[pair_idx] = 1.0
            pair_idx += 1

    # Binary cross-entropy
    loss = F.binary_cross_entropy_with_logits(
        exist_logits,
        labels,
        pos_weight=torch.tensor(weight_pos) if weight_pos else None,
        reduction="mean",
    )

    return loss


def edge_type_loss(type_logits, edge_index, edge_attr_cat, num_nodes):
    """
    Categorical cross-entropy loss for edge types (only on existing edges).

    Args:
        type_logits: [P, n_types] logits for all pairs
        edge_index: [2, E] true edges
        edge_attr_cat: [E] true edge types
        num_nodes: number of nodes

    Returns:
        loss: scalar
    """
    # Create target labels for all pairs
    P = type_logits.shape[0]
    n_types = type_logits.shape[-1]

    # Initialize with invalid class index (-1, will be ignored)
    targets = torch.full((P,), -1, device=type_logits.device, dtype=torch.long)

    # Create mapping: edge_index to pair index
    pair_to_edge = {}
    for edge_idx, (u, v) in enumerate(edge_index.t()):
        u_idx, v_idx = u.item(), v.item()
        # Normalize to upper triangle
        if u_idx > v_idx:
            u_idx, v_idx = v_idx, u_idx
        # Map to pair index
        pair_idx = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if i == u_idx and j == v_idx:
                    pair_to_edge[(i, j)] = edge_idx
                pair_idx += 1

    # Assign edge types to pair indices
    pair_idx = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) in pair_to_edge:
                edge_idx = pair_to_edge[(i, j)]
                targets[pair_idx] = edge_attr_cat[edge_idx]
            pair_idx += 1

    # Cross-entropy (ignores targets == -1)
    loss = F.cross_entropy(type_logits, targets, ignore_index=-1, reduction="mean")

    return loss


def edge_cont_loss(cont_mu, cont_logvar, edge_index, edge_attr_cont, num_nodes):
    """
    Gaussian negative log-likelihood for continuous edge features.

    Args:
        cont_mu: [P, F_e] predicted means for all pairs
        cont_logvar: [P, F_e] predicted log-variances for all pairs
        edge_index: [2, E] true edges
        edge_attr_cont: [E, F_e] true edge features
        num_nodes: number of nodes

    Returns:
        loss: scalar
    """
    if cont_mu is None or cont_mu.shape[0] == 0:
        return torch.tensor(
            0.0, device=cont_mu.device if cont_mu is not None else edge_index.device
        )

    if edge_attr_cont is None or edge_attr_cont.shape[0] == 0:
        return torch.tensor(0.0, device=cont_mu.device)

    # Create mapping similar to edge_type_loss
    P = cont_mu.shape[0]
    F_e = cont_mu.shape[-1]

    # Initialize with zeros (will be masked out)
    targets = torch.zeros((P, F_e), device=cont_mu.device, dtype=torch.float)
    mask = torch.zeros(P, device=cont_mu.device, dtype=torch.bool)

    # Create mapping
    pair_to_edge = {}
    for edge_idx, (u, v) in enumerate(edge_index.t()):
        u_idx, v_idx = u.item(), v.item()
        if u_idx > v_idx:
            u_idx, v_idx = v_idx, u_idx
        pair_to_edge[(u_idx, v_idx)] = edge_idx

    # Assign targets
    pair_idx = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) in pair_to_edge:
                edge_idx = pair_to_edge[(i, j)]
                targets[pair_idx] = edge_attr_cont[edge_idx]
                mask[pair_idx] = True
            pair_idx += 1

    # Gaussian NLL: 0.5 * ||x - mu||^2 / sigma^2 + 0.5 * log(2*pi*sigma^2)
    sigma = torch.exp(0.5 * cont_logvar)
    nll = 0.5 * ((targets - cont_mu) / sigma).pow(2) + 0.5 * (
        cont_logvar + torch.log(torch.tensor(2 * 3.14159))
    )

    # Apply mask (only compute loss on true edges)
    loss = nll[mask].mean()

    return loss


def node_cont_loss(cont_mu, cont_logvar, x_cont):
    """
    Gaussian negative log-likelihood for continuous node features.

    Args:
        cont_mu: [N, F_x_cont] predicted means
        cont_logvar: [N, F_x_cont] predicted log-variances
        x_cont: [N, F_x_cont] true continuous node features

    Returns:
        loss: scalar
    """
    if cont_mu is None or cont_mu.shape[0] == 0:
        return torch.tensor(0.0)

    if x_cont is None or x_cont.shape[0] == 0:
        return torch.tensor(0.0)

    sigma = torch.exp(0.5 * cont_logvar)
    nll = 0.5 * ((x_cont - cont_mu) / sigma).pow(2) + 0.5 * (
        cont_logvar + torch.log(torch.tensor(2 * 3.14159, device=cont_logvar.device))
    )

    loss = nll.mean()

    return loss


def node_cat_loss(cat_logits, x_cat):
    """
    Categorical cross-entropy for categorical node features.

    Args:
        cat_logits: [N, n_categories] predicted logits
        x_cat: [N] true categorical labels

    Returns:
        loss: scalar
    """
    if cat_logits is None or cat_logits.shape[0] == 0:
        return torch.tensor(0.0)

    if x_cat is None or x_cat.shape[0] == 0:
        return torch.tensor(0.0)

    loss = F.cross_entropy(cat_logits, x_cat, reduction="mean")

    return loss


def geometric_loss(
    node_positions, edge_index, node_radius, exclude_endpoints=True, reduction="sum"
):
    """
    Geometric loss that penalizes nodes that lie too close to an edge segment.

    For each edge (i, j) we compute the distance from every node k (k != i, j)
    to the segment p_i - p_j. If the distance d < node_radius we add
    (node_radius - d)^2 to the loss (squared hinge). The total loss is
    returned as a scalar.

    Args:
        node_positions: Tensor [N, >=2] node coordinates (x, y, ...). Only the
                        first two dims are used.
        edge_index: Tensor [2, E] edge endpoints in torch_geometric format.
        node_radius: float or scalar tensor, radius threshold.
        exclude_endpoints: if True, nodes that are the endpoints of the edge are
                           excluded from the penalty.
        reduction: "sum" or "mean". "sum" returns summed loss, "mean" returns
                   averaged loss across contributing terms (non-zero contributions).

    Returns:
        loss (torch.Tensor): scalar tensor
    """
    # Basic checks
    if node_positions is None or node_positions.numel() == 0:
        return torch.tensor(0.0)
    if edge_index is None or edge_index.numel() == 0:
        return torch.tensor(0.0, device=node_positions.device)

    device = node_positions.device
    pos = node_positions[:, :2].to(device)
    edge_idx = edge_index.to(device)

    N = pos.shape[0]
    E = edge_idx.shape[1]

    node_radius_t = (
        node_radius
        if isinstance(node_radius, torch.Tensor)
        else torch.tensor(float(node_radius), device=device)
    )

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    # iterate edges (E typically small enough; vectorizing across edges would be
    # more complex but can be added later if needed)
    for e in range(E):
        i = int(edge_idx[0, e].item())
        j = int(edge_idx[1, e].item())

        p1 = pos[i]  # [2]
        p2 = pos[j]  # [2]

        # Vector from p1 to p2
        seg = p2 - p1  # [2]
        seg_len2 = (seg * seg).sum().clamp(min=1e-8)  # avoid div by zero

        # Compute projection t of all points onto the segment (scalar per point)
        # t = dot(p - p1, seg) / |seg|^2
        vecs = pos - p1.unsqueeze(0)  # [N,2]
        t = (vecs * seg.unsqueeze(0)).sum(dim=1) / seg_len2  # [N]
        t_clamped = t.clamp(0.0, 1.0).unsqueeze(1)  # [N,1]

        proj = p1.unsqueeze(0) + t_clamped * seg.unsqueeze(0)  # [N,2]
        dists = torch.norm(pos - proj, dim=1)  # [N]

        # Exclude endpoints if requested
        if exclude_endpoints:
            mask = torch.ones(N, dtype=torch.bool, device=device)
            mask[i] = False
            mask[j] = False
        else:
            mask = torch.ones(N, dtype=torch.bool, device=device)

        # Apply hinge penalty (only where d < node_radius)
        violation = node_radius_t - dists
        violation = violation * mask.to(violation.dtype)
        relu = F.relu(violation)
        sq = relu.pow(2)

        total_loss = total_loss + sq.sum()
        count += int(mask.sum().item())

    if reduction == "mean":
        if count == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / float(count)

    return total_loss


def compute_vae_loss(
    outputs,
    mu,
    logvar,
    x_cont,
    x_cat,
    edge_index,
    edge_attr_cat,
    edge_attr_cont,
    num_nodes,
    beta=1.0,
    lambda_edge_type=1.0,
    lambda_edge_cont=1.0,
    lambda_node_cat=1.0,
    lambda_geo: float = 0.0,
    node_radius: float = 0.3,
    geo_exclude_endpoints: bool = True,
):
    """
    Compute total VAE loss.

    Args:
        outputs: dict from vae.forward() with all model outputs
        mu: [N, latent_dim] encoder mean
        logvar: [N, latent_dim] encoder log-variance
        x_cont: [N, F_x_cont] true continuous node features
        x_cat: [N] true categorical node features
        edge_index: [2, E] true edges
        edge_attr_cat: [E] true edge types
        edge_attr_cont: [E, F_e_cont] true continuous edge features
        num_nodes: number of nodes
        beta: weight for KL term
        lambda_edge_type: weight for edge type loss
        lambda_edge_cont: weight for edge continuous feature loss
        lambda_node_cat: weight for node categorical feature loss

    Returns:
        loss_dict: dict with individual loss terms
        total_loss: scalar
    """
    # KL divergence
    kl = kl_divergence(mu, logvar).mean()

    # Edge losses
    L_exist = edge_existence_loss(outputs["exist_logits"], edge_index, num_nodes)
    L_etype = edge_type_loss(
        outputs["type_logits"], edge_index, edge_attr_cat, num_nodes
    )
    L_econt = edge_cont_loss(
        outputs["cont_mu_e"],
        outputs["cont_logvar_e"],
        edge_index,
        edge_attr_cont,
        num_nodes,
    )

    # Node losses
    L_xcont = node_cont_loss(outputs["cont_mu_x"], outputs["cont_logvar_x"], x_cont)
    L_xcat = node_cat_loss(outputs["cat_logits_x"], x_cat)

    # Geometric loss (optional)
    if lambda_geo and lambda_geo != 0.0:
        # Use predicted node positions (mean) for geometric loss
        try:
            L_geo = geometric_loss(
                outputs["cont_mu_x"],
                edge_index,
                node_radius,
                exclude_endpoints=geo_exclude_endpoints,
                reduction="mean",
            )
        except Exception:
            # Fallback to zero if geometric loss cannot be computed
            L_geo = torch.tensor(
                0.0, device=mu.device if mu is not None else x_cont.device
            )
    else:
        L_geo = torch.tensor(0.0, device=mu.device if mu is not None else x_cont.device)

    # Total
    total_loss = (
        L_exist
        + lambda_edge_type * L_etype
        + lambda_edge_cont * L_econt
        + L_xcont
        + lambda_node_cat * L_xcat
        + lambda_geo * L_geo
        + beta * kl
    )

    loss_dict = {
        "total": total_loss.item(),
        "kl": kl.item(),
        "edge_exist": L_exist.item(),
        "edge_type": L_etype.item(),
        "edge_cont": L_econt.item(),
        "node_cont": L_xcont.item(),
        "node_cat": L_xcat.item(),
        "geo": L_geo.item(),
    }

    return loss_dict, total_loss
