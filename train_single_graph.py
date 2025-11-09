#!/usr/bin/env python3
"""
Train on a single graph for quick testing.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "archgen", "src"))

from config import (
    DEVICE,
    LATENT_DIM,
    HIDDEN_DIM,
    ENCODER_NUM_LAYERS,
    NODE_DECODER_NUM_MP_LAYERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRADIENT_CLIP_MAX_NORM,
    BETA,
    KL_ANNEAL_EPOCHS,
    DROPOUT_RATE,
    USE_BATCH_NORM,
    USE_CUDA,
    OUT_NODE_CONT_DIM,
    OUT_NODE_CAT_DIM,
    DATA_PATH,
)
from data_utils import load_graphs, create_dataset
from models import RelationalGraphVAE
from torch_geometric.data import Batch

# Import trainer from train.py
sys.path.insert(0, os.path.dirname(__file__))
from train import GraphDataset, custom_collate_fn, Trainer


def main():
    """Train on only the first graph."""
    import argparse

    parser = argparse.ArgumentParser(description="Train on a single graph")
    parser.add_argument(
        "--num_epochs", type=int, default=1000, help="Number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    # Set device
    device = DEVICE
    print(f"Using device: {device}\n")  # Load data
    print(f"Loading data from {DATA_PATH}...")
    graphs = load_graphs(DATA_PATH)
    print(f"Loaded {len(graphs)} graphs")

    train_data, val_data, test_data, node_types, edge_types = create_dataset(graphs)
    print(f"Total train: {len(train_data)}")

    # Use only the first graph for training
    print("\n" + "=" * 80)
    print("SINGLE GRAPH TRAINING MODE")
    print("=" * 80)
    print(f"Using only graph 0 for training")
    print(f"  Nodes: {train_data[0].x_cont.shape[0]}")
    if train_data[0].edge_index is not None:
        print(f"  Edges: {train_data[0].edge_index.shape[1]}")
    else:
        print(f"  Edges: 0")
    print()

    # Use only the first graph for all splits (train, val, test)
    single_graph = [train_data[0]]
    train_loader = DataLoader(
        GraphDataset(single_graph),
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        GraphDataset(single_graph),
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        GraphDataset(single_graph),
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    # Create model
    print(f"Creating model...")
    model = RelationalGraphVAE(
        in_node_cont_dim=3,
        in_node_cat_dim=4,
        in_edge_cat_dim=3,
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
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Create trainer with high patience to avoid early stopping
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        kl_anneal_epochs=KL_ANNEAL_EPOCHS,
        checkpoint_dir="checkpoints_single_graph",
        patience=10000,  # Very high patience to train full 500 epochs
    )

    # Train
    trainer.fit(num_epochs=args.num_epochs)

    # Evaluate on test set
    print(f"\n{'='*80}")
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate_test()
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"{'='*80}")

    # --- Generate 10 example graphs after evaluation ---
    print("\nGenerating 10 example graphs from trained model...")
    from sample import GraphSampler
    import networkx as nx
    from pathlib import Path

    # Load best checkpoint (from trainer)
    if hasattr(trainer, "best_model_path") and trainer.best_model_path is not None:
        checkpoint_path = trainer.best_model_path
    else:
        checkpoint_path = os.path.join("checkpoints_single_graph", "best_epoch_0.pt")
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sampler = GraphSampler(model, device=device)
    num_nodes = (
        train_data[0].x_cont.shape[0] if hasattr(train_data[0], "x_cont") else 50
    )
    generated_graphs = sampler.sample(num_nodes=num_nodes, num_samples=10)

    output_dir = Path("generated_graphs_single_graph")
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, data in enumerate(generated_graphs):
        G = sampler.to_networkx(data, remove_isolated=True)
        save_path = output_dir / f"generated_{i}.graphml"
        nx.write_graphml(G, save_path)
        print(
            f"  Saved generated_{i}.graphml: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
    print(f"\nExample graphs saved to {output_dir}")


if __name__ == "__main__":
    main()
