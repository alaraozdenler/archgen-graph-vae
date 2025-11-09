#!/usr/bin/env python3
"""
Train Relational GraphVAE on the full dataset with custom splits.

This script trains on the entire archgen_graphs.pkl dataset with:
- Full dataset training (no single-graph limitation)
- Configurable train/val/test splits
- Full model capacity
- Extended training epochs
"""

import os
import sys
import argparse
from pathlib import Path

import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "archgen", "src"))

from config import (
    DEVICE,
    LATENT_DIM,
    HIDDEN_DIM,
    ENCODER_NUM_LAYERS,
    NODE_DECODER_NUM_MP_LAYERS,
    BATCH_SIZE,
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
)
from data_utils import load_graphs, create_dataset
from models import RelationalGraphVAE
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch


class GraphDataset(Dataset):
    """Wrapper for list of torch_geometric Data objects."""

    def __init__(self, data_list):
        """Initialize with list of Data objects."""
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def custom_collate_fn(batch):
    """Custom collate function for torch_geometric Data objects."""
    return Batch.from_data_list(batch)


def main():
    parser = argparse.ArgumentParser(
        description="Train Relational GraphVAE on full dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../archgen_graphs.pkl",
        help="Path to graph pickle file",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1000, help="Number of epochs to train"
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay"
    )
    parser.add_argument("--beta", type=float, default=BETA, help="Final KL weight")
    parser.add_argument(
        "--kl_anneal_epochs",
        type=int,
        default=KL_ANNEAL_EPOCHS,
        help="Number of epochs for KL annealing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Device to train on",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints_full_dataset",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.7,
        help="Proportion of data for training",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Proportion of data for validation",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Resume training from checkpoint"
    )

    args = parser.parse_args()

    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    graphs = load_graphs(args.data_path)
    print(f"Loaded {len(graphs)} graphs")

    train_data, val_data, test_data, node_types, edge_types = create_dataset(
        graphs, train_split=args.train_split, val_split=args.val_split
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create data loaders
    train_loader = DataLoader(
        GraphDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        GraphDataset(val_data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        GraphDataset(test_data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    # Create model
    print(f"\nCreating model...")
    model = RelationalGraphVAE(
        in_node_cont_dim=3,  # pos_x, pos_y, angle
        in_node_cat_dim=4,  # 4 node types (vocab size)
        in_edge_cat_dim=3,  # 3 edge types (vocab size)
        in_edge_cont_dim=1,  # length
        out_node_cont_dim=OUT_NODE_CONT_DIM,
        out_node_cat_dim=OUT_NODE_CAT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        encoder_num_layers=ENCODER_NUM_LAYERS,
        node_decoder_num_layers=NODE_DECODER_NUM_MP_LAYERS,
        use_batch_norm=USE_BATCH_NORM,
        dropout=DROPOUT_RATE,
    )
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Import trainer
    from train import Trainer

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        beta=args.beta,
        kl_anneal_epochs=args.kl_anneal_epochs,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train
    print(f"\n{'='*80}")
    print(f"Training on {len(train_data)} graphs (full dataset mode)")
    print(f"{'='*80}\n")
    trainer.fit(num_epochs=args.num_epochs, resume_from=args.resume_from)

    # Evaluate on test set
    print(f"\n{'='*80}")
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate_test()
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
