#!/usr/bin/env python3
"""
Training loop for Relational GraphVAE.

Implements full training pipeline with:
- Mini-batch stochastic gradient descent
- KL annealing schedule (β increases from 0 to 1 over warmup epochs)
- Gradient clipping
- Validation evaluation
- Checkpoint saving
- Early stopping
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    DATA_PATH,
)
from data_utils import load_graphs, create_dataset
from models import RelationalGraphVAE
from losses import compute_vae_loss
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


class Trainer:
    """Trainer class for Relational GraphVAE."""

    def __init__(
        self,
        model: RelationalGraphVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu",
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        gradient_clip: float = GRADIENT_CLIP_MAX_NORM,
        kl_anneal_epochs: int = KL_ANNEAL_EPOCHS,
        beta: float = BETA,
        checkpoint_dir: str = "checkpoints",
        patience: int = 20,
    ):
        """
        Initialize trainer.

        Args:
            model: RelationalGraphVAE instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader
            device: Device to train on ("cpu" or "cuda")
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            gradient_clip: Max gradient norm for clipping
            kl_anneal_epochs: Number of epochs to anneal KL from 0 to beta
            beta: Final KL weight
            checkpoint_dir: Directory to save checkpoints
            patience: Early stopping patience (# epochs with no improvement)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.gradient_clip = gradient_clip
        self.kl_anneal_epochs = kl_anneal_epochs
        self.beta = beta
        self.patience = patience

        self.optimizer = Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_model_path = None
        self.patience_counter = 0

        # Metrics tracking
        self.train_history = {
            "epoch": [],
            "train_loss": [],
            "train_kl": [],
            "train_recon": [],
            "val_loss": [],
            "val_kl": [],
            "val_recon": [],
            "beta": [],
            "lr": [],
        }

    def get_beta(self, epoch: int) -> float:
        """Get KL weight for current epoch (annealing schedule)."""
        if epoch < self.kl_anneal_epochs:
            # Linear annealing from 0 to beta
            return self.beta * (epoch / self.kl_anneal_epochs)
        return self.beta

    def train_epoch(self, epoch: int, log_every: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        beta = self.get_beta(epoch)

        total_loss = 0.0
        total_kl = 0.0
        total_recon = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = batch.to(self.device)

            # Forward pass
            try:
                outputs, mu, logvar = self.model(
                    batch.x_cont,
                    batch.x_cat,
                    batch.edge_index,
                    batch.edge_attr_cat,
                    batch.edge_attr_cont,
                )
            except Exception as e:
                print(f"Error in forward pass for batch {batch_idx}: {e}")
                continue

            # Compute loss
            try:
                loss_dict, total_batch_loss = compute_vae_loss(
                    outputs=outputs,
                    mu=mu,
                    logvar=logvar,
                    x_cont=batch.x_cont,
                    x_cat=batch.x_cat,
                    edge_index=batch.edge_index,
                    edge_attr_cat=batch.edge_attr_cat,
                    edge_attr_cont=batch.edge_attr_cont,
                    num_nodes=batch.x_cont.shape[0],
                    beta=beta,
                )
            except Exception as e:
                print(f"Error in loss computation for batch {batch_idx}: {e}")
                continue

            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            # Accumulate metrics
            total_loss += total_batch_loss.item()
            total_kl += loss_dict["kl"]  # Already a float
            total_recon += (
                loss_dict["edge_exist"]
                + loss_dict["edge_type"]
                + loss_dict["edge_cont"]
                + loss_dict["node_cont"]
                + loss_dict["node_cat"]
            )  # Already floats
            num_batches += 1

            # Progress logging
            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                avg_loss = total_loss / num_batches
                if (batch_idx + 1) % log_every == 0:
                    print(
                        f"  Batch {batch_idx + 1}/{len(self.train_loader)}: "
                        f"loss={avg_loss:.4f}, β={beta:.4f}"
                    )

        # Average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        avg_kl = total_kl / num_batches if num_batches > 0 else 0.0
        avg_recon = total_recon / num_batches if num_batches > 0 else 0.0

        return {"loss": avg_loss, "kl": avg_kl, "recon": avg_recon, "beta": beta}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        beta = self.get_beta(self.epoch)

        total_loss = 0.0
        total_kl = 0.0
        total_recon = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)

            try:
                outputs, mu, logvar = self.model(
                    batch.x_cont,
                    batch.x_cat,
                    batch.edge_index,
                    batch.edge_attr_cat,
                    batch.edge_attr_cont,
                )

                loss_dict, total_batch_loss = compute_vae_loss(
                    outputs=outputs,
                    mu=mu,
                    logvar=logvar,
                    x_cont=batch.x_cont,
                    x_cat=batch.x_cat,
                    edge_index=batch.edge_index,
                    edge_attr_cat=batch.edge_attr_cat,
                    edge_attr_cont=batch.edge_attr_cont,
                    num_nodes=batch.x_cont.shape[0],
                    beta=beta,
                )
            except Exception as e:
                print(f"Error in validation: {e}")
                continue

            total_loss += total_batch_loss.item()
            total_kl += loss_dict["kl"]  # Already a float
            total_recon += (
                loss_dict["edge_exist"]
                + loss_dict["edge_type"]
                + loss_dict["edge_cont"]
                + loss_dict["node_cont"]
                + loss_dict["node_cat"]
            )  # Already floats
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        avg_kl = total_kl / num_batches if num_batches > 0 else 0.0
        avg_recon = total_recon / num_batches if num_batches > 0 else 0.0

        return {"loss": avg_loss, "kl": avg_kl, "recon": avg_recon}

    def save_checkpoint(self, name: str = "best"):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}_epoch_{self.epoch}.pt"

        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.train_history,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_history = checkpoint["history"]
        print(f"Checkpoint loaded: {checkpoint_path}")

    def fit(self, num_epochs: int, resume_from: Optional[str] = None, log_every: int = 25):
        """
        Train model for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from (optional)
        """
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
            print(f"Resuming training from epoch {self.epoch + 1}")

        print(f"\n{'='*80}")
        print(f"Starting training on {self.device}")
        print(
            f"  Model: {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  KL anneal epochs: {self.kl_anneal_epochs}")
        print(f"  Final β: {self.beta}")
        print(f"  Gradient clip: {self.gradient_clip}")
        print(f"{'='*80}\n")

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            if (epoch + 1) % log_every == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(epoch, log_every=log_every)
            if (epoch + 1) % log_every == 0:
                print(
                    f"  Train: loss={train_metrics['loss']:.4f}, "
                    f"kl={train_metrics['kl']:.4f}, "
                    f"recon={train_metrics['recon']:.4f}"
                )

            # Validate
            val_metrics = self.validate()
            if (epoch + 1) % log_every == 0:
                print(
                    f"  Val:   loss={val_metrics['loss']:.4f}, "
                    f"kl={val_metrics['kl']:.4f}, "
                    f"recon={val_metrics['recon']:.4f}"
                )
            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.train_history["epoch"].append(epoch)
            self.train_history["train_loss"].append(train_metrics["loss"])
            self.train_history["train_kl"].append(train_metrics["kl"])
            self.train_history["train_recon"].append(train_metrics["recon"])
            self.train_history["val_loss"].append(val_metrics["loss"])
            self.train_history["val_kl"].append(val_metrics["kl"])
            self.train_history["val_recon"].append(val_metrics["recon"])
            self.train_history["beta"].append(train_metrics["beta"])
            self.train_history["lr"].append(current_lr)

            # Early stopping & checkpointing
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self.best_model_path = self.save_checkpoint("best")
            else:
                self.patience_counter += 1
                if (epoch + 1) % log_every == 0:
                    print(
                        f"  No improvement for {self.patience_counter}/{self.patience} epochs"
                    )

            # Save periodic checkpoint
            if (epoch + 1) % max(1, num_epochs // 5) == 0:
                self.save_checkpoint(f"epoch_{epoch}")

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

        # Save final history
        self.save_history()
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
        print(f"Best model: {self.best_model_path}")

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / "history.json"

        # Convert numpy arrays to lists for JSON serialization
        history = {
            k: (v if not isinstance(v, np.ndarray) else v.tolist())
            for k, v in self.train_history.items()
        }

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"History saved: {history_path}")

    @torch.no_grad()
    def evaluate_test(self) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.test_loader:
            batch = batch.to(self.device)

            try:
                outputs, mu, logvar = self.model(
                    batch.x_cont,
                    batch.x_cat,
                    batch.edge_index,
                    batch.edge_attr_cat,
                    batch.edge_attr_cont,
                )

                loss_dict, total_batch_loss = compute_vae_loss(
                    outputs=outputs,
                    mu=mu,
                    logvar=logvar,
                    x_cont=batch.x_cont,
                    x_cat=batch.x_cat,
                    edge_index=batch.edge_index,
                    edge_attr_cat=batch.edge_attr_cat,
                    edge_attr_cont=batch.edge_attr_cont,
                    num_nodes=batch.x_cont.shape[0],
                    beta=self.beta,
                )
            except Exception as e:
                print(f"Error in test evaluation: {e}")
                continue

            total_loss += total_batch_loss.item()
            num_batches += 1

        test_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        return {"loss": test_loss}


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train Relational GraphVAE")
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help="Path to graph pickle file",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate"
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
        default="checkpoints",
        help="Directory to save checkpoints",
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

    train_data, val_data, test_data, node_types, edge_types = create_dataset(graphs)
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
    trainer.fit(num_epochs=args.num_epochs, resume_from=args.resume_from)

    # Evaluate on test set
    print(f"\n{'='*80}")
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate_test()
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
