#!/usr/bin/env python3
"""
Hyperparameter search runner for Relational GraphVAE.

Supports simple grid search or random search over a small set of hyperparameters.
This script runs short trials (few epochs) and records validation loss for each
trial so you can select promising settings.

Usage examples:
  python hpo.py --mode grid --lambda_geo 0.0 0.1 0.5 --node_radius 0.1 0.2 --epochs 3 --dry_run

Be cautious: actual training can be expensive. Use --dry_run to only enumerate
experiments without training.
"""

import argparse
import itertools
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader

# Add src to path for imports (same as other scripts)
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_graphs, create_dataset
from models import RelationalGraphVAE
from train import GraphDataset, custom_collate_fn, Trainer
import config

DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-3
DEFAULT_KL_ANNEAL = 10
DEFAULT_DATA_PATH = getattr(
    config, "DATA_PATH", os.path.join(os.path.dirname(__file__), "data")
)


def make_experiments_grid(search_space: Dict[str, List[Any]]):
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def make_experiments_random(search_space: Dict[str, List[Any]], n: int, seed: int = 0):
    random.seed(seed)
    keys = list(search_space.keys())
    for _ in range(n):
        combo = {k: random.choice(search_space[k]) for k in keys}
        yield combo


def run_trial(cfg: Dict[str, Any], graphs, dry_run: bool = False) -> Dict[str, Any]:
    """Run a single trial: instantiate model/trainer, run short training, return metrics."""
    # Create dataset
    train_data, val_data, test_data, node_types, edge_types = create_dataset(graphs)

    train_loader = DataLoader(
        GraphDataset(train_data),
        batch_size=cfg.get("batch_size", DEFAULT_BATCH_SIZE),
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        GraphDataset(val_data),
        batch_size=cfg.get("batch_size", DEFAULT_BATCH_SIZE),
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        GraphDataset(test_data),
        batch_size=cfg.get("batch_size", DEFAULT_BATCH_SIZE),
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    if dry_run:
        # Return dataset sizes and planned cfg
        return {
            "cfg": cfg,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
        }

    # Build model: only pass kwargs that are provided to avoid passing None where ints are expected
    model_kwargs = {
        "in_node_cont_dim": cfg.get("in_node_cont_dim", 3),
        "in_node_cat_dim": cfg.get("in_node_cat_dim", 4),
        "in_edge_cat_dim": cfg.get("in_edge_cat_dim", 3),
        "in_edge_cont_dim": cfg.get("in_edge_cont_dim", 1),
        "out_node_cont_dim": cfg.get(
            "out_node_cont_dim", getattr(config, "OUT_NODE_CONT_DIM", 3)
        ),
        "out_node_cat_dim": cfg.get(
            "out_node_cat_dim", getattr(config, "OUT_NODE_CAT_DIM", 4)
        ),
        "use_batch_norm": cfg.get(
            "use_batch_norm", getattr(config, "USE_BATCH_NORM", False)
        ),
        "dropout": cfg.get("dropout", getattr(config, "DROPOUT_RATE", 0.0)),
    }

    # optional int params
    optional_ints = [
        ("hidden_dim", "hidden_dim"),
        ("latent_dim", "latent_dim"),
        ("encoder_num_layers", "encoder_layers"),
        ("node_decoder_num_layers", "node_decoder_layers"),
    ]
    for param_name, cfg_key in optional_ints:
        val = cfg.get(cfg_key)
        if val is not None:
            model_kwargs[param_name] = int(val)

    model = RelationalGraphVAE(**model_kwargs)

    device = cfg.get("device", config.DEVICE)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=cfg.get("learning_rate", DEFAULT_LR),
        weight_decay=cfg.get("weight_decay", 0.0),
        gradient_clip=cfg.get("gradient_clip", 0.0),
        kl_anneal_epochs=cfg.get("kl_anneal_epochs", DEFAULT_KL_ANNEAL),
        beta=cfg.get("beta", 1.0),
        checkpoint_dir=cfg.get("checkpoint_dir", "hpo_checkpoints"),
        patience=cfg.get("patience", 5),
    )

    # Run short training
    trainer.fit(num_epochs=cfg.get("epochs", 3), log_every=2)

    # After training, collect validation metric and return
    result = {
        "cfg": cfg,
        "best_val_loss": trainer.best_val_loss,
        "best_model": str(trainer.best_model_path),
        "history": trainer.train_history,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search runner")
    parser.add_argument("--mode", choices=["grid", "random"], default="grid")
    parser.add_argument(
        "--dry_run", action="store_true", help="Only enumerate experiments"
    )
    parser.add_argument("--output_dir", type=str, default="hpo_results")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per trial")
    parser.add_argument(
        "--trials", type=int, default=10, help="Number of random trials"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define a simple search space (customize as needed)
    search_space = {
        "lambda_geo": [0.3, 0.5, 0.7],
        "node_radius": [0.1, 0.5, 1.0],
        "learning_rate": [1e-3, 5e-4],
        "beta": [0.1, 1.0],
        "batch_size": [8, 16, 32],
        "latent_dim": [8, 16, 32],
    }

    data_path = getattr(config, "DATA_PATH", DEFAULT_DATA_PATH)
    print("Loading data from", data_path)
    graphs = load_graphs(data_path)
    print(f"Loaded {len(graphs)} graphs")

    experiments = []
    if args.mode == "grid":
        experiments = list(make_experiments_grid(search_space))
    else:
        experiments = list(
            make_experiments_random(search_space, n=args.trials, seed=args.seed)
        )

    print(f"Planned {len(experiments)} experiments")

    results = []
    for idx, cfg in enumerate(experiments, 1):
        # augment cfg with epochs and checkpoint dir per trial
        cfg["epochs"] = args.epochs
        cfg["checkpoint_dir"] = str(output_dir / f"trial_{idx}_ckpt")
        cfg["device"] = config.DEVICE

        print(f"\n[{idx}/{len(experiments)}] Trial cfg: {cfg}")
        res = run_trial(cfg, graphs, dry_run=args.dry_run)

        # Save per-trial JSON
        out_file = output_dir / f"trial_{idx}.json"
        with open(out_file, "w") as f:
            json.dump(
                res,
                f,
                default=lambda o: (
                    o if isinstance(o, (int, float, str, bool)) else str(o)
                ),
                indent=2,
            )
        results.append(res)

        if args.dry_run:
            # in dry-run mode just enumerate
            continue

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            results,
            f,
            default=lambda o: o if isinstance(o, (int, float, str, bool)) else str(o),
            indent=2,
        )

    print(f"\nAll done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
