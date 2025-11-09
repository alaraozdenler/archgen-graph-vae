#!/usr/bin/env python3
"""
Batch visualize all generated graphs from a directory.
"""

import os
from pathlib import Path
from visualize_graph import visualize_graph


def visualize_batch(input_dir, output_dir=None, pattern="generated_*.graphml"):
    """
    Visualize all graphs matching a pattern in a directory.

    Args:
        input_dir: Directory containing .graphml files
        output_dir: Directory to save visualizations (optional)
        pattern: Glob pattern for graph files
    """
    input_path = Path(input_dir)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path / "visualizations"
        output_path.mkdir(parents=True, exist_ok=True)

    # Find all matching files
    graph_files = sorted(input_path.glob(pattern))

    if not graph_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return

    print(f"Found {len(graph_files)} graphs to visualize\n")

    for i, graph_file in enumerate(graph_files, 1):
        print(f"[{i}/{len(graph_files)}] Visualizing {graph_file.name}...")

        # Generate output filename
        output_file = output_path / f"{graph_file.stem}.png"

        try:
            visualize_graph(
                str(graph_file),
                output_path=str(output_file),
                title=f"Generated Graph {i}",
            )
        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\nAll visualizations saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch visualize architectural graphs")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="generated_graphs_final",
        help="Input directory with .graphml files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for visualizations (default: input_dir/visualizations)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="generated_*.graphml",
        help="Glob pattern for graph files",
    )

    args = parser.parse_args()
    visualize_batch(args.input_dir, args.output_dir, args.pattern)
