#!/usr/bin/env python3
"""
Visualize generated or reconstructed graphs.
"""

import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def visualize_graph(graphml_path, output_path=None, title=None):
    """
    Load and visualize a graph from GraphML file.

    Args:
        graphml_path: Path to .graphml file
        output_path: Path to save visualization (optional)
        title: Title for the plot
    """
    # Load graph
    G = nx.read_graphml(graphml_path)

    if title is None:
        title = Path(graphml_path).stem

    print(f"Graph: {title}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Print node details
    print("\n  Node Details:")
    node_type_counts = {}
    for node in G.nodes():
        arch_type = G.nodes[node].get("arch_type", "Unknown")
        pos_x = G.nodes[node].get("pos_x", "N/A")
        pos_y = G.nodes[node].get("pos_y", "N/A")
        angle = G.nodes[node].get("angle", "N/A")
        print(
            f"    Node {node}: type={arch_type}, pos=({pos_x:.2f}, {pos_y:.2f}), angle={angle:.2f}"
        )
        node_type_counts[arch_type] = node_type_counts.get(arch_type, 0) + 1

    print("\n  Node Type Summary:")
    for node_type, count in sorted(node_type_counts.items()):
        print(f"    {node_type}: {count}")

    # Print edge details
    print("\n  Edge Details:")
    for u, v in G.edges():
        edge_type = G[u][v].get("edge_type", "Unknown")
        length = G[u][v].get("length", "N/A")
        print(f"    Edge {u}-{v}: type={edge_type}, length={length:.4f}")

    # Extract node positions
    pos = {}
    for node in G.nodes():
        if "pos_x" in G.nodes[node] and "pos_y" in G.nodes[node]:
            x = float(G.nodes[node]["pos_x"])
            y = float(G.nodes[node]["pos_y"])
            pos[node] = (x, y)
        else:
            # Use spring layout if positions not available
            pass

    if len(pos) < G.number_of_nodes():
        print("  Using spring layout for nodes without position data")
        pos = nx.spring_layout(G, k=2, iterations=50)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, alpha=0.6, edge_color="gray")

    # Color nodes by type
    node_colors = []
    color_map = {
        "Wall": "#FF6B6B",
        "Column": "#4ECDC4",
        "Door": "#45B7D1",
        "Window": "#FFA07A",
    }

    for node in G.nodes():
        arch_type = G.nodes[node].get("arch_type", "Unknown")
        node_colors.append(color_map.get(arch_type, "#999999"))

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=500,
        ax=ax,
        edgecolors="black",
        linewidths=2,
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    # Add edge labels (types)
    edge_labels = {}
    for u, v in G.edges():
        edge_type = G[u][v].get("edge_type", "?")
        edge_labels[(u, v)] = edge_type

    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.axis("off")

    # Add legend
    legend_elements = [
        plt.scatter(
            [], [], c=color, s=200, edgecolors="black", linewidths=2, label=arch_type
        )
        for arch_type, color in color_map.items()
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize architectural graphs")
    parser.add_argument(
        "--graph", type=str, required=True, help="Path to .graphml file"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save visualization (optional)"
    )
    parser.add_argument("--title", type=str, default=None, help="Title for the plot")

    args = parser.parse_args()

    visualize_graph(args.graph, args.output, args.title)


if __name__ == "__main__":
    main()
