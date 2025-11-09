#!/usr/bin/env python3
"""
Filter and sparsify generated graphs to reduce edge density.

Options:
- Remove edges based on length threshold
- Keep only top-K edges by probability or strength
- Random edge removal
"""

import os
from pathlib import Path
import networkx as nx
import argparse
from typing import Optional


def filter_graph_by_edge_count(G: nx.Graph, max_edges: int) -> nx.Graph:
    """
    Keep only the top max_edges edges, sorted by some criterion.

    Args:
        G: NetworkX graph
        max_edges: Maximum number of edges to keep

    Returns:
        Filtered graph
    """
    if G.number_of_edges() <= max_edges:
        return G.copy()

    # Sort edges by length (prefer shorter edges)
    edges_with_length = []
    for u, v in G.edges():
        length = abs(float(G[u][v].get("length", 1.0)))
        edges_with_length.append((length, u, v))

    edges_with_length.sort()  # Sort by length ascending

    # Create new graph with only top edges
    G_filtered = nx.Graph()

    # Add nodes
    for node in G.nodes():
        G_filtered.add_node(node, **G.nodes[node])

    # Add top edges
    for _, u, v in edges_with_length[:max_edges]:
        G_filtered.add_edge(u, v, **G[u][v])

    return G_filtered


def filter_graph_by_length_threshold(G: nx.Graph, max_length: float = 2.0) -> nx.Graph:
    """
    Remove edges with length > max_length.

    Args:
        G: NetworkX graph
        max_length: Maximum edge length to keep

    Returns:
        Filtered graph
    """
    G_filtered = G.copy()

    edges_to_remove = []
    for u, v in G_filtered.edges():
        length = abs(float(G_filtered[u][v].get("length", 0.0)))
        if length > max_length:
            edges_to_remove.append((u, v))

    G_filtered.remove_edges_from(edges_to_remove)

    return G_filtered


def filter_graph_by_degree(G: nx.Graph, max_degree: int = 10) -> nx.Graph:
    """
    Remove edges from high-degree nodes to limit connectivity.

    Args:
        G: NetworkX graph
        max_degree: Maximum degree per node

    Returns:
        Filtered graph
    """
    G_filtered = G.copy()

    # Iteratively remove edges from high-degree nodes
    changed = True
    iterations = 0
    max_iterations = 100

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        # Get all nodes and their current degrees
        node_list = list(G_filtered.nodes())
        for node in node_list:
            # Count neighbors
            neighbors = list(G_filtered.neighbors(node))
            degree = len(neighbors)

            if degree > max_degree:
                # Get edges sorted by length
                edges = []
                for neighbor in neighbors:
                    edge_data = G_filtered[node][neighbor]
                    length = abs(float(edge_data.get("length", 0)))
                    edges.append((length, node, neighbor))

                # Sort by length (remove longest first)
                edges.sort(reverse=True)

                # Remove longest edges beyond max_degree
                edges_to_remove = edges[: degree - max_degree]
                for _, u, v in edges_to_remove:
                    G_filtered.remove_edge(u, v)
                    changed = True

    return G_filtered


def batch_filter_graphs(
    input_dir: str, output_dir: str, filter_type: str = "edge_count", **filter_args
):
    """
    Filter all graphs in a directory.

    Args:
        input_dir: Directory with .graphml files
        output_dir: Output directory for filtered graphs
        filter_type: Type of filtering: 'edge_count', 'length', or 'degree'
        **filter_args: Arguments for the filter function
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all graphml files
    graph_files = sorted(input_path.glob("*.graphml"))

    if not graph_files:
        print(f"No GraphML files found in {input_dir}")
        return

    print(f"Filtering {len(graph_files)} graphs with '{filter_type}' method\n")

    stats = {
        "original_edges": [],
        "filtered_edges": [],
        "original_nodes": [],
        "filtered_nodes": [],
    }

    for i, graph_file in enumerate(graph_files, 1):
        print(f"[{i}/{len(graph_files)}] Processing {graph_file.name}...")

        # Load graph
        G = nx.read_graphml(graph_file)
        print(f"  Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Filter
        if filter_type == "edge_count":
            G_filtered = filter_graph_by_edge_count(G, **filter_args)
        elif filter_type == "length":
            G_filtered = filter_graph_by_length_threshold(G, **filter_args)
        elif filter_type == "degree":
            G_filtered = filter_graph_by_degree(G, **filter_args)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        print(
            f"  Filtered:  {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges"
        )

        # Save
        output_file = output_path / graph_file.name
        nx.write_graphml(G_filtered, output_file)

        # Track stats
        stats["original_edges"].append(G.number_of_edges())
        stats["filtered_edges"].append(G_filtered.number_of_edges())
        stats["original_nodes"].append(G.number_of_nodes())
        stats["filtered_nodes"].append(G_filtered.number_of_nodes())

    # Print summary
    print(f"\n{'='*60}")
    print(f"Filtering Summary ({filter_type}):")
    print(f"{'='*60}")
    print(
        f"Original edges:   {sum(stats['original_edges']):5d} avg: {sum(stats['original_edges'])/len(stats['original_edges']):.1f}"
    )
    print(
        f"Filtered edges:   {sum(stats['filtered_edges']):5d} avg: {sum(stats['filtered_edges'])/len(stats['filtered_edges']):.1f}"
    )
    print(
        f"Edge reduction:   {100*(1 - sum(stats['filtered_edges'])/sum(stats['original_edges'])):.1f}%"
    )
    print(f"\nGraphs saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and sparsify graphs")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="generated_graphs_final",
        help="Input directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_graphs_filtered",
        help="Output directory",
    )
    parser.add_argument(
        "--filter_type",
        type=str,
        choices=["edge_count", "length", "degree"],
        default="degree",
        help="Filter type",
    )
    parser.add_argument(
        "--max_edges",
        type=int,
        default=100,
        help="Max edges per graph (for edge_count filter)",
    )
    parser.add_argument(
        "--max_length",
        type=float,
        default=2.0,
        help="Max edge length (for length filter)",
    )
    parser.add_argument(
        "--max_degree",
        type=int,
        default=8,
        help="Max degree per node (for degree filter)",
    )

    args = parser.parse_args()

    if args.filter_type == "edge_count":
        batch_filter_graphs(
            args.input_dir, args.output_dir, "edge_count", max_edges=args.max_edges
        )
    elif args.filter_type == "length":
        batch_filter_graphs(
            args.input_dir, args.output_dir, "length", max_length=args.max_length
        )
    elif args.filter_type == "degree":
        batch_filter_graphs(
            args.input_dir, args.output_dir, "degree", max_degree=args.max_degree
        )
