"""Reusable functions for the cookbook."""

import sqlite3
from typing import Any

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def build_graph(
        conn: sqlite3.Connection,
        nodes_as_names: bool = False
        ) -> nx.MultiDiGraph:
    """Build graph using canonical entity IDs and names."""
    graph = nx.MultiDiGraph()

    # Always load canonical mappings
    entity_to_canonical, canonical_names = _load_entity_maps(conn)
    event_temporal_map = _load_event_temporal(conn)

    for t in get_all_triplets(conn):
        if not t["subject_id"]:
            continue

        event_attrs = event_temporal_map.get(t["event_id"])
        _add_triplet_edge(
            graph,
            t,
            entity_to_canonical,
            canonical_names,
            event_attrs,
            nodes_as_names,
        )

    return graph

def _load_entity_maps(conn: sqlite3.Connection) -> tuple[dict[bytes, bytes], dict[bytes, str]]:
    """
    Return mappings for canonical entities:
    • entity_to_canonical: maps entity ID → canonical ID (using resolved_id)
    • canonical_names: maps canonical ID → canonical name.
    """
    cur = conn.cursor()

    # Get all entities with their resolved IDs
    cur.execute("""
        SELECT id, name, resolved_id
        FROM entities
    """)

    entity_to_canonical: dict[bytes, bytes] = {}
    canonical_names: dict[bytes, str] = {}

    for row in cur.fetchall():
        entity_id = row[0]
        name = row[1]
        resolved_id = row[2]

        if resolved_id:
            # If entity has a resolved_id, map to that
            entity_to_canonical[entity_id] = resolved_id
            # Store name of the canonical entity
            canonical_names[resolved_id] = name
        else:
            # If no resolved_id, entity is its own canonical version
            entity_to_canonical[entity_id] = entity_id
            canonical_names[entity_id] = name

    return entity_to_canonical, canonical_names

def _load_event_temporal(conn: sqlite3.Connection) -> dict[bytes, dict[str, Any]]:
    """
    Read the `events` table once and build a mapping
    event_id (bytes) → dict of temporal / descriptive attributes.
    Only the columns that are useful on the graph edges are pulled;
    extend this list freely if you need more.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT  id,
                statement,
                statement_type,
                temporal_type,
                created_at,
                valid_at,
                expired_at,
                invalid_at,
                invalidated_by
        FROM events
    """)
    event_map: dict[bytes, dict[str, Any]] = {}
    for (
        eid,
        statement,
        statement_type,
        temporal_type,
        created_at,
        valid_at,
        expired_at,
        invalid_at,
        invalidated_by,
    ) in cur.fetchall():
        event_map[eid] = {
            "statement": statement,
            "statement_type": statement_type,
            "temporal_type": temporal_type,
            "created_at": created_at,
            "valid_at": valid_at,
            "expired_at": expired_at,
            "invalid_at": invalid_at,
            "invalidated_by": invalidated_by,
        }
    return event_map


def _add_triplet_edge(
        graph: nx.MultiDiGraph, t: dict,
        entity_to_canonical: dict[bytes, bytes],
        canonical_names: dict[bytes, str],
        event_attrs: dict[str, Any] | None = None,
        use_names: bool = False,
        ) -> None:
    """Add one edge using canonical IDs and names."""
    subj_id = t["subject_id"]
    obj_id = t["object_id"]

    if subj_id is None:
        return

    # Get canonical IDs
    canonical_subj = entity_to_canonical.get(subj_id, subj_id)
    canonical_obj = entity_to_canonical.get(obj_id, obj_id) if obj_id else None

    # Get canonical names
    subj_name = canonical_names.get(canonical_subj, t["subject_name"]) if canonical_subj is not None else t["subject_name"]
    obj_name = canonical_names.get(canonical_obj, t["object_name"]) if canonical_obj is not None else t["object_name"]

    subj_node = subj_name if use_names else canonical_subj
    obj_node  = obj_name  if use_names else canonical_obj

    # Add nodes with canonical names
    graph.add_node(
        subj_node,
        canonical_id=canonical_subj,
        name=subj_name,
    )

    # Core edge attributes (triplet-specific)
    edge_attrs: dict[str, Any] = {
        "predicate": t["predicate"],
        "triplet_id": t["id"],
        "event_id": t["event_id"],
        "value": t["value"],
        "canonical_subject_name": subj_name,
        "canonical_object_name": obj_name,
    }

    # Merge in temporal data, if we have it
    if event_attrs:
        edge_attrs.update(event_attrs)

    if canonical_obj is None:
        # Handle self-loops for null objects
        graph.add_edge(
            subj_node, subj_node,
            key=t["predicate"],
            **edge_attrs,
            literal_object=t["object_name"],
        )
    else:
        graph.add_node(
            obj_node,
            canonical_id=canonical_obj,
            name=obj_name,
        )
        graph.add_edge(
            subj_node, obj_node,
            key=t["predicate"],
            **edge_attrs,
        )


def dump_graph_properties():
    G = build_graph(conn)
    # Print descriptive notes about the graph
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Get some basic graph statistics
    print(f"Graph density: {G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1)):.4f}")

    # Sample some nodes to see their attributes
    sample_nodes = list(G.nodes(data=True))[:5]
    print("\nSample nodes (first 5):")
    for node_id, attrs in sample_nodes:
        print(f"  {node_id}: {attrs}")

    # Sample some edges to see their attributes
    sample_edges = list(G.edges(data=True))[:5]
    print("\nSample edges (first 5):")
    for u, v, attrs in sample_edges:
        print(f"  {u} -> {v}: {attrs}")

    # Get degree statistics
    degrees = [d for _, d in G.degree()]
    print("\nDegree statistics:")
    print(f"  Min degree: {min(degrees)}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Average degree: {sum(degrees) / len(degrees):.2f}")

    # Check if graph is connected (considering it as undirected for connectivity)
    undirected_G = G.to_undirected()
    print("\nConnectivity:")
    print(f"  Number of connected components: {len(list(nx.connected_components(undirected_G)))}")
    print(f"  Is weakly connected: {nx.is_weakly_connected(G)}")


def visualize_graph():
    # Create a smaller subgraph for visualization (reduce data for clarity)
    # Get nodes with highest degrees for a meaningful visualization
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]  # Reduced from 30 to 20
    visualization_nodes = [node for node, _ in top_nodes]

    # Create subgraph with these high-degree nodes
    graph = G.subgraph(visualization_nodes)
    print(f"Visualization subgraph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Create the plot with better styling
    fig, ax = plt.subplots(figsize=(18, 14))
    fig.patch.set_facecolor("white")

    # Use hierarchical layout for better structure
    try:
        # Try hierarchical layout first
        pos = nx.nx_agraph.graphviz_layout(graph, prog="neato")
    except (ImportError, nx.NetworkXException):
        # Fall back to spring layout with better parameters
        pos = nx.spring_layout(graph, k=5, iterations=100, seed=42)

    # Calculate node properties
    node_degrees = [degrees[node] for node in graph.nodes()]
    max_degree = max(node_degrees)
    min_degree = min(node_degrees)

    # Create better color scheme
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(node_degrees)))
    node_colors = [colors[i] for i in range(len(node_degrees))]

    # Draw nodes with improved styling
    node_sizes = [max(200, min(2000, deg * 50)) for deg in node_degrees]  # Better size scaling
    nx.draw_networkx_nodes(graph, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9,
                           edgecolors="black",
                           linewidths=1.5,
                           ax=ax)

    # Draw edges with better styling
    edge_weights = []
    for _, _, _ in graph.edges(data=True):
        edge_weights.append(1)

    nx.draw_networkx_edges(graph, pos,
                           alpha=0.4,
                           edge_color="#666666",
                           width=1.0,
                           arrows=True,
                           arrowsize=15,
                           arrowstyle="->",
                           ax=ax)

    # Add labels for all nodes with better formatting
    labels = {}
    for node in graph.nodes():
        node_name = graph.nodes[node].get("name", str(node))
        # Truncate long names
        if len(node_name) > 15:
            node_name = node_name[:12] + "..."
        labels[node] = node_name

    nx.draw_networkx_labels(graph, pos, labels,
                            font_size=9,
                            font_weight="bold",
                            font_color="black",  # changed from 'white' to 'black'
                            ax=ax)

    # Improve title and styling
    ax.set_title("Temporal Knowledge Graph Visualization\n(Top 20 Most Connected Entities)",
                 fontsize=18, fontweight="bold", pad=20)
    ax.axis("off")

    # Add a better colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                               norm=plt.Normalize(vmin=min_degree, vmax=max_degree))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=30)
    cbar.set_label("Node Degree (Number of Connections)", rotation=270, labelpad=25, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Add margin around the graph
    ax.margins(0.1)

    plt.tight_layout()
    plt.show()

    # Print some information about the visualized nodes
    print("\nTop entities in visualization:")
    for i, (node, degree) in enumerate(top_nodes[:10]):
        node_name = G.nodes[node].get("name", "Unknown")
        print(f"{i+1:2d}. {node_name} (connections: {degree})")

    # Create an improved function for easier graph visualization
    def visualise_graph(G, num_nodes=20, figsize=(16, 12)):
        """
        Visualize a NetworkX graph with improved styling and reduced data.

        Args:
            G: NetworkX graph
            num_nodes: Number of top nodes to include in visualization (default: 20)
            figsize: Figure size tuple

        """
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:num_nodes]
        visualization_nodes = [node for node, _ in top_nodes]

        # Create subgraph
        subgraph = G.subgraph(visualization_nodes)

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")

        # Layout with better parameters
        try:
            pos = nx.nx_agraph.graphviz_layout(subgraph, prog="neato")
        except (ImportError, nx.NetworkXException):
            pos = nx.spring_layout(subgraph, k=4, iterations=100, seed=42)

        # Node properties
        node_degrees = [degrees[node] for node in subgraph.nodes()]
        max_degree = max(node_degrees)
        min_degree = min(node_degrees)

        # Better color scheme
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(node_degrees)))
        node_colors = list(colors)

        # Draw nodes
        node_sizes = [max(200, min(2000, deg * 50)) for deg in node_degrees]
        nx.draw_networkx_nodes(subgraph, pos,
                               node_color=node_colors,
                               node_size=node_sizes,
                               alpha=0.9,
                               edgecolors="black",
                               linewidths=1.5,
                               ax=ax)

        # Draw edges
        nx.draw_networkx_edges(subgraph, pos,
                               alpha=0.4,
                               edge_color="#666666",
                               width=1.0,
                               arrows=True,
                               arrowsize=15,
                               ax=ax)

        # Labels
        labels = {}
        for node in subgraph.nodes():
            node_name = subgraph.nodes[node].get("name", str(node))
            if len(node_name) > 15:
                node_name = node_name[:12] + "..."
            labels[node] = node_name

        nx.draw_networkx_labels(subgraph, pos, labels,
                                font_size=9,
                                font_weight="bold",
                                font_color="black",  # changed from 'white' to 'black'
                                ax=ax)

        ax.set_title(f"Temporal Knowledge Graph\n(Top {num_nodes} Most Connected Entities)",
                     fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                   norm=plt.Normalize(vmin=min_degree, vmax=max_degree))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label("Connections", rotation=270, labelpad=20)

        ax.margins(0.1)
        plt.tight_layout()
        plt.show()

        return subgraph
