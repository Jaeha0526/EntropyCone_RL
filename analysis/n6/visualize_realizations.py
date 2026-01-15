#!/usr/bin/env python3
"""
Visualize all perfect integer realizations: Ray 146, 180, and 181.

Note: Ray 181 has a disconnected bulk node, so we draw it with N=17.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from pathlib import Path

def combination_to_index(a, b, N):
    if a > b:
        a, b = b, a
    return int(a * (N - 1) - a * (a - 1) // 2 + b - a - 1)

def index_to_combination(idx, N):
    for a in range(N - 1):
        for b in range(a + 1, N):
            if combination_to_index(a, b, N) == idx:
                return (a, b)
    return None

def get_node_positions(N, n=6, inner_rotation_deg=5):
    positions = {}
    outer_radius = 2.5
    n_outer = n + 1

    for i in range(n):
        angle = np.pi/2 + (i / n_outer) * 2 * np.pi
        positions[i] = (outer_radius * np.cos(angle), outer_radius * np.sin(angle))

    purifier_angle = np.pi/2 + (n / n_outer) * 2 * np.pi
    positions[N - 1] = (outer_radius * np.cos(purifier_angle), outer_radius * np.sin(purifier_angle))

    n_bulk = N - n - 1
    inner_radius = 1.35  # Enlarged to spread out internal nodes
    if n_bulk > 0:
        # Rotate so that edge between node 1 and 2 is horizontal, plus optional clockwise rotation
        start_angle = np.pi/2 - np.pi/n_bulk - inner_rotation_deg * np.pi/180
        for i, node_idx in enumerate(range(n, N - 1)):
            angle = start_angle + (i / n_bulk) * 2 * np.pi
            positions[node_idx] = (inner_radius * np.cos(angle), inner_radius * np.sin(angle))

    return positions

def get_node_label(node_idx, N, n=6, custom_labels=None):
    if custom_labels and node_idx in custom_labels:
        return custom_labels[node_idx]
    if node_idx < n:
        return chr(65 + node_idx)
    elif node_idx == N - 1:
        return 'O'
    else:
        return str(node_idx - n + 1)

def get_node_color(node_idx, N, n=6):
    if node_idx < n:
        return '#4ECDC4'
    elif node_idx == N - 1:
        return '#FF6B6B'
    else:
        return '#95E1D3'

def get_weight_color(weight, weight_values):
    colors = [
        '#E74C3C', '#3498DB', '#2ECC71', '#9B59B6',
        '#F39C12', '#1ABC9C', '#E91E63',
    ]
    sorted_weights = sorted(weight_values, reverse=True)
    if weight in sorted_weights:
        idx = sorted_weights.index(weight)
        return colors[idx % len(colors)]
    return '#95A5A6'


def is_outer_node(node_idx, N, n=6):
    """Check if a node is on the outer ring (parties A-F or purifier O)."""
    return node_idx < n or node_idx == N - 1


def get_curve_direction(a, b, positions, N, n=6):
    """
    Determine if and how an edge should be curved to avoid passing through other nodes.
    Returns: (should_curve, curve_factor) where curve_factor > 0 curves outward, < 0 curves inward.
    """
    a_outer = is_outer_node(a, N, n)
    b_outer = is_outer_node(b, N, n)

    pa = np.array(positions[a])
    pb = np.array(positions[b])
    midpoint = (pa + pb) / 2
    dist_to_center = np.linalg.norm(midpoint)

    # Outer-to-outer edges: curve outward (away from center)
    if a_outer and b_outer:
        # Check if the edge would cross through the inner region
        if dist_to_center < 1.5:  # Edge passes close to center
            return True, 0.4  # Curve outward

    # Inner-to-inner edges: curve based on position
    if not a_outer and not b_outer:
        # These edges often cross each other, curve slightly
        return True, 0.25

    # Outer-to-inner edges: check if they cross inner nodes
    if (a_outer and not b_outer) or (not a_outer and b_outer):
        # These usually don't need curving unless they cross the center
        if dist_to_center < 0.5:
            return True, 0.2

    return False, 0


def draw_curved_edge(ax, p1, p2, color, linewidth, alpha, curve_factor):
    """Draw a curved edge using a quadratic Bezier curve."""
    p1 = np.array(p1)
    p2 = np.array(p2)

    # Calculate midpoint and perpendicular direction
    midpoint = (p1 + p2) / 2
    direction = p2 - p1

    # Perpendicular vector (rotate 90 degrees)
    perp = np.array([-direction[1], direction[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-10)

    # Determine curve direction: outward from center
    if np.dot(midpoint, perp) < 0:
        perp = -perp

    # Control point for quadratic Bezier
    edge_length = np.linalg.norm(direction)
    control = midpoint + perp * curve_factor * edge_length

    # Create Bezier curve path
    verts = [tuple(p1), tuple(control), tuple(p2)]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    path = MplPath(verts, codes)

    patch = FancyArrowPatch(
        path=path,
        arrowstyle='-',
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=1
    )
    ax.add_patch(patch)

def visualize_weight_map(w, N, n=6, title="Weight Map", ax=None, custom_labels=None, inner_rotation_deg=5):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    positions = get_node_positions(N, n, inner_rotation_deg=inner_rotation_deg)
    weight_values = sorted(set(wt for wt in w if wt > 0), reverse=True)

    edges = []
    for idx, weight in enumerate(w):
        if weight > 0:
            a, b = index_to_combination(idx, N)
            edges.append((a, b, weight))

    for a, b, weight in edges:
        color = get_weight_color(weight, weight_values)
        x = [positions[a][0], positions[b][0]]
        y = [positions[a][1], positions[b][1]]
        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.8, zorder=1)

    for node_idx in range(N):
        x, y = positions[node_idx]
        color = get_node_color(node_idx, N, n)
        label = get_node_label(node_idx, N, n, custom_labels)

        if node_idx == N - 1:
            size = 800
        elif node_idx < n:
            size = 700
        else:
            size = 300  # Smaller internal nodes to reduce overlap

        ax.scatter(x, y, s=size, c=color, edgecolors='black', linewidths=1.5, zorder=4)
        ax.text(x, y, label, fontsize=12, ha='center', va='center', fontweight='bold', zorder=5)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=28, fontweight='bold')

    edge_legend = []
    for weight in weight_values:
        color = get_weight_color(weight, weight_values)
        edge_legend.append(Line2D([0], [0], color=color, linewidth=3, label=f'w={int(weight)}'))
    # Place legend below plot, at most 3 rows
    ncols = math.ceil(len(weight_values) / 3)
    ax.legend(handles=edge_legend, loc='upper center', bbox_to_anchor=(0.5, 0.02),
              fontsize=20, ncol=ncols, frameon=False, columnspacing=1.0)

    return ax

def load_ray146_weights():
    N = 18
    n_edges = N * (N - 1) // 2
    w = [0] * n_edges
    
    def edge_to_idx(i, j):
        if i > j: i, j = j, i
        return combination_to_index(i, j, N)
    
    def colleague_to_our(v):
        if 1 <= v <= 6: return v - 1
        elif v == 7: return 17
        else: return v - 2
    
    edges = [
        (1, 9, 12), (1, 18, 12), (2, 12, 12), (2, 13, 12), (3, 8, 12), (3, 10, 12),
        (4, 9, 12), (4, 12, 12), (4, 16, 12), (5, 9, 12), (5, 13, 12), (5, 14, 12),
        (6, 13, 12), (6, 17, 12), (6, 18, 12), (7, 8, 12), (7, 12, 12), (7, 18, 12),
        (8, 10, 8), (8, 11, 6), (8, 13, 12), (8, 14, 6), (8, 15, 7),
        (9, 17, 12), (10, 11, 6), (10, 14, 4), (10, 15, 8), (10, 17, 12),
        (11, 14, 2), (11, 15, 3), (11, 17, 6), (12, 15, 12),
        (14, 15, 6), (14, 17, 6), (16, 17, 12), (17, 18, 12),
    ]
    
    for (v1, v2, weight) in edges:
        idx = edge_to_idx(colleague_to_our(v1), colleague_to_our(v2))
        w[idx] = weight
    return w

def load_ray180_weights():
    w = [0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 12, 0, 0, 0, 0, 0, 12, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 12, 0, 12, 0, 0, 0, 0, 12, 0, 0, 0, 12, 12, 0, 0, 0, 0, 0, 12, 0, 0, 12, 12, 0,
         0, 0, 12, 12, 12, 0, 0, 0, 7, 0, 12, 0, 0, 0, 12, 0, 12, 0, 0, 0, 12, 12, 0, 0, 0, 0, 0,
         12, 0, 0, 12, 0, 0, 12, 0, 12]
    return w

def load_ray181_weights_N17():
    """
    Load Ray 181 weights remapped to N=17 (excluding disconnected node 8 -> bulk node "3").
    
    Original N=18: node 8 (bulk "3") is disconnected.
    New N=17: use labels 1-10 for bulk nodes, O for purifier.
    """
    # Original weights for N=18
    w181_N18 = [0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0,
                0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 4, 5,
                9, 0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 0,
                0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0,
                0, 6, 0, 0, 0, 9, 0, 0, 0, 0, 9, 0, 9, 9, 9, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0,
                0, 0, 9]
    
    N_old = 18
    N_new = 17
    removed_node = 8  # bulk node "3" in original numbering
    
    # Create mapping from old node indices to new
    old_to_new = {}
    new_idx = 0
    for old_idx in range(N_old):
        if old_idx == removed_node:
            old_to_new[old_idx] = None  # removed
        else:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    
    # Build new weight array
    n_edges_new = N_new * (N_new - 1) // 2
    w_new = [0] * n_edges_new
    
    for idx, weight in enumerate(w181_N18):
        if weight > 0:
            a, b = index_to_combination(idx, N_old)
            new_a = old_to_new[a]
            new_b = old_to_new[b]
            if new_a is not None and new_b is not None:
                new_idx = combination_to_index(new_a, new_b, N_new)
                w_new[new_idx] = weight
    
    # Custom labels: A-F for parties, 1-10 for bulk, O for purifier
    custom_labels = {}
    for i in range(6):
        custom_labels[i] = chr(65 + i)  # A-F
    for i in range(10):
        custom_labels[6 + i] = str(i + 1)  # 1-10
    custom_labels[16] = 'O'
    
    return w_new, custom_labels

def main():
    output_dir = Path(__file__).parent

    w146 = load_ray146_weights()
    w180 = load_ray180_weights()
    w181, labels181 = load_ray181_weights_N17()

    # Create 3-panel figure with reduced spacing and room for legends
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    visualize_weight_map(w146, N=18, n=6,
                         title="Ray 146",
                         ax=axes[0])

    visualize_weight_map(w180, N=14, n=6,
                         title="Ray 180",
                         ax=axes[1], inner_rotation_deg=0)

    visualize_weight_map(w181, N=17, n=6,
                         title="Ray 181",
                         ax=axes[2], custom_labels=labels181)

    # No main title, tighter spacing between subplots, room for legends at bottom
    plt.subplots_adjust(wspace=-0.05, bottom=0.15)
    
    output_path = output_dir / "ray146_ray180_ray181_realizations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Also update individual Ray 181 plot
    fig2, ax2 = plt.subplots(figsize=(14, 12))
    visualize_weight_map(w181, N=17, n=6,
                         title="Ray 181 Realization (N=17)\nS = 9 × ray_181",
                         ax=ax2, custom_labels=labels181)
    plt.savefig(output_dir / "ray181_realization_graph.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'ray181_realization_graph.png'}")
    plt.close()

if __name__ == "__main__":
    main()
