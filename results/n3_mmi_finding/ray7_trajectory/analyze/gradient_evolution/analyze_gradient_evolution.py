#!/usr/bin/env python3
"""
Comprehensive gradient evolution analysis across stages (Cold Start).
Generates: distribution_evolution, mmi_tracking, gradient_facet_heatmap plots.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Get experiment directory from script location
script_dir = os.path.dirname(os.path.abspath(__file__))
experiment_dir = os.path.dirname(os.path.dirname(script_dir))  # go up from gradient_evolution -> analyze -> experiment
experiment_name = os.path.basename(experiment_dir)

print("="*80)
print(f"GRADIENT EVOLUTION ANALYSIS (Cold Start)")
print(f"Experiment: {experiment_name}")
print("="*80)

# Find project root (go up from gradient_evolution -> analyze -> experiment -> n3_mmi_finding -> results -> entropyconeRL)
project_root = os.path.abspath(os.path.join(script_dir, '../../../../../..'))
facets_dir = os.path.join(project_root, 'extremal_rays/n3/MMI_finding')

# Load SA and MMI facets
def conventional_to_binary(vec):
    """Convert from conventional [S_A,S_B,S_C,S_AB,S_AC,S_BC,S_ABC] to binary [S_A,S_B,S_AB,S_C,S_AC,S_BC,S_ABC]"""
    result = vec.copy()
    result[2], result[3] = vec[3], vec[2]  # Swap S_C (idx 2) and S_AB (idx 3)
    return result

sa_facets = []
sa_file = os.path.join(facets_dir, 'sa_facets.txt')
if os.path.exists(sa_file):
    with open(sa_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                sa_facets.append([float(x) for x in line.split()])
    sa_facets = np.array(sa_facets)
    # Convert SA facets from conventional to binary ordering
    sa_facets = np.array([conventional_to_binary(f) for f in sa_facets])
else:
    sa_facets = np.array([]).reshape(0, 7)

mmi_facets = []
mmi_file = os.path.join(facets_dir, 'mmi_facets.txt')
if os.path.exists(mmi_file):
    with open(mmi_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                mmi_facets.append([float(x) for x in line.split()])
    mmi_facets = np.array(mmi_facets)
else:
    # Use standard MMI direction for n=3
    mmi_facets = np.array([[-1, -1, +1, -1, +1, +1, -1]])

if len(sa_facets) > 0:
    all_facets = np.vstack([sa_facets, mmi_facets])
else:
    all_facets = mmi_facets

n_sa = len(sa_facets)
n_mmi = len(mmi_facets)

print(f"Loaded {n_sa} SA facets + {n_mmi} MMI facets")

# Normalize facets
def normalize_facets(facets):
    if len(facets) == 0:
        return facets
    norms = np.linalg.norm(facets, axis=1, keepdims=True)
    return facets / norms

sa_normed = normalize_facets(sa_facets)
mmi_normed = normalize_facets(mmi_facets)
if len(sa_normed) > 0:
    all_normed = np.vstack([sa_normed, mmi_normed])
else:
    all_normed = mmi_normed

# Load experiment results
exp_file = os.path.join(experiment_dir, 'experiment_results.json')
if not os.path.exists(exp_file):
    print(f"ERROR: No experiment_results.json found")
    print("Experiment may still be running...")
    exit(1)

with open(exp_file, 'r') as f:
    exp_data = json.load(f)

# Extract stage data
stages = []
positions = []
gradients = []

for entry in exp_data:
    if 'stage' in entry and 'position' in entry:
        stages.append(entry['stage'])
        positions.append(np.array(entry['position']))
        if 'gradient' in entry:
            gradients.append(np.array(entry['gradient']))
        else:
            gradients.append(None)

print(f"Found {len(stages)} stages with data")

if len(stages) == 0:
    print("No stage data found yet. Experiment may still be on first stage.")
    exit(0)

# Compute inner products for each stage
mmi_colors = ['red', 'darkorange', 'purple']

# ========== Plot 1: Distribution Evolution ==========
valid_stages_grads = [(s, g) for s, g in zip(stages, gradients) if g is not None]
n_plots = min(len(valid_stages_grads), 6)  # Max 6 stages

if n_plots > 0:
    fig, axes = plt.subplots(1, max(n_plots, 1), figsize=(4*max(n_plots, 1), 4))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(f'{experiment_name}: Gradient-Facet Distribution Evolution (Cold Start)', fontsize=12, weight='bold')

    for i, (stage, grad) in enumerate(valid_stages_grads[:n_plots]):
        ax = axes[i]
        grad_norm = grad / (np.linalg.norm(grad) + 1e-10)

        # Compute inner products with all facets
        if len(sa_normed) > 0:
            sa_inner = sa_normed @ grad_norm
            ax.hist(sa_inner, bins=12, alpha=0.7, color='blue', edgecolor='black', label='SA facets')

        mmi_inner = mmi_normed @ grad_norm

        # Mark MMI facets
        for j, ip in enumerate(mmi_inner):
            ax.axvline(ip, color=mmi_colors[j % len(mmi_colors)], linestyle='--', linewidth=2,
                      label=f'MMI_{j}: {ip:+.3f}')

        ax.axvline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel('Inner Product (Cosine Similarity)')
        ax.set_ylabel('Number of Facets')
        ax.set_title(f'Stage {stage}')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'distribution_evolution.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: distribution_evolution.png")
    plt.close()

# ========== Plot 2: MMI Tracking ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'{experiment_name}: MMI Facet Tracking (Cold Start)', fontsize=14, weight='bold')

# Track MMI alignments
mmi_alignments = [[] for _ in range(n_mmi)]
valid_stages = []

for stage, grad in zip(stages, gradients):
    if grad is not None:
        grad_norm = grad / (np.linalg.norm(grad) + 1e-10)
        valid_stages.append(stage)
        for j in range(n_mmi):
            mmi_alignments[j].append(mmi_normed[j] @ grad_norm)

# Plot 2a: MMI Alignment over stages
ax = axes[0, 0]
for j in range(n_mmi):
    if mmi_alignments[j]:
        ax.plot(valid_stages, mmi_alignments[j], 'o-', color=mmi_colors[j % len(mmi_colors)],
               linewidth=2, markersize=8, label=f'MMI_{j}')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.axhline(0.5, color='green', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(-0.5, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Stage')
ax.set_ylabel('Gradient-Facet Alignment')
ax.set_title('MMI Facet Alignment with Gradient\n(Positive = moving toward boundary)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2b: MMI facet values (distance from boundary) - using normalized facets
ax = axes[0, 1]
for i, pos in enumerate(positions):
    mmi_vals = mmi_normed @ pos  # Use normalized facets for proper distance
    for j in range(n_mmi):
        if i == 0:
            ax.plot(stages[i], mmi_vals[j], 'o', color=mmi_colors[j % len(mmi_colors)], markersize=10, label=f'MMI_{j}')
        else:
            ax.plot(stages[i], mmi_vals[j], 'o', color=mmi_colors[j % len(mmi_colors)], markersize=10)

# Connect with lines
for j in range(n_mmi):
    vals = [mmi_normed[j] @ pos for pos in positions]  # Use normalized facets
    ax.plot(stages, vals, '-', color=mmi_colors[j % len(mmi_colors)], linewidth=2, alpha=0.7)

ax.axhline(0, color='green', linestyle='--', linewidth=2, label='Boundary')
ax.set_xlabel('Stage')
ax.set_ylabel('MMI Distance (normalized)')
ax.set_title('Distance from MMI Boundary\n(Goal: Increase toward 0)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2c: Angle from gradient (degrees)
ax = axes[1, 0]
for j in range(n_mmi):
    if mmi_alignments[j]:
        angles = [np.arccos(np.clip(a, -1, 1)) * 180 / np.pi for a in mmi_alignments[j]]
        ax.plot(valid_stages, angles, 'o-', color=mmi_colors[j % len(mmi_colors)],
               linewidth=2, markersize=8, label=f'MMI_{j}')
ax.axhline(90, color='gray', linestyle='--', linewidth=1, label='Orthogonal')
ax.set_xlabel('Stage')
ax.set_ylabel('Angle (degrees)')
ax.set_title('MMI Facet Angle from Gradient\n(<90° = moving toward boundary)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2d: Summary - MMI violation and best alignment
ax = axes[1, 1]
mmi_min_values = [min(mmi_facets @ pos) for pos in positions]

ax2 = ax.twinx()
ax.plot(stages, mmi_min_values, 'o-', color='red', linewidth=2, markersize=8, label='MMI min (violation)')
ax.axhline(0, color='green', linestyle='--', linewidth=2, label='Boundary')
ax.set_xlabel('Stage')
ax.set_ylabel('MMI min value', color='red')
ax.tick_params(axis='y', labelcolor='red')

if valid_stages and mmi_alignments[0]:
    best_alignments = []
    for i in range(len(valid_stages)):
        vals = [mmi_alignments[j][i] if i < len(mmi_alignments[j]) else 0 for j in range(n_mmi)]
        best_alignments.append(max(vals))

    ax2.plot(valid_stages, best_alignments, 's-', color='blue', linewidth=2, markersize=8, label='Best alignment')
    ax2.set_ylabel('Best MMI alignment', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

ax.set_title('MMI Violation & Best Alignment')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'mmi_tracking.png'), dpi=150, bbox_inches='tight')
print(f"Saved: mmi_tracking.png")
plt.close()

# ========== Plot 3: Gradient-Facet Heatmap ==========
if valid_stages_grads:
    n_valid = len(valid_stages_grads)

    heatmap_data = np.zeros((len(all_facets), n_valid))
    for i, (s, g) in enumerate(valid_stages_grads):
        g_norm = g / (np.linalg.norm(g) + 1e-10)
        heatmap_data[:, i] = all_normed @ g_norm

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    if n_sa > 0:
        ax.axhline(n_sa - 0.5, color='white', linewidth=2)
    ax.set_xlabel('Stage')
    ax.set_ylabel('Facet Index')
    ax.set_title(f'{experiment_name}: Gradient-Facet Inner Products (Cold Start)\n'
                f'(Red=positive alignment, Blue=negative)')
    ax.set_xticks(range(n_valid))
    ax.set_xticklabels([s for s, _ in valid_stages_grads])
    plt.colorbar(im, ax=ax, label='Inner Product')

    # Add labels for SA/MMI sections
    if n_sa > 0:
        ax.text(n_valid + 0.2, n_sa/2, 'SA', fontsize=10, va='center')
        ax.text(n_valid + 0.2, n_sa + n_mmi/2, 'MMI', fontsize=10, va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'gradient_facet_heatmap.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: gradient_facet_heatmap.png")
    plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
