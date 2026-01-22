#!/usr/bin/env python3
"""
Plot evolution of all SA and MMI facet values over experiment stages.
This version is for cold start experiments in n3_mmi_finding directory.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Get experiment directory from script location
script_dir = os.path.dirname(os.path.abspath(__file__))
experiment_dir = os.path.dirname(script_dir)
experiment_name = os.path.basename(experiment_dir)

print("="*80)
print(f"n=3 MMI FINDING: FACET EVOLUTION ANALYSIS (Cold Start)")
print(f"Experiment: {experiment_name}")
print("="*80)

# Find project root (go up from analyze -> experiment_dir -> n3_mmi_finding -> results -> entropyconeRL)
project_root = os.path.abspath(os.path.join(script_dir, '../../../../..'))
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
    print(f"Loaded {len(sa_facets)} SA facets from {sa_file} (converted to binary ordering)")
else:
    print(f"WARNING: SA facets file not found: {sa_file}")
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
    print(f"Loaded {len(mmi_facets)} MMI facets from {mmi_file}")
else:
    print(f"WARNING: MMI facets file not found: {mmi_file}")
    # Use standard MMI direction for n=3: I(1:2:3) = S1 + S2 + S3 - S12 - S13 - S23 + S123 >= 0
    mmi_facets = np.array([[-1, -1, +1, -1, +1, +1, -1]])
    print("Using default MMI facet")

n_sa = len(sa_facets)
n_mmi = len(mmi_facets)
if n_sa > 0:
    all_facets = np.vstack([sa_facets, mmi_facets])
else:
    all_facets = mmi_facets

print(f"Total: {n_sa} SA facets + {n_mmi} MMI facets")

# Load experiment results
exp_file = os.path.join(experiment_dir, 'experiment_results.json')
if not os.path.exists(exp_file):
    print(f"ERROR: No experiment_results.json found in {experiment_dir}")
    print("Experiment may still be running...")
    sys.exit(1)

with open(exp_file, 'r') as f:
    exp_data = json.load(f)

# Extract positions from each stage
positions = []
stages = []
rewards = []

for entry in exp_data:
    if 'position' in entry:
        if 'stage' in entry:
            stages.append(entry['stage'])
            positions.append(entry['position'])
            rewards.append(entry.get('reward', None))
        elif 'start_points' in entry:
            stages.insert(0, -1)
            positions.insert(0, entry['start_points'])
            rewards.insert(0, None)

if len(positions) < 1:
    print(f"Insufficient data (only {len(positions)} positions)")
    sys.exit(1)

positions = np.array(positions)
n_stages = len(stages)

print(f"Found {n_stages} positions (stages: {stages})")

# Normalize facets to unit vectors (so facet value = signed distance from hyperplane)
facet_norms = np.linalg.norm(all_facets, axis=1, keepdims=True)
all_facets_normalized = all_facets / facet_norms
print(f"Normalized facets to unit vectors (SA norm={np.linalg.norm(sa_facets[0]):.4f}→1, MMI norm={np.linalg.norm(mmi_facets[0]):.4f}→1)")

# Compute facet values for each position (now represents signed distance)
facet_values = np.zeros((n_stages, len(all_facets)))
for i, pos in enumerate(positions):
    facet_values[i] = all_facets_normalized @ pos

# ========== Plot 1: Comprehensive 4-panel plot ==========
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'n=3 MMI Finding: Facet Evolution (Cold Start)\n{experiment_name}', fontsize=14, weight='bold')

# Plot 1a: All SA facets evolution
if n_sa > 0:
    for j in range(n_sa):
        color = plt.cm.Blues(0.3 + 0.7 * j / n_sa)
        axes[0, 0].plot(stages, facet_values[:, j], '-', color=color, alpha=0.7, linewidth=1)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2, label='Boundary')
axes[0, 0].set_xlabel('Stage')
axes[0, 0].set_ylabel('Facet Value (a·S)')
axes[0, 0].set_title(f'SA Facet Values Over Stages (all {n_sa} constraints)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 1b: MMI facets evolution (THE KEY PLOT)
mmi_colors = ['red', 'orange', 'purple']
for j in range(n_mmi):
    mmi_idx = n_sa + j
    axes[0, 1].plot(stages, facet_values[:, mmi_idx], 'o-', color=mmi_colors[j % len(mmi_colors)],
                   linewidth=2, markersize=8, label=f'MMI_{j}')
axes[0, 1].axhline(0, color='green', linestyle='--', linewidth=2, label='MMI boundary (goal)')
axes[0, 1].set_xlabel('Stage')
axes[0, 1].set_ylabel('MMI Value (a·S)')
axes[0, 1].set_title('MMI Constraint Values Over Stages\n(Goal: Increase toward 0)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 1c: Heatmap
im = axes[1, 0].imshow(facet_values.T, aspect='auto', cmap='RdYlGn',
                       vmin=-0.3, vmax=0.3)
if n_sa > 0:
    axes[1, 0].axhline(n_sa - 0.5, color='white', linewidth=2)
axes[1, 0].set_xlabel('Stage')
axes[1, 0].set_ylabel('Facet Index')
axes[1, 0].set_title('All Facet Values Heatmap\n(Green=satisfied, Red=violated)')
axes[1, 0].set_xticks(range(n_stages))
axes[1, 0].set_xticklabels(stages)
plt.colorbar(im, ax=axes[1, 0], label='Facet Value')
if n_sa > 0:
    axes[1, 0].text(n_stages + 0.3, n_sa/2, 'SA', fontsize=10, va='center')
    axes[1, 0].text(n_stages + 0.3, n_sa + n_mmi/2, 'MMI', fontsize=10, va='center')

# Plot 1d: Summary statistics with MMI distance
if n_sa > 0:
    sa_min = facet_values[:, :n_sa].min(axis=1)
    axes[1, 1].plot(stages, sa_min, 'o-', color='blue', linewidth=2, markersize=8, label='SA min')

mmi_min = facet_values[:, n_sa:].min(axis=1)
axes[1, 1].plot(stages, mmi_min, 'o-', color='red', linewidth=2, markersize=8, label='MMI min (violation)')
axes[1, 1].fill_between(stages, mmi_min, 0, alpha=0.3, color='red', label='Violation region')
axes[1, 1].axhline(0, color='green', linestyle='--', linewidth=2, label='Boundary')
axes[1, 1].set_xlabel('Stage')
axes[1, 1].set_ylabel('Facet Value')
axes[1, 1].set_title('Facet Summary\n(MMI distance from boundary)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'facet_evolution.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(script_dir, 'facet_evolution.png')}")
plt.close()

# ========== Plot 2: MMI Distance Tracking ==========
initial_mmi_min = mmi_min[0]
relative_progress = (mmi_min - initial_mmi_min) / np.abs(initial_mmi_min + 1e-10) * 100

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'n=3 MMI Finding: Distance from MMI Boundary (Cold Start)\n{experiment_name}', fontsize=14, weight='bold')

# Plot 2a: Absolute MMI violation over stages
for j in range(n_mmi):
    mmi_idx = n_sa + j
    abs_violation = np.abs(np.minimum(facet_values[:, mmi_idx], 0))
    axes[0, 0].plot(stages, abs_violation, 'o-', color=mmi_colors[j % len(mmi_colors)],
                   linewidth=2, markersize=8, label=f'|MMI_{j} violation|')

axes[0, 0].set_xlabel('Stage')
axes[0, 0].set_ylabel('|Violation| = |min(facet_value, 0)|')
axes[0, 0].set_title('Absolute MMI Violation\n(Goal: Decrease toward 0)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 2b: MMI min value with initial reference
ax1 = axes[0, 1]
ax1.plot(stages, mmi_min, 'o-', color='red', linewidth=2, markersize=8)
ax1.axhline(initial_mmi_min, color='gray', linestyle=':', linewidth=1, label=f'Initial: {initial_mmi_min:.6f}')
ax1.axhline(0, color='green', linestyle='--', linewidth=2, label='Boundary')
ax1.fill_between(stages, mmi_min, initial_mmi_min, alpha=0.3, color='green' if mmi_min[-1] > initial_mmi_min else 'red')
ax1.set_xlabel('Stage')
ax1.set_ylabel('MMI min value')
ax1.set_title('MMI Min Value Evolution\n(Distance from boundary)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2c: Relative progress (% change from initial)
axes[1, 0].bar(stages, relative_progress, color=['green' if p > 0 else 'red' for p in relative_progress])
axes[1, 0].axhline(0, color='gray', linestyle='-', linewidth=1)
axes[1, 0].set_xlabel('Stage')
axes[1, 0].set_ylabel('% Change from Initial')
axes[1, 0].set_title('Relative Progress Toward MMI Boundary\n(Positive = improving)')
axes[1, 0].grid(True, alpha=0.3, axis='y')

for i, (s, p) in enumerate(zip(stages, relative_progress)):
    axes[1, 0].annotate(f'{p:+.1f}%', (s, p), ha='center',
                       va='bottom' if p > 0 else 'top', fontsize=9)

# Plot 2d: Reward vs MMI violation
valid_rewards = [(stages[i], rewards[i], mmi_min[i]) for i in range(len(stages)) if rewards[i] is not None]
if valid_rewards:
    r_stages, r_rewards, r_mmi = zip(*valid_rewards)
    ax2 = axes[1, 1]
    scatter = ax2.scatter(r_mmi, r_rewards, c=r_stages, cmap='viridis', s=100, edgecolors='black')
    ax2.set_xlabel('MMI min value (violation)')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward vs MMI Violation\n(Color = stage)')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='green', linestyle='--', linewidth=1, label='MMI boundary')
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='Stage')

    for s, r, m in valid_rewards:
        ax2.annotate(f'S{s}', (m, r), xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'mmi_distance_tracking.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(script_dir, 'mmi_distance_tracking.png')}")
plt.close()

# Print numerical summary
print(f"\nFacet Value Summary:")
if n_sa > 0:
    print(f"{'Stage':>6} {'SA min':>10} {'MMI min':>10} {'MMI change':>12} {'% Progress':>12} {'Reward':>10}")
    print(f"{'-'*70}")
    for i, s in enumerate(stages):
        r_str = f"{rewards[i]:.6f}" if rewards[i] is not None else "N/A"
        change = mmi_min[i] - initial_mmi_min
        pct = relative_progress[i]
        print(f"{s:>6} {sa_min[i]:>10.6f} {mmi_min[i]:>10.6f} {change:>+12.6f} {pct:>+11.1f}% {r_str:>10}")
else:
    print(f"{'Stage':>6} {'MMI min':>10} {'MMI change':>12} {'% Progress':>12} {'Reward':>10}")
    print(f"{'-'*60}")
    for i, s in enumerate(stages):
        r_str = f"{rewards[i]:.6f}" if rewards[i] is not None else "N/A"
        change = mmi_min[i] - initial_mmi_min
        pct = relative_progress[i]
        print(f"{s:>6} {mmi_min[i]:>10.6f} {change:>+12.6f} {pct:>+11.1f}% {r_str:>10}")

print(f"\nProgress Analysis:")
print(f"  Initial MMI violation: {initial_mmi_min:.6f}")
print(f"  Final MMI violation: {mmi_min[-1]:.6f}")
print(f"  Total change: {mmi_min[-1] - initial_mmi_min:+.6f}")
print(f"  Total progress: {relative_progress[-1]:+.1f}%")

if mmi_min[-1] > initial_mmi_min:
    print(f"  → Moving TOWARD MMI boundary (good!)")
elif mmi_min[-1] < initial_mmi_min:
    print(f"  → Moving AWAY from MMI boundary (need to investigate)")
else:
    print(f"  → No net movement")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
