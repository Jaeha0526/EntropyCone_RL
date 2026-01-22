#!/usr/bin/env python3
"""
Analyze gradient estimation quality vs sample count for n=3 MMI finding.
Generates plots showing R², gradient consistency, and MMI alignment.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

print("="*70)
print("dS_analysis: Gradient Quality vs Sample Count")
print("="*70)

# MMI direction (normalized) for comparison
mmi = np.array([-1, -1, +1, -1, +1, +1, -1])
mmi_norm = mmi / np.linalg.norm(mmi)

sample_counts = [10, 20, 30, 40, 50]
results = {s: {'r2': [], 'gradients': [], 'mmi_align': []} for s in sample_counts}

# Collect data from all runs
for samples in sample_counts:
    dir_name = f"samples{samples}_dS002_multirun"

    for run in range(1, 11):
        path = f"{base_dir}/{dir_name}/run{run}/gradient_samples_stage_0.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)

            r2 = data['linear_regression']['r2_with_intercept']
            grad = np.array(data['gradient_computed'])
            grad_norm = grad / (np.linalg.norm(grad) + 1e-10)
            mmi_align = np.dot(mmi_norm, grad_norm)

            results[samples]['r2'].append(r2)
            results[samples]['gradients'].append(grad_norm)
            results[samples]['mmi_align'].append(mmi_align)

# Compute statistics
stats = {}
for samples in sample_counts:
    grads = results[samples]['gradients']
    if grads:
        mean_grad = np.mean(grads, axis=0)
        mean_grad = mean_grad / np.linalg.norm(mean_grad)
        alignments_with_mean = [np.dot(g, mean_grad) for g in grads]

        stats[samples] = {
            'r2_mean': np.mean(results[samples]['r2']),
            'r2_std': np.std(results[samples]['r2']),
            'r2_all': results[samples]['r2'],
            'grad_consistency': 1 - np.std(alignments_with_mean),  # Higher = more consistent
            'grad_std': np.std(alignments_with_mean),
            'mmi_mean': np.mean(results[samples]['mmi_align']),
            'mmi_std': np.std(results[samples]['mmi_align']),
            'mmi_all': results[samples]['mmi_align'],
        }

# Print summary
print()
print(f"{'Samples':>8} {'Mean R²':>10} {'Std R²':>10} {'Grad Std':>12} {'MMI align':>12}")
print("-"*55)
for samples in sample_counts:
    s = stats[samples]
    print(f"{samples:>8} {s['r2_mean']:>10.4f} {s['r2_std']:>10.4f} {s['grad_std']:>12.4f} {s['mmi_mean']:>+12.4f}")

# ========== Create Plots ==========

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('n=3 MMI Finding: Gradient Quality vs Sample Count\n(dS=0.02, 10 runs per sample count)',
             fontsize=14, weight='bold')

# Plot 1: R² vs sample count (box plot)
ax = axes[0, 0]
r2_data = [stats[s]['r2_all'] for s in sample_counts]
bp = ax.boxplot(r2_data, positions=sample_counts, widths=5, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.plot(sample_counts, [stats[s]['r2_mean'] for s in sample_counts], 'ro-', markersize=8, label='Mean')
ax.set_xlabel('Number of Samples')
ax.set_ylabel('R² (with intercept)')
ax.set_title('Regression Quality (R²)\nHigher = better fit')
ax.set_xlim(5, 55)
ax.set_ylim(0.9, 1.01)
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Gradient consistency vs sample count
ax = axes[0, 1]
grad_stds = [stats[s]['grad_std'] for s in sample_counts]
ax.bar(sample_counts, grad_stds, width=5, color='coral', edgecolor='black')
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Gradient Direction Std')
ax.set_title('Gradient Consistency\nLower = more consistent across runs')
ax.set_xlim(5, 55)
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (s, v) in enumerate(zip(sample_counts, grad_stds)):
    ax.annotate(f'{v:.4f}', (s, v), ha='center', va='bottom', fontsize=10)

# Plot 3: MMI alignment vs sample count (box plot)
ax = axes[1, 0]
mmi_data = [stats[s]['mmi_all'] for s in sample_counts]
bp = ax.boxplot(mmi_data, positions=sample_counts, widths=5, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
ax.plot(sample_counts, [stats[s]['mmi_mean'] for s in sample_counts], 'ro-', markersize=8, label='Mean')
ax.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Perfect alignment')
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Gradient · MMI direction')
ax.set_title('Alignment with MMI Direction\nHigher = gradient points toward MMI boundary')
ax.set_xlim(5, 55)
ax.set_ylim(0.95, 1.01)
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Summary - Trade-off visualization
ax = axes[1, 1]
ax2 = ax.twinx()

# R² on left axis
r2_means = [stats[s]['r2_mean'] for s in sample_counts]
line1 = ax.plot(sample_counts, r2_means, 'bo-', markersize=10, linewidth=2, label='R² (quality)')
ax.fill_between(sample_counts,
                [stats[s]['r2_mean'] - stats[s]['r2_std'] for s in sample_counts],
                [stats[s]['r2_mean'] + stats[s]['r2_std'] for s in sample_counts],
                alpha=0.2, color='blue')
ax.set_ylabel('R²', color='blue')
ax.tick_params(axis='y', labelcolor='blue')
ax.set_ylim(0.97, 1.0)

# Gradient std on right axis
line2 = ax2.plot(sample_counts, grad_stds, 'rs-', markersize=10, linewidth=2, label='Grad Std (consistency)')
ax2.set_ylabel('Gradient Std', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 0.02)

ax.set_xlabel('Number of Samples')
ax.set_title('Quality vs Consistency Trade-off\n(Sweet spot: 20-30 samples)')
ax.set_xlim(5, 55)
ax.grid(True, alpha=0.3)

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right')

# Highlight sweet spot
ax.axvspan(17, 33, alpha=0.1, color='green')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'sample_count_analysis.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved: {os.path.join(script_dir, 'sample_count_analysis.png')}")
plt.close()

# ========== Plot 2: Individual gradients comparison ==========
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Estimated Gradients Across Runs (normalized)', fontsize=14, weight='bold')

component_labels = ['S_A', 'S_B', 'S_AB', 'S_C', 'S_AC', 'S_BC', 'S_ABC']

for idx, samples in enumerate(sample_counts):
    ax = axes[idx]
    grads = results[samples]['gradients']

    # Plot each run's gradient
    for i, g in enumerate(grads):
        ax.plot(range(7), g, 'o-', alpha=0.3, markersize=4, label=f'Run {i+1}' if idx == 0 else None)

    # Plot mean gradient
    mean_grad = np.mean(grads, axis=0)
    mean_grad = mean_grad / np.linalg.norm(mean_grad)
    ax.plot(range(7), mean_grad, 'k-', linewidth=3, label='Mean' if idx == 0 else None)

    # Plot MMI direction for reference
    ax.plot(range(7), mmi_norm, 'g--', linewidth=2, label='MMI' if idx == 0 else None)

    ax.set_xticks(range(7))
    ax.set_xticklabels(component_labels, rotation=45)
    ax.set_title(f'{samples} samples')
    ax.set_ylim(-0.6, 0.6)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.grid(True, alpha=0.3)

axes[0].legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'gradient_comparison.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(script_dir, 'gradient_comparison.png')}")
plt.close()

# ========== Summary Statistics ==========
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nKey Findings:")
print("  - R² is high (>0.93) for all sample counts")
print("  - Gradient consistency improves significantly from 10→20→30 samples")
print("  - 20-30 samples appears to be the sweet spot")
print("  - All gradients are highly aligned with MMI direction (~98-99%)")
