#!/usr/bin/env python3
"""
Generate gradient quality analysis plots for ALL stages.
Creates one plot per stage in the per_stage_analysis folder.
Also creates a summary plot comparing all stages.

Color-codes each sample point by whether it's inside or outside the HEC cone.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Get experiment directory from script location
script_dir = os.path.dirname(os.path.abspath(__file__))
analyze_dir = os.path.dirname(script_dir)
experiment_dir = os.path.dirname(analyze_dir)
experiment_name = os.path.basename(experiment_dir)

# Find project root and load facets
project_root = os.path.abspath(os.path.join(script_dir, '../../../../../..'))
facets_dir = os.path.join(project_root, 'extremal_rays/n3/MMI_finding')

# Load SA facets
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

# Load MMI facets
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
    mmi_facets = np.array([[-1, -1, +1, -1, +1, +1, -1]])

# Combine all facets for cone membership check
if len(sa_facets) > 0:
    all_hec_facets = np.vstack([sa_facets, mmi_facets])
else:
    all_hec_facets = mmi_facets

n_sa = len(sa_facets)
n_mmi = len(mmi_facets)

def check_inside_cone(S_vector, tolerance=-1e-8):
    """Check if S_vector is inside the HEC cone (all facet values >= 0)."""
    facet_values = all_hec_facets @ np.array(S_vector)
    return np.all(facet_values >= tolerance)

def get_violated_facets(S_vector, tolerance=-1e-8):
    """Return which facets are violated (SA count, MMI count)."""
    facet_values = all_hec_facets @ np.array(S_vector)
    sa_violated = np.sum(facet_values[:n_sa] < tolerance) if n_sa > 0 else 0
    mmi_violated = np.sum(facet_values[n_sa:] < tolerance)
    return sa_violated, mmi_violated

print("="*80)
print(f"PER-STAGE GRADIENT QUALITY ANALYSIS (Cold Start)")
print(f"Experiment: {experiment_name}")
print("="*80)

# Find gradient sample files
gradient_files = sorted(glob.glob(os.path.join(experiment_dir, 'gradient_samples_stage_*.json')))

if not gradient_files:
    print("No gradient_samples_stage_*.json files found")
    exit(1)

# Sort by stage number
def get_stage_num(f):
    return int(f.split('_stage_')[-1].split('.')[0])

gradient_files = sorted(gradient_files, key=get_stage_num)
print(f"Found {len(gradient_files)} gradient sample files")

# Correct MMI direction for n=3
mmi_raw = np.array([-1, -1, +1, -1, +1, +1, -1])
mmi_norm = mmi_raw / np.linalg.norm(mmi_raw)

# Store results for summary
all_results = []

# Process each stage
for gradient_file in gradient_files:
    stage_num = get_stage_num(gradient_file)

    with open(gradient_file, 'r') as f:
        data = json.load(f)

    # Extract data
    samples = data['samples']
    base_reward = data['base_reward']
    n_samples = data['n_samples']
    dS = data['dS']
    gradient_computed = np.array(data['gradient_computed'])

    r2_from_file = data['linear_regression']['r2_with_intercept']
    intercept = data['linear_regression']['intercept']

    directions = np.array([s['direction'] for s in samples])
    rewards = np.array([s['reward'] for s in samples])
    reward_changes = np.array([s['reward_change'] for s in samples])
    S_perturbed_all = np.array([s['S_perturbed'] for s in samples])

    # Check cone membership for each sample
    inside_cone = np.array([check_inside_cone(S) for S in S_perturbed_all])
    n_inside = np.sum(inside_cone)
    n_outside = len(inside_cone) - n_inside

    # Get violation details for outside points
    violation_details = [get_violated_facets(S) for S in S_perturbed_all]
    sa_violations = np.array([v[0] for v in violation_details])
    mmi_violations = np.array([v[1] for v in violation_details])

    n_positive = np.sum(reward_changes > 0)

    # Compute -log(1-r) transformation
    neg_log_scores = -np.log(1 - rewards + 1e-10)
    base_neg_log = -np.log(1 - base_reward + 1e-10)
    neg_log_changes = neg_log_scores - base_neg_log

    # Fit linear model with intercept
    X_with_intercept = np.column_stack([directions, np.ones(len(directions))])
    y = neg_log_changes

    params = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    y_pred = X_with_intercept @ params

    # Compute metrics
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    correlation = np.corrcoef(y_pred, y)[0, 1] if len(y) > 1 else 0
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    signal_variance = np.var(y_pred)
    noise_variance = np.var(y - y_pred)
    snr = signal_variance / noise_variance if noise_variance > 0 else float('inf')

    grad_norm = gradient_computed / (np.linalg.norm(gradient_computed) + 1e-10)
    cosine_mmi = np.dot(grad_norm, mmi_norm)

    # Store results
    all_results.append({
        'stage': stage_num,
        'r2': r2,
        'r2_file': r2_from_file,
        'snr': snr,
        'correlation': correlation,
        'rmse': rmse,
        'n_positive': n_positive,
        'n_samples': n_samples,
        'cosine_mmi': cosine_mmi,
        'base_reward': base_reward,
        'mean_reward_change': np.mean(reward_changes),
        'gradient': gradient_computed,
        'y': y,
        'y_pred': y_pred,
        'reward_changes': reward_changes,
        'inside_cone': inside_cone,
        'n_inside': n_inside,
        'n_outside': n_outside,
        'mmi_violations': mmi_violations
    })

    # Create per-stage plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Stage {stage_num}: Gradient Quality Analysis\n{experiment_name}', fontsize=12, weight='bold')

    # Plot 1: Predicted vs Actual (color-coded by cone membership)
    ax1 = axes[0, 0]
    # Color: green = inside HEC cone, red = outside
    colors = np.where(inside_cone, 'green', 'red')
    ax1.scatter(y_pred[inside_cone], y[inside_cone], alpha=0.6, s=30, c='green', label=f'Inside ({n_inside})', edgecolors='darkgreen', linewidths=0.5)
    ax1.scatter(y_pred[~inside_cone], y[~inside_cone], alpha=0.6, s=30, c='red', label=f'Outside ({n_outside})', edgecolors='darkred', linewidths=0.5)
    lim_min = min(y_pred.min(), y.min())
    lim_max = max(y_pred.max(), y.max())
    margin = (lim_max - lim_min) * 0.1
    ax1.plot([lim_min-margin, lim_max+margin], [lim_min-margin, lim_max+margin], 'k--', linewidth=2, label='y=x')
    ax1.set_xlabel('Predicted -log(1-r) Change')
    ax1.set_ylabel('Actual -log(1-r) Change')
    ax1.set_title(f'R²={r2:.4f}, Corr={correlation:.4f}\n(Green=inside HEC, Red=outside)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=8)

    # Plot 2: Histogram of reward changes (stacked by cone membership)
    ax2 = axes[0, 1]
    # Separate histograms for inside/outside
    ax2.hist(reward_changes[inside_cone], bins=30, edgecolor='darkgreen', alpha=0.7, color='green', label=f'Inside ({n_inside})')
    ax2.hist(reward_changes[~inside_cone], bins=30, edgecolor='darkred', alpha=0.7, color='red', label=f'Outside ({n_outside})')
    ax2.axvline(np.mean(reward_changes), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(reward_changes):.6f}')
    ax2.axvline(0, color='black', linestyle=':', linewidth=2)
    ax2.set_xlabel('Reward Change')
    ax2.set_ylabel('Frequency')
    pct_pos = 100 * n_positive / n_samples
    pct_inside = 100 * n_inside / n_samples
    ax2.set_title(f'{n_samples} samples, {pct_pos:.0f}% positive, {pct_inside:.0f}% inside cone')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals (color-coded by cone membership)
    ax3 = axes[1, 0]
    residuals = y - y_pred
    ax3.scatter(y_pred[inside_cone], residuals[inside_cone], alpha=0.6, s=30, c='green', label=f'Inside ({n_inside})', edgecolors='darkgreen', linewidths=0.5)
    ax3.scatter(y_pred[~inside_cone], residuals[~inside_cone], alpha=0.6, s=30, c='red', label=f'Outside ({n_outside})', edgecolors='darkred', linewidths=0.5)
    ax3.axhline(0, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted -log(1-r) Change')
    ax3.set_ylabel('Residuals')
    ax3.set_title(f'RMSE={rmse:.4f}, SNR={snr:.1f}')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Gradient components vs MMI
    ax4 = axes[1, 1]
    x_pos = np.arange(7)
    labels = ['S(1)', 'S(2)', 'S(12)', 'S(3)', 'S(13)', 'S(23)', 'S(123)']
    width = 0.35

    grad_normalized = gradient_computed / (np.linalg.norm(gradient_computed) + 1e-10)

    ax4.bar(x_pos - width/2, grad_normalized, width, label='Gradient', alpha=0.8, color='steelblue')
    ax4.bar(x_pos + width/2, mmi_norm, width, label='MMI Direction', alpha=0.8, color='orange')
    ax4.set_xlabel('S-vector Component')
    ax4.set_ylabel('Normalized Value')
    ax4.set_title(f'Cosine(grad, MMI): {cosine_mmi:.4f} ({abs(cosine_mmi)*100:.1f}%)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    output_file = os.path.join(script_dir, f'gradient_quality_stage_{stage_num:02d}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Stage {stage_num:2d}: R²={r2:.4f}, SNR={snr:5.1f}, {pct_pos:5.1f}% pos, {pct_inside:5.1f}% inside cone, Cos(MMI)={cosine_mmi:+.4f}")

# Sort results by stage
all_results = sorted(all_results, key=lambda x: x['stage'])

# ========== Create Summary Plot ==========
n_stages = len(all_results)
stages = [r['stage'] for r in all_results]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle(f'Gradient Quality Summary Across All Stages\n{experiment_name}', fontsize=14, weight='bold')

# Plot 1: R² over stages
ax = axes[0, 0]
ax.plot(stages, [r['r2'] for r in all_results], 'o-', color='blue', linewidth=2, markersize=8)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='R²=0.5')
ax.set_xlabel('Stage')
ax.set_ylabel('R² (with intercept)')
ax.set_title('Regression Quality Over Stages')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

# Plot 2: SNR over stages
ax = axes[0, 1]
ax.plot(stages, [min(r['snr'], 10) for r in all_results], 'o-', color='green', linewidth=2, markersize=8)
ax.axhline(1, color='gray', linestyle='--', alpha=0.5, label='SNR=1')
ax.set_xlabel('Stage')
ax.set_ylabel('SNR (capped at 10)')
ax.set_title('Signal-to-Noise Ratio Over Stages')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: % Positive samples
ax = axes[0, 2]
pct_positive = [100 * r['n_positive'] / r['n_samples'] for r in all_results]
colors = ['green' if p > 50 else 'red' for p in pct_positive]
ax.bar(stages, pct_positive, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(50, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Stage')
ax.set_ylabel('% Positive Samples')
ax.set_title('Fraction of Samples with Reward Increase')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Cosine with MMI direction
ax = axes[1, 0]
cosines = [r['cosine_mmi'] for r in all_results]
colors = ['green' if c > 0 else 'red' for c in cosines]
ax.bar(stages, cosines, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0, color='gray', linestyle='-', linewidth=1)
ax.set_xlabel('Stage')
ax.set_ylabel('Cosine(gradient, MMI)')
ax.set_title('Gradient Alignment with MMI Direction')
ax.grid(True, alpha=0.3, axis='y')

# Plot 5: Base reward over stages
ax = axes[1, 1]
ax.plot(stages, [r['base_reward'] for r in all_results], 'o-', color='purple', linewidth=2, markersize=8)
ax.set_xlabel('Stage')
ax.set_ylabel('Base Reward')
ax.set_title('Base Reward Evolution')
ax.grid(True, alpha=0.3)

# Plot 6: Mean reward change over stages
ax = axes[1, 2]
mean_changes = [r['mean_reward_change'] for r in all_results]
colors = ['green' if c > 0 else 'red' for c in mean_changes]
ax.bar(stages, [c * 1000 for c in mean_changes], color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0, color='gray', linestyle='-', linewidth=1)
ax.set_xlabel('Stage')
ax.set_ylabel('Mean Reward Change (×10³)')
ax.set_title('Mean Reward Change per Stage')
ax.grid(True, alpha=0.3, axis='y')

# Plot 7: % Inside HEC Cone over stages (NEW)
ax = axes[0, 3]
pct_inside_list = [100 * r['n_inside'] / r['n_samples'] for r in all_results]
colors = ['green' if p > 50 else 'orange' if p > 10 else 'red' for p in pct_inside_list]
ax.bar(stages, pct_inside_list, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(50, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Stage')
ax.set_ylabel('% Inside HEC Cone')
ax.set_title('Sample Points Inside HEC Cone')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 100)

# Plot 8: Mean MMI violations for outside points
ax = axes[1, 3]
mean_mmi_violations = []
for r in all_results:
    outside_mask = ~r['inside_cone']
    if np.any(outside_mask):
        mean_mmi_violations.append(np.mean(r['mmi_violations'][outside_mask]))
    else:
        mean_mmi_violations.append(0)
ax.bar(stages, mean_mmi_violations, color='red', alpha=0.7, edgecolor='darkred')
ax.set_xlabel('Stage')
ax.set_ylabel('Mean # MMI Violated')
ax.set_title('MMI Violations (Outside Points)')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
summary_file = os.path.join(script_dir, 'summary_all_stages.png')
plt.savefig(summary_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved summary: {summary_file}")

# Print summary table
print("\n" + "="*100)
print("SUMMARY TABLE")
print("="*100)
print(f"{'Stage':>6} {'R²':>8} {'SNR':>8} {'%Pos':>8} {'%InCone':>9} {'Cos(MMI)':>10} {'BaseReward':>12} {'MeanΔr':>12}")
print("-"*90)
for r in all_results:
    pct = 100 * r['n_positive'] / r['n_samples']
    pct_in = 100 * r['n_inside'] / r['n_samples']
    print(f"{r['stage']:>6} {r['r2']:>8.4f} {r['snr']:>8.1f} {pct:>7.1f}% {pct_in:>8.1f}% {r['cosine_mmi']:>+10.4f} {r['base_reward']:>12.6f} {r['mean_reward_change']:>+12.6f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print(f"Generated {len(gradient_files)} per-stage plots + 1 summary plot")
print("="*80)
