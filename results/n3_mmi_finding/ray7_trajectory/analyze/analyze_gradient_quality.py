#!/usr/bin/env python3
"""
Analyze gradient estimation quality from RL v2 gradient samples (Cold Start).
Generates quality metrics: R², correlation, SNR, RMSE.
Uses same approach as n=5 experiment: intercept + -log(1-r) transformation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob

# Get experiment directory from script location
script_dir = os.path.dirname(os.path.abspath(__file__))
experiment_dir = os.path.dirname(script_dir)
experiment_name = os.path.basename(experiment_dir)

print("="*80)
print(f"GRADIENT QUALITY ANALYSIS (Cold Start) - {experiment_name}")
print("="*80)

# Find gradient sample files
gradient_files = sorted(glob.glob(os.path.join(experiment_dir, 'gradient_samples_stage_*.json')))

if not gradient_files:
    print("No gradient_samples_stage_*.json files found")
    print("Experiment may still be running...")
    exit(1)

print(f"Found {len(gradient_files)} gradient sample files")

# Correct MMI direction for n=3
# S-vector order: [S(1), S(2), S(12), S(3), S(13), S(23), S(123)]
# MMI: S(12) + S(13) + S(23) >= S(1) + S(2) + S(3) + S(123)
mmi_raw = np.array([-1, -1, +1, -1, +1, +1, -1])
mmi_norm = mmi_raw / np.linalg.norm(mmi_raw)

# Analyze each stage
all_r2 = []
all_snr = []
all_stages = []
all_cosine_mmi = []

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'n=3 MMI Finding: Cold Start Gradient Quality\n{experiment_name}', fontsize=12, weight='bold')

# Use first stage for detailed analysis (Stage 0)
first_file = gradient_files[0]
stage_num = int(first_file.split('_stage_')[-1].split('.')[0])

with open(first_file, 'r') as f:
    data = json.load(f)

# Extract data from samples
samples = data['samples']
base_reward = data['base_reward']
n_samples = data['n_samples']
dS = data['dS']
gradient_computed = np.array(data['gradient_computed'])

# Get R² from file
r2_from_file = data['linear_regression']['r2_with_intercept']
intercept = data['linear_regression']['intercept']

directions = np.array([s['direction'] for s in samples])
rewards = np.array([s['reward'] for s in samples])
reward_changes = np.array([s['reward_change'] for s in samples])

# Count positive vs negative
n_positive = np.sum(reward_changes > 0)
n_negative = np.sum(reward_changes < 0)

# Compute -log(1-r) transformation (like n=5)
neg_log_scores = -np.log(1 - rewards + 1e-10)
base_neg_log = -np.log(1 - base_reward + 1e-10)
neg_log_changes = neg_log_scores - base_neg_log

# Fit linear model WITH INTERCEPT
X_with_intercept = np.column_stack([directions, np.ones(len(directions))])
y = neg_log_changes

params = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
gradient_estimate = params[:-1]
intercept_computed = params[-1]
y_pred = X_with_intercept @ params

# Compute R²
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

# Compute correlation
correlation = np.corrcoef(y_pred, y)[0, 1] if len(y) > 1 else 0

# Compute RMSE
rmse = np.sqrt(np.mean((y - y_pred) ** 2))

# Compute SNR (signal-to-noise ratio)
signal_variance = np.var(y_pred)
noise_variance = np.var(y - y_pred)
snr = signal_variance / noise_variance if noise_variance > 0 else float('inf')

# Cosine similarity with MMI direction
grad_norm = gradient_computed / (np.linalg.norm(gradient_computed) + 1e-10)
cosine_mmi = np.dot(grad_norm, mmi_norm)

print(f"\nStage {stage_num} Analysis ({n_samples} samples, dS={dS}):")
print(f"  Base reward: {base_reward:.6f}")
print(f"  R² (from file): {r2_from_file:.4f}")
print(f"  R² (-log transform): {r2:.4f}")
print(f"  Correlation: {correlation:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  SNR: {snr:.1f}")
print(f"  Positive samples: {n_positive}/{n_samples} ({100*n_positive/n_samples:.1f}%)")
print(f"  Mean reward change: {np.mean(reward_changes):.6f}")
print(f"  Cosine(gradient, MMI): {cosine_mmi:.4f} ({abs(cosine_mmi)*100:.1f}%)")
print(f"  Gradient: [{', '.join([f'{x:.4f}' for x in gradient_computed])}]")

# Plot 1: Predicted vs Actual (-log transform)
ax1 = axes[0, 0]
ax1.scatter(y_pred, y, alpha=0.5, s=30, c='blue')
lim_min = min(y_pred.min(), y.min())
lim_max = max(y_pred.max(), y.max())
margin = (lim_max - lim_min) * 0.1
ax1.plot([lim_min-margin, lim_max+margin], [lim_min-margin, lim_max+margin], 'r--', linewidth=2, label='y=x (perfect fit)')
ax1.set_xlabel('Predicted -log(1-r) Change')
ax1.set_ylabel('Actual -log(1-r) Change')
ax1.set_title(f'Stage {stage_num}: R²={r2:.4f}, Corr={correlation:.4f}')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Histogram of reward changes
ax2 = axes[0, 1]
ax2.hist(reward_changes, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax2.axvline(np.mean(reward_changes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(reward_changes):.6f}')
ax2.axvline(0, color='black', linestyle=':', linewidth=2)
ax2.set_xlabel('Reward Change')
ax2.set_ylabel('Frequency')
pct_pos = 100 * n_positive / n_samples
ax2.set_title(f'Cold Start Distribution ({n_samples} samples)\n{pct_pos:.0f}% positive')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals
ax3 = axes[1, 0]
residuals = y - y_pred
ax3.scatter(y_pred, residuals, alpha=0.5, s=30, c='purple')
ax3.axhline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Predicted -log(1-r) Change')
ax3.set_ylabel('Residuals')
ax3.set_title(f'Residuals\nRMSE={rmse:.4f}, SNR={snr:.1f}')
ax3.grid(True, alpha=0.3)

# Plot 4: Gradient components vs MMI
ax4 = axes[1, 1]
x_pos = np.arange(7)
labels = ['S(1)', 'S(2)', 'S(12)', 'S(3)', 'S(13)', 'S(23)', 'S(123)']
width = 0.35

# Normalize for comparison
grad_normalized = gradient_computed / (np.linalg.norm(gradient_computed) + 1e-10)

ax4.bar(x_pos - width/2, grad_normalized, width, label='Estimated Gradient', alpha=0.8, color='steelblue')
ax4.bar(x_pos + width/2, mmi_norm, width, label='Ideal MMI Direction', alpha=0.8, color='orange')
ax4.set_xlabel('S-vector Component')
ax4.set_ylabel('Normalized Value')
ax4.set_title(f'Gradient vs MMI Direction\nCosine Similarity: {cosine_mmi:.4f} ({abs(cosine_mmi)*100:.1f}%)')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(labels, rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'gradient_quality_analysis.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved: {os.path.join(script_dir, 'gradient_quality_analysis.png')}")
plt.close()

# Analyze all stages
if len(gradient_files) > 1:
    print(f"\nAll Stages Summary:")
    print(f"{'Stage':>6} {'Samples':>8} {'R²':>8} {'SNR':>8} {'%Positive':>10} {'Cosine(MMI)':>12}")
    print("-" * 60)

    for gf in gradient_files:
        stage = int(gf.split('_stage_')[-1].split('.')[0])
        with open(gf, 'r') as f:
            data = json.load(f)

        samples = data['samples']
        base_rew = data['base_reward']
        n_samp = data['n_samples']
        grad = np.array(data['gradient_computed'])

        dirs = np.array([s['direction'] for s in samples])
        rews = np.array([s['reward'] for s in samples])
        rew_changes = np.array([s['reward_change'] for s in samples])

        # Count positive
        n_pos = np.sum(rew_changes > 0)

        # Use -log(1-r) transformation with intercept
        neg_log = -np.log(1 - rews + 1e-10)
        base_nl = -np.log(1 - base_rew + 1e-10)
        nl_changes = neg_log - base_nl

        X_int = np.column_stack([dirs, np.ones(len(dirs))])
        params_s = np.linalg.lstsq(X_int, nl_changes, rcond=None)[0]
        pred = X_int @ params_s

        ss_r = np.sum((nl_changes - pred) ** 2)
        ss_t = np.sum((nl_changes - np.mean(nl_changes)) ** 2)
        r2_s = 1 - (ss_r / ss_t) if ss_t > 0 else 0

        sig_var = np.var(pred)
        noise_var = np.var(nl_changes - pred)
        snr_s = sig_var / noise_var if noise_var > 0 else float('inf')

        # Cosine with MMI
        grad_n = grad / (np.linalg.norm(grad) + 1e-10)
        cos_mmi = np.dot(grad_n, mmi_norm)

        pct_pos = 100 * n_pos / n_samp
        print(f"{stage:>6} {n_samp:>8} {r2_s:>8.4f} {snr_s:>8.1f} {pct_pos:>9.1f}% {cos_mmi:>12.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
