#!/usr/bin/env python3
"""
Plot R² and Cosine similarity vs sample size comparing dS=0.01 and dS=0.001.
With error bars from multiple runs.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Correct MMI formula for n=3
mmi_raw = np.array([-1, -1, +1, -1, +1, +1, -1])
mmi_norm = mmi_raw / np.linalg.norm(mmi_raw)

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

def get_gradient_quality(path):
    """Extract R² and Cos(MMI) from gradient samples file."""
    data = json.load(open(path))
    r2 = data.get('linear_regression', {}).get('r2_with_intercept', 0)
    grad = np.array(data.get('gradient_computed', [0]*7))
    if np.linalg.norm(grad) > 0:
        grad_norm = grad / np.linalg.norm(grad)
        cos_mmi = np.dot(grad_norm, mmi_norm)
    else:
        cos_mmi = 0
    return r2, cos_mmi

def collect_multirun_data(pattern_func, samples_list):
    """Collect data from multirun experiments."""
    results = {}
    for samples in samples_list:
        results[samples] = []
        for run in range(1, 6):
            path = pattern_func(samples, run)
            if os.path.exists(path):
                r2, cos = get_gradient_quality(path)
                results[samples].append((r2, cos))
    return results

def compute_stats(results):
    """Compute mean and std from results."""
    stats = {}
    for samples, runs in results.items():
        if runs:
            r2_vals = [r[0] for r in runs]
            cos_vals = [r[1] for r in runs]
            stats[samples] = {
                'r2_mean': np.mean(r2_vals),
                'r2_std': np.std(r2_vals),
                'cos_mean': np.mean(cos_vals),
                'cos_std': np.std(cos_vals),
                'n_runs': len(runs)
            }
    return stats

# Collect dS=0.01 multirun data
dS001_small = collect_multirun_data(
    lambda s, r: os.path.join(base_dir, f'samples{s}_multirun/run{r}/gradient_samples_stage_0.json'),
    [10, 20, 30]
)
dS001_large = collect_multirun_data(
    lambda s, r: os.path.join(base_dir, f'samples{s}_multirun/run{r}/gradient_samples_stage_0.json'),
    [40, 50, 60, 70]
)
dS001_all = {**dS001_small, **dS001_large}
dS001_stats = compute_stats(dS001_all)

# Collect dS=0.001 multirun data
dS0001_all = collect_multirun_data(
    lambda s, r: os.path.join(base_dir, f'dS0001_samples{s}_multirun/run{r}/gradient_samples_stage_0.json'),
    [10, 20, 30, 40, 50, 60, 70]
)
dS0001_stats = compute_stats(dS0001_all)

if not dS001_stats and not dS0001_stats:
    print("No results found yet. Experiments may still be running.")
    exit(0)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot dS=0.01
if dS001_stats:
    samples = sorted(dS001_stats.keys())
    r2_mean = [dS001_stats[s]['r2_mean'] for s in samples]
    r2_std = [dS001_stats[s]['r2_std'] for s in samples]
    cos_mean = [dS001_stats[s]['cos_mean'] for s in samples]
    cos_std = [dS001_stats[s]['cos_std'] for s in samples]

    ax1.errorbar(samples, r2_mean, yerr=r2_std,
                 fmt='o-', color='tab:blue', linewidth=2, markersize=8,
                 capsize=4, capthick=1.5, label='dS=0.01')
    ax2.errorbar(samples, cos_mean, yerr=cos_std,
                 fmt='o-', color='tab:blue', linewidth=2, markersize=8,
                 capsize=4, capthick=1.5, label='dS=0.01')

# Plot dS=0.001
if dS0001_stats:
    samples = sorted(dS0001_stats.keys())
    r2_mean = [dS0001_stats[s]['r2_mean'] for s in samples]
    r2_std = [dS0001_stats[s]['r2_std'] for s in samples]
    cos_mean = [dS0001_stats[s]['cos_mean'] for s in samples]
    cos_std = [dS0001_stats[s]['cos_std'] for s in samples]

    ax1.errorbar(samples, r2_mean, yerr=r2_std,
                 fmt='s--', color='tab:red', linewidth=2, markersize=8,
                 capsize=4, capthick=1.5, label='dS=0.001')
    ax2.errorbar(samples, cos_mean, yerr=cos_std,
                 fmt='s--', color='tab:red', linewidth=2, markersize=8,
                 capsize=4, capthick=1.5, label='dS=0.001')

# Configure axes
ax1.axhline(y=0.95, color='gray', linestyle=':', linewidth=1.5, label='R²=0.95')
ax1.set_xlabel('Number of gradient samples', fontsize=12)
ax1.set_ylabel('R² (gradient reliability)', fontsize=12)
ax1.set_title('Gradient Reliability (R²) vs Sample Size', fontsize=13)
ax1.set_ylim(0, 1.05)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Number of gradient samples', fontsize=12)
ax2.set_ylabel('Cos(MMI) (alignment with MMI direction)', fontsize=12)
ax2.set_title('MMI Alignment (Cosine) vs Sample Size', fontsize=13)
ax2.set_ylim(0, 1.05)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle('Gradient Quality: dS=0.01 vs dS=0.001\n(5 runs each, mean±std)', fontsize=14, y=1.02)
plt.tight_layout()

# Save plot
output_path = os.path.join(script_dir, 'samples_vs_gradient_quality_dS_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

output_pdf = os.path.join(script_dir, 'samples_vs_gradient_quality_dS_comparison.pdf')
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Saved: {output_pdf}")

plt.close()

# Print summary tables
print("\n" + "="*80)
print("dS=0.01 MULTIRUN RESULTS:")
print("-"*80)
print(f"{'Samples':<10} {'R² mean':<12} {'R² std':<12} {'Cos mean':<12} {'Cos std':<12} {'Runs':<6}")
print("-"*80)
for samples in sorted(dS001_stats.keys()):
    s = dS001_stats[samples]
    print(f"{samples:<10} {s['r2_mean']:<12.4f} {s['r2_std']:<12.4f} {s['cos_mean']:<+12.4f} {s['cos_std']:<12.4f} {s['n_runs']:<6}")

print("\n" + "="*80)
print("dS=0.001 MULTIRUN RESULTS:")
print("-"*80)
print(f"{'Samples':<10} {'R² mean':<12} {'R² std':<12} {'Cos mean':<12} {'Cos std':<12} {'Runs':<6}")
print("-"*80)
for samples in sorted(dS0001_stats.keys()):
    s = dS0001_stats[samples]
    print(f"{samples:<10} {s['r2_mean']:<12.4f} {s['r2_std']:<12.4f} {s['cos_mean']:<+12.4f} {s['cos_std']:<12.4f} {s['n_runs']:<6}")
print("="*80)
