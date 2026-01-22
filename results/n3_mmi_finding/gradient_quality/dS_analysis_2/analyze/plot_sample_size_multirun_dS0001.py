#!/usr/bin/env python3
"""
Plot R² and Cosine similarity vs sample size with error bars from multiple runs.
For dS=0.001 experiments.
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

# Collect multirun data for dS=0.001
multirun_results = {}  # {samples: [(r2, cos), ...]}

for samples in [10, 20, 30, 40, 50, 60, 70]:
    multirun_results[samples] = []
    for run in range(1, 6):
        path = os.path.join(base_dir, f'dS0001_samples{samples}_multirun/run{run}/gradient_samples_stage_0.json')
        if os.path.exists(path):
            r2, cos = get_gradient_quality(path)
            multirun_results[samples].append((r2, cos))

# Calculate statistics for multirun
multirun_stats = {}
for samples, runs in multirun_results.items():
    if runs:
        r2_vals = [r[0] for r in runs]
        cos_vals = [r[1] for r in runs]
        multirun_stats[samples] = {
            'r2_mean': np.mean(r2_vals),
            'r2_std': np.std(r2_vals),
            'cos_mean': np.mean(cos_vals),
            'cos_std': np.std(cos_vals),
            'n_runs': len(runs)
        }

if not multirun_stats:
    print("No results found yet. Experiments may still be running.")
    exit(0)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Prepare data for plotting
multi_samples = sorted(multirun_stats.keys())
multi_r2_mean = [multirun_stats[s]['r2_mean'] for s in multi_samples]
multi_r2_std = [multirun_stats[s]['r2_std'] for s in multi_samples]
multi_cos_mean = [multirun_stats[s]['cos_mean'] for s in multi_samples]
multi_cos_std = [multirun_stats[s]['cos_std'] for s in multi_samples]

# ===== Left plot: R² =====
ax1.errorbar(multi_samples, multi_r2_mean, yerr=multi_r2_std,
             fmt='o-', color='tab:blue', linewidth=2, markersize=10,
             capsize=5, capthick=2, label='dS=0.001 (5 runs, mean±std)')

ax1.axhline(y=0.95, color='gray', linestyle=':', linewidth=1.5, label='R²=0.95 threshold')
ax1.set_xlabel('Number of gradient samples', fontsize=12)
ax1.set_ylabel('R² (gradient reliability)', fontsize=12)
ax1.set_title('Gradient Reliability (R²) vs Sample Size\ndS=0.001', fontsize=13)
ax1.set_ylim(0, 1.05)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)

# ===== Right plot: Cos(MMI) =====
ax2.errorbar(multi_samples, multi_cos_mean, yerr=multi_cos_std,
             fmt='o-', color='tab:red', linewidth=2, markersize=10,
             capsize=5, capthick=2, label='dS=0.001 (5 runs, mean±std)')

ax2.set_xlabel('Number of gradient samples', fontsize=12)
ax2.set_ylabel('Cos(MMI) (alignment with MMI direction)', fontsize=12)
ax2.set_title('MMI Alignment (Cosine) vs Sample Size\ndS=0.001', fontsize=13)
ax2.set_ylim(0, 1.05)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle('Gradient Quality vs Sample Size (dS=0.001)\nn=3 symmetric point', fontsize=14, y=1.02)
plt.tight_layout()

# Save plot
output_path = os.path.join(script_dir, 'samples_vs_gradient_quality_dS0001_multirun.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

output_pdf = os.path.join(script_dir, 'samples_vs_gradient_quality_dS0001_multirun.pdf')
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Saved: {output_pdf}")

plt.close()

# Print summary table
print("\n" + "="*70)
print("MULTIRUN RESULTS (dS=0.001, 5 runs each):")
print("-"*70)
print(f"{'Samples':<10} {'R² mean':<12} {'R² std':<12} {'Cos mean':<12} {'Cos std':<12}")
print("-"*70)
for samples in sorted(multirun_stats.keys()):
    s = multirun_stats[samples]
    print(f"{samples:<10} {s['r2_mean']:<12.4f} {s['r2_std']:<12.4f} {s['cos_mean']:<+12.4f} {s['cos_std']:<12.4f}")
print("="*70)
