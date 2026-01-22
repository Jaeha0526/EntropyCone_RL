#!/usr/bin/env python3
"""
Plot R² and Cosine similarity vs sample size for dS=0.01, max1 experiments.
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# Correct MMI formula for n=3
mmi_raw = np.array([-1, -1, +1, -1, +1, +1, -1])
mmi_norm = mmi_raw / np.linalg.norm(mmi_raw)

# Collect data for sample size experiments
results = []
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# Look for sample size experiments
for path in sorted(glob.glob(os.path.join(base_dir, 'samples*_dS001_max1/gradient_samples_stage_0.json'))):
    name = os.path.basename(os.path.dirname(path))
    data = json.load(open(path))

    # Get R² from linear_regression
    r2 = data.get('linear_regression', {}).get('r2_with_intercept', 0)

    # Compute Cos(MMI) from gradient
    grad = np.array(data.get('gradient_computed', [0]*7))
    if np.linalg.norm(grad) > 0:
        grad_norm = grad / np.linalg.norm(grad)
        cos_mmi = np.dot(grad_norm, mmi_norm)
    else:
        cos_mmi = 0

    n_samples = data.get('n_samples', 0)
    results.append((n_samples, r2, cos_mmi))

# Also include the existing dS001_max1 experiment (50 samples)
path_50 = os.path.join(base_dir, 'dS001_max1/gradient_samples_stage_0.json')
if os.path.exists(path_50):
    data = json.load(open(path_50))
    r2 = data.get('linear_regression', {}).get('r2_with_intercept', 0)
    grad = np.array(data.get('gradient_computed', [0]*7))
    if np.linalg.norm(grad) > 0:
        grad_norm = grad / np.linalg.norm(grad)
        cos_mmi = np.dot(grad_norm, mmi_norm)
    else:
        cos_mmi = 0
    n_samples = data.get('n_samples', 50)
    results.append((n_samples, r2, cos_mmi))

if not results:
    print("No results found yet. Experiments may still be running.")
    exit(0)

# Sort by sample size
results.sort(key=lambda x: x[0])
samples_values = [r[0] for r in results]
r2_values = [r[1] for r in results]
cos_values = [r[2] for r in results]

# Create figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot R² on left y-axis
color1 = 'tab:blue'
ax1.set_xlabel('Number of gradient samples', fontsize=12)
ax1.set_ylabel('R² (gradient reliability)', color=color1, fontsize=12)
line1 = ax1.plot(samples_values, r2_values, 'o-', color=color1, linewidth=2, markersize=10, label='R²')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1.05)

# Add horizontal line at R² = 0.95 (threshold for reliable gradient)
ax1.axhline(y=0.95, color=color1, linestyle='--', alpha=0.5, linewidth=1)
ax1.text(max(samples_values)*1.02, 0.95, 'R²=0.95', color=color1, alpha=0.7, va='center')

# Plot Cos(MMI) on right y-axis
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Cos(MMI) (alignment with MMI direction)', color=color2, fontsize=12)
line2 = ax2.plot(samples_values, cos_values, 's-', color=color2, linewidth=2, markersize=10, label='Cos(MMI)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0.8, 1.02)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right', fontsize=11)

# Title and grid
plt.title('Gradient Quality vs Number of Samples\ndS=0.01, max1, n=3 symmetric point', fontsize=14)
ax1.grid(True, alpha=0.3)

# Add annotations for key points
for i, (n_samp, r2, cos) in enumerate(results):
    if n_samp in [10, 50]:
        ax1.annotate(f'n={n_samp}\nR²={r2:.2f}',
                    xy=(n_samp, r2), xytext=(n_samp+5, r2-0.08),
                    fontsize=9, color=color1,
                    arrowprops=dict(arrowstyle='->', color=color1, alpha=0.5))

plt.tight_layout()

# Save plot
output_path = os.path.join(script_dir, 'samples_vs_gradient_quality.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

# Also save as PDF
output_pdf = os.path.join(script_dir, 'samples_vs_gradient_quality.pdf')
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Saved: {output_pdf}")

plt.close()

# Print summary table
print("\n" + "="*60)
print(f"{'Samples':<10} {'R²':<10} {'Cos(MMI)':<10}")
print("-"*30)
for n_samp, r2, cos in results:
    print(f"{n_samp:<10} {r2:<10.4f} {cos:<+10.4f}")
print("="*60)
