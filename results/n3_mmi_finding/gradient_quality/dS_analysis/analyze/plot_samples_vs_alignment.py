#!/usr/bin/env python3
"""
Plot gradient alignment with analytical gradient vs sample count.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# ===== Compute analytical gradient for the symmetric point =====
def compute_optimal_achieved(S_target):
    """Compute optimal achieved point for a symmetric target in MMI-violated region."""
    s = S_target[0]  # S_A
    t = S_target[2]  # S_AB (in binary ordering)
    u = S_target[6]  # S_ABC

    # Check which region
    if t > 2*s:  # SA violated
        D = np.sqrt(15*(s + 2*t)**2 + 25*u**2)
        s_p = (s + 2*t) / D
        t_p = 2 * s_p
        u_p = 5*u / D
    elif u > 3*(t - s):  # MMI violated (but SA satisfied)
        rho = (3*s + 4*t + u) / (4*s + 3*t - u)
        rho = max(1, min(2, rho))
        s_p = 1 / np.sqrt(6 * (2*rho**2 - 3*rho + 2))
        t_p = rho * s_p
        u_p = 3 * (rho - 1) * s_p
    else:  # Inside HEC
        s_p, t_p, u_p = s, t, u
        norm = np.sqrt(3*s_p**2 + 3*t_p**2 + u_p**2)
        s_p, t_p, u_p = s_p/norm, t_p/norm, u_p/norm

    return np.array([s_p, s_p, t_p, s_p, t_p, t_p, u_p])

def compute_reward(S_target):
    S_achieved = compute_optimal_achieved(S_target)
    return np.dot(S_target, S_achieved) / (np.linalg.norm(S_target) * np.linalg.norm(S_achieved))

def compute_analytical_gradient(S_target):
    """Compute analytical gradient numerically."""
    eps = 1e-6
    grad = np.zeros(7)
    for i in range(7):
        S_plus = S_target.copy()
        S_plus[i] += eps
        S_minus = S_target.copy()
        S_minus[i] -= eps
        grad[i] = (compute_reward(S_plus) - compute_reward(S_minus)) / (2*eps)
    return grad

# The symmetric point used in experiments: (1/7, 1/7, ..., 1/7)
base_S = np.array([1/7]*7)
analytical_grad = compute_analytical_gradient(base_S)
analytical_grad_norm = analytical_grad / np.linalg.norm(analytical_grad)

print(f"Analytical gradient (normalized): {analytical_grad_norm}")

# Also show MMI for comparison
mmi = np.array([-1, -1, +1, -1, +1, +1, -1])
mmi_norm = mmi / np.linalg.norm(mmi)
print(f"Analytical vs MMI alignment: {np.dot(analytical_grad_norm, mmi_norm):.4f}")

# ===== Collect data =====
sample_counts = [10, 20, 30, 40, 50]
results = {s: {'alignments': []} for s in sample_counts}

for samples in sample_counts:
    dir_name = f"samples{samples}_dS002_multirun"

    for run in range(1, 11):
        path = f"{base_dir}/{dir_name}/run{run}/gradient_samples_stage_0.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)

            grad = np.array(data['gradient_computed'])
            grad_norm = grad / (np.linalg.norm(grad) + 1e-10)
            # Compute alignment with ANALYTICAL gradient, not MMI
            alignment = np.dot(analytical_grad_norm, grad_norm)
            results[samples]['alignments'].append(alignment)

# Compute statistics
stats = {}
for samples in sample_counts:
    aligns = results[samples]['alignments']
    if aligns:
        stats[samples] = {
            'mean': np.mean(aligns),
            'std': np.std(aligns),
            'all': aligns,
        }

# Print summary
print()
print(f"{'Samples':>8} {'Mean Align':>12} {'Std':>10}")
print("-"*35)
for samples in sample_counts:
    s = stats[samples]
    print(f"{samples:>8} {s['mean']:>12.4f} {s['std']:>10.4f}")

# ===== Create Plot =====
fig, ax = plt.subplots(figsize=(8, 6))

# Points with error bars
means = [stats[s]['mean'] for s in sample_counts]
stds = [stats[s]['std'] for s in sample_counts]
ax.errorbar(sample_counts, means, yerr=stds, fmt='o-', color='tab:green',
            markersize=10, linewidth=2, capsize=5, capthick=2, elinewidth=2)

ax.set_xlabel('Number of Samples', fontsize=16)
ax.set_ylabel('Cosine Similarity with Analytical Gradient', fontsize=16)
ax.set_xlim(5, 55)
ax.set_ylim(0.96, 1.01)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=14)

plt.tight_layout(pad=1.5)

# Save
output_path = os.path.join(script_dir, 'samples_vs_alignment.png')
plt.savefig(output_path, dpi=150)
print(f"\nSaved: {output_path}")

output_pdf = os.path.join(script_dir, 'samples_vs_alignment.pdf')
plt.savefig(output_pdf)
print(f"Saved: {output_pdf}")

plt.close()
