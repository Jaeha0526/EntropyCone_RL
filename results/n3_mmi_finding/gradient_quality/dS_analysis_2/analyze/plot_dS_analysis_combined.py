#!/usr/bin/env python3
"""
Plot R² and Cosine similarity (vs analytical gradient) vs dS for both max1 and max10 experiments.
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

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

print(f"Base point: {base_S}")
print(f"Analytical gradient (normalized): {analytical_grad_norm}")

# Also compute MMI normal for reference
mmi_normal = np.array([-1, -1, +1, -1, +1, +1, -1], dtype=float)
mmi_normal = mmi_normal / np.linalg.norm(mmi_normal)
print(f"MMI normal: {mmi_normal}")
print(f"Analytical vs MMI alignment: {np.dot(analytical_grad_norm, mmi_normal):.4f}")

# ===== Collect data =====
def collect_data(pattern):
    """Collect R² and Cos(analytical) for experiments matching pattern."""
    results = []
    for path in sorted(glob.glob(os.path.join(base_dir, pattern))):
        name = os.path.basename(os.path.dirname(path))
        data = json.load(open(path))

        r2 = data.get('linear_regression', {}).get('r2_with_intercept', 0)
        grad = np.array(data.get('gradient_computed', [0]*7))
        if np.linalg.norm(grad) > 0:
            grad_norm = grad / np.linalg.norm(grad)
            cos_analytical = np.dot(grad_norm, analytical_grad_norm)
        else:
            cos_analytical = 0

        dS = data.get('dS', 0)
        results.append((dS, r2, cos_analytical))

    results.sort(key=lambda x: x[0])
    return results

# Collect data for max1 and max10
max1_data = collect_data('dS*_max1/gradient_samples_stage_0.json')
max10_data = collect_data('dS*_max10/gradient_samples_stage_0.json')

# Extract values
dS_max1 = [r[0] for r in max1_data]
r2_max1 = [r[1] for r in max1_data]
cos_max1 = [r[2] for r in max1_data]

dS_max10 = [r[0] for r in max10_data]
r2_max10 = [r[1] for r in max10_data]
cos_max10 = [r[2] for r in max10_data]

# Create figure with more space between plots (less wide, taller)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# ===== Left plot: R² =====
ax1.plot(dS_max1, r2_max1, 'o-', color='tab:blue', linewidth=2, markersize=10, label='max1')
ax1.plot(dS_max10, r2_max10, 's--', color='tab:blue', linewidth=2, markersize=10, alpha=0.7, label='max10')
ax1.set_xlabel(r'$\delta S_{\max}$', fontsize=16)
ax1.set_ylabel(r'$R^2$', fontsize=16)
ax1.set_title('Linear Regression Quality\nvs Perturbation Size', fontsize=17, pad=15)
ax1.set_xscale('log')
ax1.set_ylim(0.4, 1.02)
ax1.legend(loc='lower right', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', labelsize=13)

# ===== Right plot: Cos(analytical gradient) =====
ax2.plot(dS_max1, cos_max1, 'o-', color='tab:red', linewidth=2, markersize=10, label='max1')
ax2.plot(dS_max10, cos_max10, 's--', color='tab:red', linewidth=2, markersize=10, alpha=0.7, label='max10')
ax2.set_xlabel(r'$\delta S_{\max}$', fontsize=16)
ax2.set_ylabel('Cosine similarity with analytical gradient', fontsize=16)
ax2.set_title('Gradient Direction Accuracy\nvs Perturbation Size', fontsize=17, pad=15)
ax2.set_xscale('log')
ax2.set_ylim(0.85, 1.01)
ax2.legend(loc='lower right', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', labelsize=13)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)  # Add space between plots after tight_layout

# Save plot
output_path = os.path.join(script_dir, 'dS_vs_gradient_quality_combined.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

output_pdf = os.path.join(script_dir, 'dS_vs_gradient_quality_combined.pdf')
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Saved: {output_pdf}")

plt.close()

# Print summary table
print("\n" + "="*80)
print(f"{'dS':<10} {'R² max1':<12} {'R² max10':<12} {'Cos(anal) max1':<16} {'Cos(anal) max10':<16}")
print("-"*80)

all_dS = sorted(set(dS_max1 + dS_max10))
max1_dict = {r[0]: (r[1], r[2]) for r in max1_data}
max10_dict = {r[0]: (r[1], r[2]) for r in max10_data}

for dS in all_dS:
    r2_1, cos_1 = max1_dict.get(dS, (None, None))
    r2_10, cos_10 = max10_dict.get(dS, (None, None))
    r2_1_str = f"{r2_1:.4f}" if r2_1 else "-"
    r2_10_str = f"{r2_10:.4f}" if r2_10 else "-"
    cos_1_str = f"{cos_1:.4f}" if cos_1 else "-"
    cos_10_str = f"{cos_10:.4f}" if cos_10 else "-"
    print(f"{dS:<10.4f} {r2_1_str:<12} {r2_10_str:<12} {cos_1_str:<16} {cos_10_str:<16}")
print("="*80)
