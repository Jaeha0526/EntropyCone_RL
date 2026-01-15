#!/usr/bin/env python3
"""
N=3 Grid Point Classification Plot

Shows max rewards sorted by analytical (optimal) reward.
Uses v6 data only (20 runs per point with dt=0.1, seed=999).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add visualization directory to path for importing the analytical formula
sys.path.insert(0, str(Path(__file__).parent.parent / 'visualization'))
from plot_reward_landscape_analytic import compute_reward_analytic

# Grid parameters
GRID_SIZE = 20
S_MIN, S_MAX = 0.02, 0.55
T_MIN, T_MAX = 0.02, 0.55


def load_results_v6():
    """
    Load results from v6 experiment only (20 runs per point with dt=0.1).
    """
    base_results = Path('/resnick/groups/OoguriGroup/jaeha/entropyconeRL/results/experiment_for_3-3')
    v6_dir = base_results / 'n3_grid_validation_v6'

    results = []

    for point_idx in range(1, GRID_SIZE * GRID_SIZE + 1):
        point_dir = v6_dir / f"point_{point_idx}"
        metadata_file = point_dir / "grid_point_metadata.json"
        results_file = point_dir / "results.json"

        if not (metadata_file.exists() and results_file.exists()):
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)
        s, t = metadata['s'], metadata['t']

        with open(results_file) as f:
            result_data = json.load(f)

        if 'multi_run_statistics' not in result_data:
            continue

        stats = result_data['multi_run_statistics']
        all_rewards = stats.get('all_rewards', [])

        if len(all_rewards) == 0:
            continue

        # Compute analytical (optimal) reward
        analytical_reward = compute_reward_analytic(s, t)

        # Determine if inside HEC
        u_sq = 1 - 3*s**2 - 3*t**2
        u = np.sqrt(max(u_sq, 0))
        sa_satisfied = t <= 2*s
        mmi_satisfied = u <= 3*(t - s)
        inside_hec = sa_satisfied and mmi_satisfied

        # Max reward from all runs
        max_reward = max(all_rewards)

        results.append({
            'point_idx': point_idx,
            's': s,
            't': t,
            'analytical_reward': analytical_reward,
            'max_reward': max_reward,
            'mean_reward': stats.get('mean_reward', np.nan),
            'std_reward': stats.get('std_reward', np.nan),
            'inside_hec': inside_hec,
            'n_runs': len(all_rewards)
        })

    return results


def transform_reward(r):
    """Transform reward to log(1-r) for better visualization near r=1."""
    return np.log(1 - np.clip(r, 0, 1 - 1e-15) + 1e-15)


def create_sorted_plot(results, output_dir):
    """
    Create sorted scatter plot:
    - X-axis: grid points sorted by analytical (optimal) reward
    - Y-axis: log(1 - reward)
    - Dotted line: optimal reward ceiling (front layer)
    - Data points: max from v6 (20 runs with dt=0.1)
    - Colors: green = inside HEC, red = outside HEC
    """
    output_dir = Path(output_dir)

    # Extract data
    analytical = np.array([r['analytical_reward'] for r in results])
    max_reward = np.array([r['max_reward'] for r in results])
    inside_hec = np.array([r['inside_hec'] for r in results])
    n_runs = np.array([r['n_runs'] for r in results])

    # Sort by analytical reward
    sorted_indices = np.argsort(analytical)

    analytical_sorted = analytical[sorted_indices]
    max_reward_sorted = max_reward[sorted_indices]
    inside_hec_sorted = inside_hec[sorted_indices]

    # Transform to log(1-r) space (negative values, closer to 0 = better)
    analytical_transformed = transform_reward(analytical_sorted)
    max_reward_transformed = transform_reward(max_reward_sorted)

    # Cap at -12 for visualization (log scale)
    y_min = -12
    analytical_transformed = np.clip(analytical_transformed, y_min, 0)
    max_reward_transformed = np.clip(max_reward_transformed, y_min, 0)

    # X positions
    x_positions = np.arange(len(sorted_indices))

    inside_mask = inside_hec_sorted
    outside_mask = ~inside_hec_sorted

    # ==================== Create plot ====================
    plt.figure(figsize=(16, 10))

    # Plot data points first (lower z-order)
    plt.scatter(x_positions[inside_mask], max_reward_transformed[inside_mask],
                c='green', marker='o', s=60, alpha=0.8,
                edgecolors='darkgreen', linewidths=1,
                label='inside HEC', zorder=2)
    plt.scatter(x_positions[outside_mask], max_reward_transformed[outside_mask],
                c='red', marker='o', s=60, alpha=0.8,
                edgecolors='darkred', linewidths=1,
                label='outside HEC', zorder=2)

    # Plot optimal ceiling as dotted line (front layer, highest z-order)
    # Use coarser dashes: (dash_length, gap_length)
    plt.plot(x_positions, analytical_transformed, 'k', linewidth=3,
             linestyle=(0, (5, 3)), label='optimal reward', zorder=10)

    plt.xlabel('Grid Points (sorted by analytical reward)', fontsize=32, labelpad=20)
    plt.ylabel(r'$\log(1 - \mathrm{reward})$', fontsize=32, labelpad=20)
    # No title
    plt.legend(fontsize=28, loc='lower left')
    plt.tick_params(axis='both', labelsize=26)
    plt.grid(True, alpha=0.3)
    plt.ylim(y_min, 0)
    plt.xlim(-5, 340)
    plt.tight_layout()

    plot_path = output_dir / 'n3_grid_classification_max_sorted.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")

    plot_path_pdf = output_dir / 'n3_grid_classification_max_sorted.pdf'
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    print(f"Saved: {plot_path_pdf}")
    plt.close()

    # ==================== Print summary statistics ====================
    print("\n" + "=" * 60)
    print("Summary Statistics (v6: dt=0.1, seed=999)")
    print("=" * 60)
    print(f"Total runs per point: {int(n_runs.min())}-{int(n_runs.max())}")

    for label, mask in [('Inside HEC', inside_hec), ('Outside HEC', ~inside_hec)]:
        if mask.any():
            print(f"\n{label} ({mask.sum()} points):")
            print(f"  Analytical:    min={analytical[mask].min():.6f}, max={analytical[mask].max():.6f}")
            print(f"  Max (20 runs): min={max_reward[mask].min():.6f}, max={max_reward[mask].max():.6f}")

    # Gap statistics
    gap = analytical - max_reward

    print(f"\nGap (analytical - RL max):")
    print(f"  mean={gap.mean():.6f}, max={gap.max():.6f}")


def main():
    print("=" * 60)
    print("N=3 Grid Classification Plot (v6 only)")
    print("=" * 60)

    # Load results from v6 only (20 runs with dt=0.1)
    print("\nLoading results from v6 (dt=0.1, seed=999, 20 runs/point)...")
    results = load_results_v6()
    print(f"Loaded {len(results)} grid points")

    if len(results) == 0:
        print("ERROR: No v6 results found. Experiment may still be running.")
        return

    # Count by HEC status
    inside = sum(1 for r in results if r['inside_hec'])
    outside = len(results) - inside
    print(f"  Inside HEC: {inside}")
    print(f"  Outside HEC: {outside}")

    # Create plot
    output_dir = Path(__file__).parent
    create_sorted_plot(results, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
