#!/usr/bin/env python3
"""
Analysis script for Section 3.3: N=3 Grid Validation

Compares RL-achieved rewards with analytical predictions across
the symmetric (s,t) slice of the entropy cone.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Grid parameters
GRID_SIZE = 20
S_MIN, S_MAX = 0.02, 0.55
T_MIN, T_MAX = 0.02, 0.55


def compute_analytical_reward(s, t, u):
    """
    Compute the analytical optimal reward for target (s, s, s, t, t, t, u).

    Returns:
        reward: float or None if formula is invalid
        region: str describing which region the point is in
    """
    # Check constraints
    sa_satisfied = t <= 2*s
    mmi_satisfied = u <= 3*(t - s)

    # Inside HEC
    if sa_satisfied and mmi_satisfied:
        return 1.0, 'inside_HEC'

    # SA violated (t > 2s)
    if not sa_satisfied:
        if 5*u > 3*(s + 2*t):
            # Corner solution
            s_opt = 1 / (2 * np.sqrt(6))
            t_opt = 1 / np.sqrt(6)
            u_opt = 3 / (2 * np.sqrt(6))
        else:
            # Interior solution on SA boundary
            D = np.sqrt(15*(s + 2*t)**2 + 25*u**2)
            s_opt = (s + 2*t) / D
            t_opt = 2 * s_opt
            u_opt = 5*u / D
        reward = 3*s*s_opt + 3*t*t_opt + u*u_opt
        return reward, 'SA_violated'

    # MMI violated (t <= 2s but u > 3(t-s))
    # Use formula: rho = (3s + 4t + u) / (4s + 3t - u), clamped to [1, 2]
    denom = 4*s + 3*t - u

    if denom <= 0:
        # Denominator non-positive means rho -> infinity or negative
        # Optimal is at corner (rho = 2)
        rho = 2.0
    else:
        rho = (3*s + 4*t + u) / denom
        # Clamp to valid range [1, 2]
        rho = np.clip(rho, 1.0, 2.0)
    s_opt = 1 / np.sqrt(6 * (2*rho**2 - 3*rho + 2))
    t_opt = rho * s_opt
    u_opt = 3 * (rho - 1) * s_opt
    reward = 3*s*s_opt + 3*t*t_opt + u*u_opt
    return reward, 'MMI_violated'


def load_results_combined(results_dirs):
    """Load all grid validation results from multiple directories and combine."""
    # First pass: collect all rewards per point
    point_data = {}

    for results_dir in results_dirs:
        if not results_dir.exists():
            continue

        for point_dir in sorted(results_dir.glob('point_*')):
            meta_file = point_dir / 'grid_point_metadata.json'
            results_file = point_dir / 'results.json'

            if not meta_file.exists():
                continue

            meta = json.load(open(meta_file))
            point_idx = meta['point_index']
            s, t, u = meta['s'], meta['t'], meta['u']
            i, j = meta['grid_indices']['i'], meta['grid_indices']['j']

            if point_idx not in point_data:
                point_data[point_idx] = {
                    'point_idx': point_idx,
                    'i': i, 'j': j,
                    's': s, 't': t, 'u': u,
                    'all_rewards': []
                }

            # Get rewards from this version
            if results_file.exists():
                try:
                    result_data = json.load(open(results_file))
                    if 'multi_run_statistics' in result_data:
                        stats = result_data['multi_run_statistics']
                        if 'all_rewards' in stats:
                            point_data[point_idx]['all_rewards'].extend(stats['all_rewards'])
                except:
                    pass

    # Second pass: compute statistics for each point
    results = []
    for point_idx, data in sorted(point_data.items()):
        all_rewards = data['all_rewards']
        completed_runs = len(all_rewards)

        if completed_runs > 0:
            rl_max = max(all_rewards)
            rl_mean = np.mean(all_rewards)
            rl_std = np.std(all_rewards)
        else:
            rl_max = rl_mean = rl_std = np.nan

        # Compute analytical
        analytical, region = compute_analytical_reward(data['s'], data['t'], data['u'])

        results.append({
            'point_idx': data['point_idx'],
            'i': data['i'], 'j': data['j'],
            's': data['s'], 't': data['t'], 'u': data['u'],
            'completed_runs': completed_runs,
            'rl_max': rl_max,
            'rl_mean': rl_mean,
            'rl_std': rl_std,
            'analytical': analytical,
            'region': region,
            'error': (rl_max - analytical) if (analytical is not None and not np.isnan(rl_max)) else None
        })

    return results


def load_results(results_dir):
    """Load all grid validation results (legacy single-directory version)."""
    return load_results_combined([results_dir])


def plot_comparison(results, output_dir):
    """Generate comparison plots for the paper."""

    # Filter to completed results with valid analytical
    valid = [r for r in results if r['completed_runs'] >= 5
             and r['analytical'] is not None
             and not np.isnan(r['rl_max'])]

    if len(valid) == 0:
        print("No valid results to plot yet!")
        return

    rl_rewards = np.array([r['rl_max'] for r in valid])
    analytical_rewards = np.array([r['analytical'] for r in valid])
    errors = np.array([r['error'] for r in valid])

    # Figure 1: Scatter plot - RL vs Analytical
    fig, ax = plt.subplots(figsize=(6, 6))

    # Color by region
    colors = {'SA_violated': 'blue', 'MMI_violated': 'red', 'inside_HEC': 'green'}
    for region in ['SA_violated', 'MMI_violated', 'inside_HEC']:
        mask = [r['region'] == region for r in valid]
        if any(mask):
            ax.scatter(
                analytical_rewards[mask],
                rl_rewards[mask],
                c=colors[region],
                alpha=0.6,
                s=30,
                label=region.replace('_', ' ')
            )

    # Perfect agreement line
    ax.plot([0.5, 1.05], [0.5, 1.05], 'k--', lw=1, label='Perfect agreement')

    ax.set_xlabel('Analytical Reward', fontsize=12)
    ax.set_ylabel('RL Reward', fontsize=12)
    ax.set_title('RL vs Analytical Reward Comparison', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.5, 1.02)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'rl_vs_analytical_scatter.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'rl_vs_analytical_scatter.pdf', bbox_inches='tight')
    plt.close()

    # Figure 2: Error histogram
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
    ax.axvline(x=np.mean(errors), color='orange', linestyle='-', lw=2,
               label=f'Mean = {np.mean(errors):.4f}')

    ax.set_xlabel('Error (RL - Analytical)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Error Distribution (n={len(errors)}, σ={np.std(errors):.4f})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_histogram.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'error_histogram.pdf', bbox_inches='tight')
    plt.close()

    # Figure 3: Heatmaps (use ALL results, not just valid analytical)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Create grids - use all results for RL, valid for analytical/error
    rl_grid = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    analytical_grid = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    error_grid = np.full((GRID_SIZE, GRID_SIZE), np.nan)

    # Fill RL grid with ALL completed results
    for r in results:
        if r['completed_runs'] >= 5 and not np.isnan(r['rl_max']):
            rl_grid[r['i'], r['j']] = r['rl_max']

    # Fill analytical and error grids with valid results only
    for r in valid:
        analytical_grid[r['i'], r['j']] = r['analytical']
        error_grid[r['i'], r['j']] = r['error']

    # Create coordinate arrays for pcolormesh (s on x-axis, t on y-axis)
    s_edges = np.linspace(S_MIN - (S_MAX-S_MIN)/(2*(GRID_SIZE-1)),
                          S_MAX + (S_MAX-S_MIN)/(2*(GRID_SIZE-1)),
                          GRID_SIZE + 1)
    t_edges = np.linspace(T_MIN - (T_MAX-T_MIN)/(2*(GRID_SIZE-1)),
                          T_MAX + (T_MAX-T_MIN)/(2*(GRID_SIZE-1)),
                          GRID_SIZE + 1)

    # RL reward heatmap (s on x-axis, t on y-axis -> transpose)
    im0 = axes[0].pcolormesh(s_edges, t_edges, rl_grid.T, cmap='viridis',
                              vmin=0.5, vmax=1.0, shading='flat')
    axes[0].set_xlabel(r'$s = S_A = S_B = S_C$', fontsize=12)
    axes[0].set_ylabel(r'$t = S_{AB} = S_{AC} = S_{BC}$', fontsize=12)
    axes[0].set_title('RL Reward (all points)', fontsize=14)
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0])

    # Analytical reward heatmap
    im1 = axes[1].pcolormesh(s_edges, t_edges, analytical_grid.T, cmap='viridis',
                              vmin=0.5, vmax=1.0, shading='flat')
    axes[1].set_xlabel(r'$s = S_A = S_B = S_C$', fontsize=12)
    axes[1].set_ylabel(r'$t = S_{AB} = S_{AC} = S_{BC}$', fontsize=12)
    axes[1].set_title('Analytical Reward (valid only)', fontsize=14)
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1])

    # Error heatmap
    im2 = axes[2].pcolormesh(s_edges, t_edges, error_grid.T, cmap='RdBu_r',
                              vmin=-0.05, vmax=0.05, shading='flat')
    axes[2].set_xlabel(r'$s = S_A = S_B = S_C$', fontsize=12)
    axes[2].set_ylabel(r'$t = S_{AB} = S_{AC} = S_{BC}$', fontsize=12)
    axes[2].set_title('Error (RL - Analytical)', fontsize=14)
    axes[2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(output_dir / 'reward_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'reward_heatmaps.pdf', bbox_inches='tight')
    plt.close()

    # Figure 4: Log-scale heatmaps (-log(1-r))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Transform rewards: -log(1-r)
    rl_log = -np.log(np.clip(1 - rl_grid, 1e-10, 1))
    analytical_log = -np.log(np.clip(1 - analytical_grid, 1e-10, 1))

    # RL reward heatmap (log scale)
    im0 = axes[0].pcolormesh(s_edges, t_edges, rl_log.T, cmap='viridis',
                              vmin=0, vmax=5, shading='flat')
    axes[0].set_xlabel(r'$s = S_A = S_B = S_C$', fontsize=12)
    axes[0].set_ylabel(r'$t = S_{AB} = S_{AC} = S_{BC}$', fontsize=12)
    axes[0].set_title(r'RL: $-\log(1-r)$', fontsize=14)
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0])

    # Analytical reward heatmap (log scale)
    im1 = axes[1].pcolormesh(s_edges, t_edges, analytical_log.T, cmap='viridis',
                              vmin=0, vmax=5, shading='flat')
    axes[1].set_xlabel(r'$s = S_A = S_B = S_C$', fontsize=12)
    axes[1].set_ylabel(r'$t = S_{AB} = S_{AC} = S_{BC}$', fontsize=12)
    axes[1].set_title(r'Analytical: $-\log(1-r)$', fontsize=14)
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1])

    # Error heatmap (same as before, not log-transformed)
    im2 = axes[2].pcolormesh(s_edges, t_edges, error_grid.T, cmap='RdBu_r',
                              vmin=-0.05, vmax=0.05, shading='flat')
    axes[2].set_xlabel(r'$s = S_A = S_B = S_C$', fontsize=12)
    axes[2].set_ylabel(r'$t = S_{AB} = S_{AC} = S_{BC}$', fontsize=12)
    axes[2].set_title('Error (RL - Analytical)', fontsize=14)
    axes[2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(output_dir / 'reward_heatmaps_log.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'reward_heatmaps_log.pdf', bbox_inches='tight')
    plt.close()

    return valid


def print_statistics(results, output_dir):
    """Print and save summary statistics."""

    # Filter completed results
    completed = [r for r in results if r['completed_runs'] >= 5]
    valid = [r for r in completed if r['analytical'] is not None and not np.isnan(r['rl_max'])]

    # Count by region
    by_region = {}
    for r in completed:
        region = r['region']
        if region not in by_region:
            by_region[region] = []
        by_region[region].append(r)

    print("=" * 70)
    print("SECTION 3.3: N=3 GRID VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nTotal grid points: {GRID_SIZE * GRID_SIZE}")
    print(f"Points with 5 runs completed: {len(completed)}")
    print(f"Points with valid analytical formula: {len(valid)}")

    print("\n--- Results by Region ---")
    for region, data in sorted(by_region.items()):
        print(f"\n{region}: {len(data)} points")
        valid_data = [d for d in data if d['error'] is not None]
        if valid_data:
            errors = [d['error'] for d in valid_data]
            print(f"  Mean error:  {np.mean(errors):+.6f}")
            print(f"  Std error:   {np.std(errors):.6f}")
            print(f"  Max error:   {np.max(errors):+.6f}")
            print(f"  Min error:   {np.min(errors):+.6f}")
            print(f"  MAE:         {np.mean(np.abs(errors)):.6f}")

    if valid:
        errors = [r['error'] for r in valid]
        rl_rewards = [r['rl_max'] for r in valid]
        analytical_rewards = [r['analytical'] for r in valid]

        print("\n--- Overall Statistics (valid points) ---")
        print(f"Mean RL reward:         {np.mean(rl_rewards):.6f}")
        print(f"Mean analytical reward: {np.mean(analytical_rewards):.6f}")
        print(f"Correlation:            {np.corrcoef(analytical_rewards, rl_rewards)[0,1]:.6f}")
        print(f"Mean error:             {np.mean(errors):+.6f}")
        print(f"Std error:              {np.std(errors):.6f}")
        print(f"MAE:                    {np.mean(np.abs(errors)):.6f}")
        print(f"Max |error|:            {np.max(np.abs(errors)):.6f}")

    # Save to JSON
    summary = {
        'grid_size': GRID_SIZE,
        'total_points': GRID_SIZE * GRID_SIZE,
        'completed_points': len(completed),
        'valid_points': len(valid),
        'by_region': {
            region: {
                'count': len(data),
                'valid_count': len([d for d in data if d['error'] is not None]),
                'mean_error': float(np.mean([d['error'] for d in data if d['error'] is not None])) if any(d['error'] is not None for d in data) else None,
                'std_error': float(np.std([d['error'] for d in data if d['error'] is not None])) if any(d['error'] is not None for d in data) else None,
            }
            for region, data in by_region.items()
        }
    }

    if valid:
        summary['overall'] = {
            'mean_rl_reward': float(np.mean(rl_rewards)),
            'mean_analytical_reward': float(np.mean(analytical_rewards)),
            'correlation': float(np.corrcoef(analytical_rewards, rl_rewards)[0,1]),
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'mae': float(np.mean(np.abs(errors))),
            'max_abs_error': float(np.max(np.abs(errors)))
        }

    with open(output_dir / 'summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {output_dir / 'summary_statistics.json'}")

    return summary


def main():
    # Paths
    script_dir = Path(__file__).parent
    base_results = Path('/resnick/groups/OoguriGroup/jaeha/entropyconeRL/results/experiment_for_3-3')

    # Load from v6 only (dt=0.1, seed=999, 20 runs per point)
    results_dirs = [
        base_results / 'n3_grid_validation_v6',   # dt=0.1, seed=999, 20 runs
    ]
    output_dir = script_dir

    print("Loading results from v6 (dt=0.1, seed=999, 20 runs per point)")
    for d in results_dirs:
        print(f"  {d.name}: {'exists' if d.exists() else 'not found'}")
    print(f"Output directory: {output_dir}")

    # Load results
    results = load_results_combined(results_dirs)
    print(f"Loaded {len(results)} grid points")

    completed = [r for r in results if r['completed_runs'] >= 5]
    print(f"Points with 5 runs: {len(completed)}")

    if len(completed) == 0:
        print("\nNo completed results yet. Run again after jobs finish.")
        return

    # Generate plots
    print("\nGenerating plots...")
    plot_comparison(results, output_dir)

    # Print statistics
    print_statistics(results, output_dir)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
