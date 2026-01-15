#!/usr/bin/env python3
"""
Analysis script for Section 3.3: N=3 RL Reward Landscape

Shows RL-discovered reward landscape across the symmetric (s,t) slice.
Note: Analytical formula assumes symmetric optimal, but RL can find
non-symmetric realizations with higher rewards.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Grid parameters
GRID_SIZE = 20
S_MIN, S_MAX = 0.02, 0.55
T_MIN, T_MAX = 0.02, 0.55


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
            s, t, u = meta['s'], meta['t'], meta['u']
            i, j = meta['grid_indices']['i'], meta['grid_indices']['j']
            point_idx = meta.get('point_index', i * GRID_SIZE + j)

            if point_idx not in point_data:
                point_data[point_idx] = {
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
        s, t, u = data['s'], data['t'], data['u']

        if completed_runs > 0:
            rl_max = max(all_rewards)
            rl_mean = np.mean(all_rewards)
            rl_std = np.std(all_rewards)
        else:
            rl_max = rl_mean = rl_std = np.nan

        # Determine region
        sa_satisfied = t <= 2*s
        mmi_satisfied = u <= 3*(t - s)

        if sa_satisfied and mmi_satisfied:
            region = 'inside_HEC'
        elif not sa_satisfied:
            region = 'SA_violated'
        else:
            region = 'MMI_violated'

        results.append({
            'i': data['i'], 'j': data['j'],
            's': s, 't': t, 'u': u,
            'completed_runs': completed_runs,
            'rl_max': rl_max,
            'rl_mean': rl_mean,
            'rl_std': rl_std,
            'region': region
        })

    return results


def load_results(results_dir):
    """Load all grid validation results (legacy single-directory version)."""
    return load_results_combined([results_dir])


def plot_reward_landscape(results, output_dir):
    """Generate reward landscape plots for the paper."""

    completed = [r for r in results if r['completed_runs'] >= 5]

    # Create reward grid
    reward_grid = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    region_grid = np.full((GRID_SIZE, GRID_SIZE), '', dtype=object)

    for r in completed:
        reward_grid[r['i'], r['j']] = r['rl_max']
        region_grid[r['i'], r['j']] = r['region']

    # Figure 1: RL Reward Heatmap with HEC boundaries
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(reward_grid, origin='lower', cmap='viridis',
                   vmin=0.5, vmax=1.0,
                   extent=[T_MIN, T_MAX, S_MIN, S_MAX],
                   aspect='auto')

    # Add HEC boundary lines
    t_range = np.linspace(T_MIN, T_MAX, 100)

    # SA boundary: t = 2s  =>  s = t/2
    s_sa = t_range / 2
    mask_sa = (s_sa >= S_MIN) & (s_sa <= S_MAX)
    ax.plot(t_range[mask_sa], s_sa[mask_sa], 'w--', lw=2, label='SA: t=2s')

    # MMI boundary: u = 3(t-s)  =>  sqrt(1-3s²-3t²) = 3(t-s)
    # This is complex, plot numerically
    s_mmi = []
    t_mmi = []
    for t_val in np.linspace(T_MIN, T_MAX, 200):
        for s_val in np.linspace(S_MIN, min(t_val, S_MAX), 100):
            u_val_sq = 1 - 3*s_val**2 - 3*t_val**2
            if u_val_sq > 0:
                u_val = np.sqrt(u_val_sq)
                if np.abs(u_val - 3*(t_val - s_val)) < 0.01:
                    s_mmi.append(s_val)
                    t_mmi.append(t_val)
    if s_mmi:
        ax.scatter(t_mmi, s_mmi, c='magenta', s=1, alpha=0.5, label='MMI boundary')

    ax.set_xlabel('t (two-party entropy)', fontsize=12)
    ax.set_ylabel('s (single-party entropy)', fontsize=12)
    ax.set_title('RL Reward Landscape on Symmetric Slice\n(N=3, 5 runs per point)', fontsize=14)
    ax.legend(loc='upper left')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('RL Reward (max of 5 runs)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'rl_reward_landscape.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'rl_reward_landscape.pdf', bbox_inches='tight')
    plt.close()

    # Figure 2: Reward histogram by region
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    regions = ['inside_HEC', 'SA_violated', 'MMI_violated']
    titles = ['Inside HEC', 'SA Violated', 'MMI Violated']
    colors = ['green', 'blue', 'red']

    for ax, region, title, color in zip(axes, regions, titles, colors):
        data = [r['rl_max'] for r in completed if r['region'] == region]
        if data:
            ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(data), color='orange', lw=2, label=f'Mean={np.mean(data):.3f}')
            ax.set_xlabel('RL Reward')
            ax.set_ylabel('Count')
            ax.set_title(f'{title}\n(n={len(data)}, σ={np.std(data):.3f})')
            ax.legend()
            ax.set_xlim(0.5, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'reward_by_region.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'reward_by_region.pdf', bbox_inches='tight')
    plt.close()


def print_statistics(results, output_dir):
    """Print and save summary statistics."""

    completed = [r for r in results if r['completed_runs'] >= 5]

    print("=" * 70)
    print("SECTION 3.3: N=3 RL REWARD LANDSCAPE")
    print("=" * 70)
    print()
    print(f"Total grid points: {GRID_SIZE * GRID_SIZE}")
    print(f"Valid points (in disk): {len(results)}")
    print(f"Completed (5 runs): {len(completed)}")
    print()

    rl_rewards = [r['rl_max'] for r in completed]
    print("--- Overall RL Statistics ---")
    print(f"Mean reward:  {np.mean(rl_rewards):.6f}")
    print(f"Std reward:   {np.std(rl_rewards):.6f}")
    print(f"Min reward:   {np.min(rl_rewards):.6f}")
    print(f"Max reward:   {np.max(rl_rewards):.6f}")
    print()

    # By region
    print("--- Results by Region ---")
    for region in ['inside_HEC', 'SA_violated', 'MMI_violated']:
        data = [r['rl_max'] for r in completed if r['region'] == region]
        if data:
            print(f"\n{region}: {len(data)} points")
            print(f"  Mean: {np.mean(data):.6f}")
            print(f"  Std:  {np.std(data):.6f}")
            print(f"  Min:  {np.min(data):.6f}")
            print(f"  Max:  {np.max(data):.6f}")

    # High reward points
    print()
    print("--- High Reward Points ---")
    for threshold in [0.99, 0.95, 0.90]:
        count = len([r for r in rl_rewards if r > threshold])
        print(f"Reward > {threshold}: {count} points ({100*count/len(rl_rewards):.1f}%)")

    # Save summary
    summary = {
        'grid_size': GRID_SIZE,
        'total_points': len(results),
        'completed_points': len(completed),
        'overall': {
            'mean': float(np.mean(rl_rewards)),
            'std': float(np.std(rl_rewards)),
            'min': float(np.min(rl_rewards)),
            'max': float(np.max(rl_rewards))
        },
        'by_region': {}
    }

    for region in ['inside_HEC', 'SA_violated', 'MMI_violated']:
        data = [r['rl_max'] for r in completed if r['region'] == region]
        if data:
            summary['by_region'][region] = {
                'count': len(data),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data))
            }

    with open(output_dir / 'rl_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Summary saved to: {output_dir / 'rl_summary.json'}")


def main():
    script_dir = Path(__file__).parent
    base_results = Path('/resnick/groups/OoguriGroup/jaeha/entropyconeRL/results/experiment_for_3-3')

    # Load from v6 only (dt=0.1, seed=999, 20 runs per point)
    results_dirs = [
        base_results / 'n3_grid_validation_v6',   # dt=0.1, seed=999, 20 runs
    ]
    output_dir = script_dir

    print("Loading from v6 (dt=0.1, seed=999, 20 runs per point)")
    for d in results_dirs:
        print(f"  {d.name}: {'exists' if d.exists() else 'not found'}")

    results = load_results_combined(results_dirs)
    print(f"Loaded {len(results)} grid points")

    completed = [r for r in results if r['completed_runs'] >= 5]
    print(f"Completed (5 runs): {len(completed)}")

    if len(completed) == 0:
        print("No completed results yet.")
        return

    print("\nGenerating plots...")
    plot_reward_landscape(results, output_dir)

    print_statistics(results, output_dir)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
