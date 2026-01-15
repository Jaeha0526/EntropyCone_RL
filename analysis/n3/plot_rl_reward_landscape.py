#!/usr/bin/env python3
"""
Plot RL Reward Landscape matching the style of reward_landscape_with_gradient.png

Style:
- s on x-axis, t on y-axis
- Clear boundary lines for disk, SA, SSA, MMI
- Same colormap and layout
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
            s, t = meta['s'], meta['t']
            i, j = meta['grid_indices']['i'], meta['grid_indices']['j']
            point_idx = meta.get('point_index', i * GRID_SIZE + j)

            if point_idx not in point_data:
                point_data[point_idx] = {
                    'i': i, 'j': j,
                    's': s, 't': t,
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

    # Second pass: compute max for each point
    results = []
    for point_idx, data in sorted(point_data.items()):
        all_rewards = data['all_rewards']
        rl_max = max(all_rewards) if all_rewards else np.nan

        results.append({
            'i': data['i'], 'j': data['j'],
            's': data['s'], 't': data['t'],
            'rl_max': rl_max
        })

    return results


def load_results(results_dir):
    """Load all grid validation results (legacy single-directory version)."""
    return load_results_combined([results_dir])


def plot_reward_landscape(results, output_dir, use_log_scale=False):
    """Generate reward landscape plot matching reference style."""

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

    # Build reward grid (i = s index, j = t index)
    # Note: i varies with s, j varies with t
    reward_grid = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    for r in results:
        if not np.isnan(r['rl_max']):
            reward_grid[r['i'], r['j']] = r['rl_max']

    # Create coordinate arrays for pcolormesh
    s_edges = np.linspace(S_MIN - (S_MAX-S_MIN)/(2*(GRID_SIZE-1)),
                          S_MAX + (S_MAX-S_MIN)/(2*(GRID_SIZE-1)),
                          GRID_SIZE + 1)
    t_edges = np.linspace(T_MIN - (T_MAX-T_MIN)/(2*(GRID_SIZE-1)),
                          T_MAX + (T_MAX-T_MIN)/(2*(GRID_SIZE-1)),
                          GRID_SIZE + 1)

    # Custom colormap similar to reference (white/cream for high, dark red/brown for low)
    colors = ['#4a1c1c', '#8b2500', '#cd5c5c', '#f4a460', '#ffe4b5', '#fffacd', '#fffff0']
    cmap = LinearSegmentedColormap.from_list('reward_cmap', colors)

    # Apply log transformation if requested
    if use_log_scale:
        # -log(1-r): transforms reward to emphasize differences near 1
        # Clip to avoid log(0)
        plot_grid = -np.log(np.clip(1 - reward_grid, 1e-10, 1))
        vmin, vmax = 0, 5  # -log(1-0.5)≈0.7, -log(1-0.99)≈4.6
        cbar_label = r'$-\log(1 - r)$'
    else:
        plot_grid = reward_grid
        vmin, vmax = 0.5, 1.0
        cbar_label = 'Reward (Cosine Similarity)'

    # Plot heatmap (transpose so s is x-axis, t is y-axis)
    # reward_grid[i, j] where i=s_idx, j=t_idx
    # For pcolormesh with s on x and t on y, we need reward_grid.T
    im = ax.pcolormesh(s_edges, t_edges, plot_grid.T,
                       cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')

    # Plot boundary lines
    s_line = np.linspace(0, 0.6, 500)
    t_line = np.linspace(0, 0.6, 500)

    # 1. Disk boundary: 3s² + 3t² + u² = 1, with u = sqrt(1-3s²-3t²) >= 0
    #    So 3s² + 3t² <= 1, i.e., s² + t² <= 1/3
    #    Boundary: s² + t² = 1/3
    theta = np.linspace(0, np.pi/2, 200)
    r_disk = 1/np.sqrt(3)
    s_disk = r_disk * np.cos(theta)
    t_disk = r_disk * np.sin(theta)
    ax.plot(s_disk, t_disk, 'k-', lw=2, label=r'$\|S\|^2 = 1$')

    # 2. Subadditivity (SA): t = 2s (red)
    s_sa = np.linspace(0, 0.3, 100)
    t_sa = 2 * s_sa
    ax.plot(s_sa, t_sa, 'r-', lw=2, label=r'Subadditivity: $t = 2s$')

    # 3. SSA: u = 2t - s, i.e., sqrt(1-3s²-3t²) = 2t - s (blue)
    # Solve: 1 - 3s² - 3t² = (2t - s)²
    # 1 - 3s² - 3t² = 4t² - 4st + s²
    # 1 = 4s² + 7t² - 4st
    s_ssa = []
    t_ssa = []
    for s_val in np.linspace(0.001, 0.5, 200):
        # 7t² - 4s*t + (4s² - 1) = 0
        a, b, c = 7, -4*s_val, 4*s_val**2 - 1
        disc = b**2 - 4*a*c
        if disc >= 0:
            t_val = (-b + np.sqrt(disc)) / (2*a)
            if t_val > 0 and t_val < 0.6:
                # Check validity: u >= 0 and 2t - s >= 0
                u_sq = 1 - 3*s_val**2 - 3*t_val**2
                if u_sq >= 0 and 2*t_val - s_val >= 0:
                    s_ssa.append(s_val)
                    t_ssa.append(t_val)
    if s_ssa:
        ax.plot(s_ssa, t_ssa, 'b-', lw=2, label=r'SSA: $u = 2t - s$')

    # 4. MMI: u = 3(t - s), i.e., sqrt(1-3s²-3t²) = 3(t - s) (magenta)
    # 1 - 3s² - 3t² = 9(t - s)²
    # 1 - 3s² - 3t² = 9t² - 18st + 9s²
    # 1 = 12s² + 12t² - 18st
    s_mmi = []
    t_mmi = []
    for s_val in np.linspace(0.001, 0.4, 200):
        # 12t² - 18s*t + (12s² - 1) = 0
        a, b, c = 12, -18*s_val, 12*s_val**2 - 1
        disc = b**2 - 4*a*c
        if disc >= 0:
            t_val = (-b + np.sqrt(disc)) / (2*a)
            if t_val > s_val and t_val < 0.6:  # MMI requires t > s for boundary
                u_sq = 1 - 3*s_val**2 - 3*t_val**2
                if u_sq >= 0:
                    s_mmi.append(s_val)
                    t_mmi.append(t_val)
    if s_mmi:
        ax.plot(s_mmi, t_mmi, 'm-', lw=2, label=r'MMI: $u = 3(t-s)$')

    # Labels and formatting
    ax.set_xlabel(r'$s = S_A = S_B = S_C$', fontsize=14)
    ax.set_ylabel(r'$t = S_{AB} = S_{AC} = S_{BC}$', fontsize=14)
    if use_log_scale:
        ax.set_title(r'RL Reward Landscape: $-\log(1-r)$' + '\n(5 runs per point, max reward)', fontsize=14)
    else:
        ax.set_title('RL Reward Landscape: N=3, Symmetric Slice\n(5 runs per point, max reward shown)', fontsize=14)

    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)

    plt.tight_layout()
    suffix = '_log' if use_log_scale else ''
    plt.savefig(output_dir / f'rl_reward_landscape_v2{suffix}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'rl_reward_landscape_v2{suffix}.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / f'rl_reward_landscape_v2{suffix}.png'}")


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
    print(f"Loaded {len(results)} points")

    print("Generating plots...")
    plot_reward_landscape(results, output_dir, use_log_scale=False)
    plot_reward_landscape(results, output_dir, use_log_scale=True)

    print("Done!")


if __name__ == "__main__":
    main()
