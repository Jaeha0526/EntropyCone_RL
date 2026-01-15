#!/usr/bin/env python3
"""
Comprehensive analysis of SA cone ray classification across different N values.

Aggregates results from N=9-20 multirun experiments:
- N=9-15: Read from results.json (completed experiments)
- N=16-20: Read from progressive_results.json (in-progress experiments)

Pools all data together and generates statistics and plots.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def load_valid_rays():
    """Load list of valid HEC rays from sanity check."""
    valid_rays_file = Path("extremal_rays/n6/from_SA_cone/sanity_check/valid_HEC_rays_ALL_S7.txt")

    valid_rays = set()
    with open(valid_rays_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ray_idx = int(line)
                valid_rays.add(ray_idx)

    print(f"Loaded {len(valid_rays)} valid HEC rays from sanity check")
    return valid_rays

def load_rays_of_interest():
    """Load the 6 rays of interest."""
    rays_file = Path("extremal_rays/n6/from_SA_cone/sanity_check/rays_of_interest.txt")

    rays = []
    with open(rays_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ray_idx = int(line)
                rays.append(ray_idx)

    print(f"Loaded {len(rays)} rays of interest")
    return rays

def extract_results_for_N(n_value, n_rays=208, use_progressive=False):
    """
    Extract multi-run statistics for a given N value.
    Checks multiple experiment directories (multirun10, multirun10_2, multirun10_3)
    and pools all found results for each ray.

    Args:
        n_value: Number of parties
        n_rays: Total number of rays
        use_progressive: If True, read from progressive_results.json

    Returns:
        dict: {ray_index: {'max_reward', 'mean_reward', 'std_reward', 'all_rewards', 'num_runs', 'total_runs'}}
    """
    # Check multiple experiment directories
    dir_suffixes = ["multirun10", "multirun10_2", "multirun10_3", "multirun10_4", "multirun10_5"]
    results_file = "progressive_results.json" if use_progressive else "results.json"

    # Pool rewards from all directories for each ray
    ray_pools = defaultdict(list)  # ray_idx -> list of all rewards from all directories

    for suffix in dir_suffixes:
        results_dir = Path(f"results/sa_cone/sa_cone_N{n_value}_experiment_{suffix}")

        if not results_dir.exists():
            continue

        for ray_idx in range(1, n_rays + 1):
            ray_dir = results_dir / f"sa_ray{ray_idx}_N{n_value}"
            data_file = ray_dir / results_file

            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)

                    if use_progressive:
                        # Progressive results format
                        rewards = data.get('all_rewards', [])
                        if rewards:
                            ray_pools[ray_idx].extend(rewards)
                    else:
                        # Final results format
                        if 'multi_run_statistics' in data:
                            stats = data['multi_run_statistics']
                            rewards = stats.get('all_rewards', [])
                            if rewards:
                                ray_pools[ray_idx].extend(rewards)

    # Calculate combined statistics for each ray
    ray_data = {}
    for ray_idx, all_rewards in ray_pools.items():
        if all_rewards:
            ray_data[ray_idx] = {
                'max_reward': max(all_rewards),
                'mean_reward': np.mean(all_rewards),
                'std_reward': np.std(all_rewards),
                'all_rewards': all_rewards,
                'num_runs': len(all_rewards),
                'total_runs': len(all_rewards)
            }

    return ray_data

def create_pooled_scatter_plots(n_values, valid_rays, rays_of_interest, output_dir):
    """Create scatter plots pooling all data from all N values together."""
    output_dir = Path(output_dir)

    # Pool all rewards from all N experiments for each ray
    ray_pools = defaultdict(list)  # ray_idx -> list of all rewards from all N

    for n in n_values:
        use_progressive = (n >= 16)
        ray_data = extract_results_for_N(n, use_progressive=use_progressive)

        for ray_idx, data in ray_data.items():
            if data['all_rewards']:
                # Add all individual rewards from this N to the pool
                ray_pools[ray_idx].extend(data['all_rewards'])

    # Calculate pooled statistics for each ray
    pooled_data = []
    for ray_idx, all_rewards in ray_pools.items():
        pooled_data.append({
            'ray': ray_idx,
            'max': np.max(all_rewards),
            'mean': np.mean(all_rewards),
            'std': np.std(all_rewards),
            'num_samples': len(all_rewards),
            'is_valid': ray_idx in valid_rays,
            'is_interest': ray_idx in rays_of_interest
        })

    # Convert to arrays for plotting
    ray_indices = np.array([d['ray'] for d in pooled_data])
    max_rewards = np.array([d['max'] for d in pooled_data])
    mean_rewards = np.array([d['mean'] for d in pooled_data])
    std_rewards = np.array([d['std'] for d in pooled_data])
    num_samples = np.array([d['num_samples'] for d in pooled_data])
    is_valid = np.array([d['is_valid'] for d in pooled_data], dtype=bool)
    is_interest = np.array([d['is_interest'] for d in pooled_data], dtype=bool)
    is_ray_180 = ray_indices == 180

    # Transform rewards to log(1-reward) space for better visualization
    def transform_reward(r):
        return np.log(1 - np.clip(r, 0, 1 - 1e-15) + 1e-15)

    max_transformed = transform_reward(max_rewards)
    mean_transformed = transform_reward(mean_rewards)

    # Calculate asymmetric error bars in transformed space (absolute values since log gives negative)
    upper_errors = np.abs(transform_reward(np.clip(mean_rewards + std_rewards, 0, 1-1e-15)) - mean_transformed)
    lower_errors = np.abs(mean_transformed - transform_reward(np.clip(mean_rewards - std_rewards, 0, 1-1e-15)))

    # Separate valid and spurious rays
    valid_mask = is_valid
    spurious_mask = ~is_valid

    # Plot 1: Max reward scatter
    plt.figure(figsize=(16, 8))
    plt.scatter(ray_indices[valid_mask], max_transformed[valid_mask],
               c='blue', marker='o', s=50, alpha=0.6, label='Valid HEC rays')
    if spurious_mask.any():
        plt.scatter(ray_indices[spurious_mask], max_transformed[spurious_mask],
                   c='red', marker='x', s=50, alpha=0.6, label='Spurious rays')

    # Highlight rays of interest with stars (excluding ray 180)
    interest_not_180 = is_interest & ~is_ray_180
    if interest_not_180.any():
        plt.scatter(ray_indices[interest_not_180], max_transformed[interest_not_180],
                   c='gold', marker='*', s=200, edgecolors='black', linewidths=1,
                   label='5 Rays of Interest', zorder=5)

    # Highlight ray 180 with different color star
    if is_ray_180.any():
        plt.scatter(ray_indices[is_ray_180], max_transformed[is_ray_180],
                   c='magenta', marker='*', s=200, edgecolors='black', linewidths=1,
                   label='Ray 180', zorder=6)

    plt.axhline(y=transform_reward(0.99), color='black', linestyle='--', linewidth=2, label='Threshold (0.99)')
    plt.xlabel('SA Cone Ray Index', fontsize=14)
    plt.ylabel('-log₁₀(1 - max_reward)', fontsize=14)
    plt.title('SA Cone Classification (Pooled All N): Max Reward by Ray', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'sa_cone_classification_all_N_max_pooled.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()

    # Plot 2: Max reward scatter (sorted in reverse - best rewards on right)
    sorted_indices = np.argsort(max_transformed)[::-1]  # Reverse sort
    valid_sorted_mask = is_valid[sorted_indices]
    interest_sorted_mask = is_interest[sorted_indices]

    plt.figure(figsize=(16, 10))
    x_positions = np.arange(len(sorted_indices))
    plt.scatter(x_positions[valid_sorted_mask], max_transformed[sorted_indices][valid_sorted_mask],
               c='blue', marker='o', s=60, alpha=0.8, edgecolors='darkblue', linewidths=0.5,
               label='realizable')
    plt.scatter(x_positions[~valid_sorted_mask], max_transformed[sorted_indices][~valid_sorted_mask],
               c='red', marker='x', s=60, alpha=0.8,
               label='not realizable')

    # Highlight all 6 mystery rays with gold stars
    if interest_sorted_mask.any():
        plt.scatter(x_positions[interest_sorted_mask], max_transformed[sorted_indices][interest_sorted_mask],
                   c='gold', marker='*', s=300, edgecolors='black', linewidths=1,
                   label='6 mystery rays', zorder=5)

        # Add arrow annotations with ray numbers for each star
        ray_indices_sorted = ray_indices[sorted_indices]
        interest_x = x_positions[interest_sorted_mask]
        interest_y = max_transformed[sorted_indices][interest_sorted_mask]
        interest_ray_nums = ray_indices_sorted[interest_sorted_mask]

        for i, (x, y, ray_num) in enumerate(zip(interest_x, interest_y, interest_ray_nums)):
            # Custom positions to avoid overlap with legend and each other
            # Direction is where the text/arrow comes FROM relative to the star
            if ray_num == 180:
                xytext = (x - 30, y + 4.0)  # from upper-left (good)
            elif ray_num == 181:
                xytext = (x + 30, y)  # from (1, 0) direction - horizontal right (good)
            elif ray_num == 146:
                xytext = (x - 30, y - 3.5)  # (good)
            elif ray_num == 110:
                xytext = (x + 45, y + 1.5)  # from (3, 1) direction - right, slightly above
            elif ray_num == 168:
                xytext = (x - 25, y - 3.0)  # from (-1, -1) direction - bottom-left
            elif ray_num == 145:
                xytext = (x + 25, y - 3.0)  # from (1, -1) direction - bottom-right
            else:
                xytext = (x + 25, y - 3.0)
            plt.annotate(f'Ray {ray_num}', xy=(x, y), xytext=xytext,
                        fontsize=20, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='black', lw=2.5,
                                       shrinkA=0, shrinkB=12),
                        ha='center', va='center')

    plt.xlabel('Rays (sorted by reward)', fontsize=32, labelpad=20)
    plt.ylabel(r'$\log(1 - \mathrm{reward})$', fontsize=32, labelpad=20)
    plt.legend(fontsize=24, loc='lower left', framealpha=0.9)
    plt.tick_params(axis='both', labelsize=22)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'sa_cone_classification_all_N_max_sorted_pooled.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()

    # Plot 3: Mean reward with error bars
    plt.figure(figsize=(16, 8))
    plt.errorbar(ray_indices[valid_mask], mean_transformed[valid_mask],
                yerr=[lower_errors[valid_mask], upper_errors[valid_mask]],
                fmt='o', color='blue', markersize=6, capsize=3,
                alpha=0.6, label='Valid HEC rays')
    if spurious_mask.any():
        plt.errorbar(ray_indices[spurious_mask], mean_transformed[spurious_mask],
                    yerr=[lower_errors[spurious_mask], upper_errors[spurious_mask]],
                    fmt='x', color='red', markersize=6, capsize=3,
                    alpha=0.6, label='Spurious rays')

    # Highlight rays of interest with stars (excluding ray 180)
    interest_not_180 = is_interest & ~is_ray_180
    if interest_not_180.any():
        plt.scatter(ray_indices[interest_not_180], mean_transformed[interest_not_180],
                   c='gold', marker='*', s=200, edgecolors='black', linewidths=1,
                   label='5 Rays of Interest', zorder=5)

    # Highlight ray 180 with different color star
    if is_ray_180.any():
        plt.scatter(ray_indices[is_ray_180], mean_transformed[is_ray_180],
                   c='magenta', marker='*', s=200, edgecolors='black', linewidths=1,
                   label='Ray 180', zorder=6)

    plt.axhline(y=transform_reward(0.99), color='black', linestyle='--', linewidth=2, label='Threshold (0.99)')
    plt.xlabel('SA Cone Ray Index', fontsize=14)
    plt.ylabel('-log₁₀(1 - mean_reward)', fontsize=14)
    plt.title('SA Cone Classification (Pooled All N): Mean Reward by Ray', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'sa_cone_classification_all_N_mean_pooled.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()

    # Plot 4: Mean reward with error bars (sorted)
    sorted_indices = np.argsort(mean_transformed)
    valid_sorted_mask = is_valid[sorted_indices]
    interest_sorted_mask = is_interest[sorted_indices]

    plt.figure(figsize=(16, 8))
    x_positions = np.arange(len(sorted_indices))
    plt.errorbar(x_positions[valid_sorted_mask], mean_transformed[sorted_indices][valid_sorted_mask],
                yerr=[lower_errors[sorted_indices][valid_sorted_mask], upper_errors[sorted_indices][valid_sorted_mask]],
                fmt='o', color='blue', markersize=6, capsize=3,
                alpha=0.6, label='Valid HEC rays')
    plt.errorbar(x_positions[~valid_sorted_mask], mean_transformed[sorted_indices][~valid_sorted_mask],
                yerr=[lower_errors[sorted_indices][~valid_sorted_mask], upper_errors[sorted_indices][~valid_sorted_mask]],
                fmt='x', color='red', markersize=6, capsize=3,
                alpha=0.6, label='Spurious rays')

    # Highlight rays of interest with stars (excluding ray 180)
    ray_180_sorted_mask = is_ray_180[sorted_indices]
    interest_not_180_sorted = interest_sorted_mask & ~ray_180_sorted_mask
    if interest_not_180_sorted.any():
        plt.scatter(x_positions[interest_not_180_sorted], mean_transformed[sorted_indices][interest_not_180_sorted],
                   c='gold', marker='*', s=200, edgecolors='black', linewidths=1,
                   label='5 Rays of Interest', zorder=5)

    # Highlight ray 180 with different color star
    if ray_180_sorted_mask.any():
        plt.scatter(x_positions[ray_180_sorted_mask], mean_transformed[sorted_indices][ray_180_sorted_mask],
                   c='magenta', marker='*', s=200, edgecolors='black', linewidths=1,
                   label='Ray 180', zorder=6)

    plt.axhline(y=transform_reward(0.99), color='black', linestyle='--', linewidth=2, label='Threshold (0.99)')
    plt.xlabel('Ray Index (sorted by reward)', fontsize=14)
    plt.ylabel('-log₁₀(1 - mean_reward)', fontsize=14)
    plt.title('SA Cone Classification (Pooled All N): Mean Reward Sorted', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'sa_cone_classification_all_N_mean_sorted_pooled.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()

    # Print pooled statistics
    print(f"\nPooled statistics across all N:")
    print(f"  Total rays with data: {len(pooled_data)}")
    print(f"  Valid HEC rays: {valid_mask.sum()}")
    print(f"  Spurious rays: {spurious_mask.sum()}")
    print(f"  Total samples: {num_samples.sum()}")
    print(f"  Valid rays - Avg max: {max_rewards[valid_mask].mean():.6f} ± {max_rewards[valid_mask].std():.6f}")
    if spurious_mask.any():
        print(f"  Spurious rays - Avg max: {max_rewards[spurious_mask].mean():.6f} ± {max_rewards[spurious_mask].std():.6f}")

    # Classification analysis
    threshold = 0.99
    valid_above = (max_rewards[valid_mask] >= threshold).sum()
    spurious_above = (max_rewards[spurious_mask] >= threshold).sum() if spurious_mask.any() else 0

    print(f"\nClassification (threshold = {threshold}):")
    print(f"  Valid rays >= {threshold}: {valid_above}/{valid_mask.sum()} ({valid_above/valid_mask.sum()*100:.1f}%)")
    if spurious_mask.any():
        print(f"  Spurious rays >= {threshold}: {spurious_above}/{spurious_mask.sum()} ({spurious_above/spurious_mask.sum()*100:.1f}%)")

def main():
    print("="*70)
    print("SA CONE CLASSIFICATION: POOLED ANALYSIS ACROSS N VALUES")
    print("="*70)

    # N values to analyze
    n_values = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Create output directory
    output_dir = Path("extremal_rays/n6/from_SA_cone/analyze")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load valid rays and rays of interest
    valid_rays = load_valid_rays()
    rays_of_interest = load_rays_of_interest()

    # Generate pooled scatter plots
    print("\n" + "="*70)
    print("GENERATING POOLED SCATTER PLOTS (ALL DATA FROM ALL N)")
    print("="*70)
    create_pooled_scatter_plots(n_values, valid_rays, rays_of_interest, output_dir)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nGenerated files in {output_dir}:")
    print("  Pooled scatter plots (all data from all N combined):")
    print("    - sa_cone_classification_all_N_max_pooled.png")
    print("    - sa_cone_classification_all_N_max_sorted_pooled.png")
    print("    - sa_cone_classification_all_N_mean_pooled.png")
    print("    - sa_cone_classification_all_N_mean_sorted_pooled.png")

if __name__ == "__main__":
    main()
