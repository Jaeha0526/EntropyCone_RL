#!/usr/bin/env python3
"""
3D Reward Landscape Plot for Section 3.3

Draws:
- Analytical reward as a smooth surface
- RL data points at grid locations
- Two versions: mean with error bars, max reward
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
from joblib import Parallel, delayed

# Add visualization directory to path for importing the analytical formula
sys.path.insert(0, str(Path(__file__).parent.parent / 'visualization'))
from plot_reward_landscape_analytic import compute_reward_analytic

# Grid parameters
GRID_SIZE = 20
S_MIN, S_MAX = 0.02, 0.55
T_MIN, T_MAX = 0.02, 0.55


def load_rl_results(results_dir):
    """Load RL results from grid validation."""
    results = []

    for point_idx in range(1, GRID_SIZE * GRID_SIZE + 1):
        point_dir = results_dir / f"point_{point_idx}"
        metadata_file = point_dir / "grid_point_metadata.json"
        results_file = point_dir / "results.json"

        if not metadata_file.exists():
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        s, t = metadata['s'], metadata['t']

        if results_file.exists():
            with open(results_file) as f:
                result_data = json.load(f)

            if 'multi_run_statistics' in result_data:
                stats = result_data['multi_run_statistics']
                max_reward = stats['max_reward']
                mean_reward = stats['mean_reward']
                std_reward = stats['std_reward']
            else:
                max_reward = result_data.get('best_optimized_reward', np.nan)
                mean_reward = max_reward
                std_reward = 0.0
        else:
            continue

        results.append({
            's': s,
            't': t,
            'max_reward': max_reward,
            'mean_reward': mean_reward,
            'std_reward': std_reward
        })

    return results


def load_rl_results_combined(results_dirs):
    """
    Load and combine RL results from multiple grid validation experiments.

    Combines individual run rewards from all directories to compute
    pooled statistics (max, mean, std across all runs).

    Args:
        results_dirs: List of Path objects to results directories

    Returns:
        List of dicts with combined statistics per grid point
    """
    results = []

    for point_idx in range(1, GRID_SIZE * GRID_SIZE + 1):
        all_rewards = []
        s, t = None, None

        for results_dir in results_dirs:
            point_dir = results_dir / f"point_{point_idx}"
            metadata_file = point_dir / "grid_point_metadata.json"
            results_file = point_dir / "results.json"

            if not metadata_file.exists() or not results_file.exists():
                continue

            # Get coordinates from first valid directory
            if s is None:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                s, t = metadata['s'], metadata['t']

            # Load individual run rewards
            with open(results_file) as f:
                result_data = json.load(f)

            if 'multi_run_statistics' in result_data:
                stats = result_data['multi_run_statistics']
                if 'all_rewards' in stats:
                    all_rewards.extend(stats['all_rewards'])
                else:
                    # Fallback: use the stats we have (less accurate for combined std)
                    all_rewards.append(stats['max_reward'])
            else:
                reward = result_data.get('best_optimized_reward')
                if reward is not None:
                    all_rewards.append(reward)

        # Skip if no valid data for this point
        if s is None or len(all_rewards) == 0:
            continue

        # Compute pooled statistics
        all_rewards = np.array(all_rewards)
        results.append({
            's': s,
            't': t,
            'max_reward': float(np.max(all_rewards)),
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'n_runs': len(all_rewards)
        })

    return results


def plot_3d_landscape(results, output_dir, version='mean'):
    """
    Create 3D plot with analytical surface and RL data points.

    version: 'mean' for mean with error bars, 'max' for max reward
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create fine grid for analytical surface
    s_fine = np.linspace(S_MIN, S_MAX, 100)
    t_fine = np.linspace(T_MIN, T_MAX, 100)
    S_grid, T_grid = np.meshgrid(s_fine, t_fine)

    # Compute analytical reward on fine grid using correct formula
    # Use NaN for points outside the valid disk (3s² + 3t² >= 1)
    Z_analytical = np.zeros_like(S_grid)
    for i in range(S_grid.shape[0]):
        for j in range(S_grid.shape[1]):
            s, t = S_grid[i, j], T_grid[i, j]
            if 3*s**2 + 3*t**2 >= 1:
                Z_analytical[i, j] = np.nan  # Outside valid disk
            else:
                Z_analytical[i, j] = compute_reward_analytic(s, t)

    # Plot analytical surface (semi-transparent, single color)
    surf = ax.plot_surface(S_grid, T_grid, Z_analytical,
                           color='lightblue', alpha=0.5,
                           linewidth=0, antialiased=True,
                           label='Analytical')

    # Extract RL data points
    s_vals = np.array([r['s'] for r in results])
    t_vals = np.array([r['t'] for r in results])

    if version == 'mean':
        z_vals = np.array([r['mean_reward'] for r in results])
        z_err = np.array([r['std_reward'] for r in results])
        title = 'Reward Landscape: Analytical Surface vs RL Mean (with std)'
        filename = 'reward_landscape_3d_mean.png'
        point_label = 'RL Mean'
    else:  # max
        z_vals = np.array([r['max_reward'] for r in results])
        z_err = None
        title = 'Reward Landscape: Analytical Surface vs RL Max'
        filename = 'reward_landscape_3d_max.png'
        point_label = 'RL Max'

    # Plot RL data points
    scatter = ax.scatter(s_vals, t_vals, z_vals,
                        c='red', s=30, alpha=0.8,
                        label=point_label, depthshade=True)

    # Add error bars for mean version
    if version == 'mean' and z_err is not None:
        for i in range(len(s_vals)):
            if z_err[i] > 0.001:  # Only draw if error is significant
                ax.plot([s_vals[i], s_vals[i]],
                       [t_vals[i], t_vals[i]],
                       [z_vals[i] - z_err[i], z_vals[i] + z_err[i]],
                       'r-', alpha=0.3, linewidth=1)

    # Labels and title
    ax.set_xlabel('s', fontsize=12, labelpad=10)
    ax.set_ylabel('t', fontsize=12, labelpad=10)
    ax.set_zlabel('Reward', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=20)

    # Set axis limits
    ax.set_xlim(S_MIN, S_MAX)
    ax.set_ylim(T_MIN, T_MAX)
    ax.set_zlim(0.5, 1.05)

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    # Set viewing angle - from (1,1,1) towards (0,0,0) to see valley
    ax.view_init(elev=35, azim=45)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / filename}")

    # Also save PDF version
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(S_grid, T_grid, Z_analytical,
                           color='lightblue', alpha=0.5,
                           linewidth=0, antialiased=True)

    scatter = ax.scatter(s_vals, t_vals, z_vals,
                        c='red', s=30, alpha=0.8,
                        depthshade=True)

    if version == 'mean' and z_err is not None:
        for i in range(len(s_vals)):
            if z_err[i] > 0.001:
                ax.plot([s_vals[i], s_vals[i]],
                       [t_vals[i], t_vals[i]],
                       [z_vals[i] - z_err[i], z_vals[i] + z_err[i]],
                       'r-', alpha=0.3, linewidth=1)

    ax.set_xlabel('s', fontsize=12, labelpad=10)
    ax.set_ylabel('t', fontsize=12, labelpad=10)
    ax.set_zlabel('Reward', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlim(S_MIN, S_MAX)
    ax.set_ylim(T_MIN, T_MAX)
    ax.set_zlim(0.5, 1.05)
    ax.view_init(elev=35, azim=45)

    plt.tight_layout()
    pdf_filename = filename.replace('.png', '.pdf')
    plt.savefig(output_dir / pdf_filename, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / pdf_filename}")


def plot_3d_landscape_wireframe(results, output_dir, version='mean'):
    """Create 3D plot with wireframe surface for better visibility."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create fine grid for analytical surface
    s_fine = np.linspace(S_MIN, S_MAX, 50)
    t_fine = np.linspace(T_MIN, T_MAX, 50)
    S_grid, T_grid = np.meshgrid(s_fine, t_fine)

    Z_analytical = np.zeros_like(S_grid)
    for i in range(S_grid.shape[0]):
        for j in range(S_grid.shape[1]):
            s, t = S_grid[i, j], T_grid[i, j]
            if 3*s**2 + 3*t**2 >= 1:
                Z_analytical[i, j] = np.nan
            else:
                Z_analytical[i, j] = compute_reward_analytic(s, t)

    # Plot wireframe for analytical (better visibility)
    ax.plot_wireframe(S_grid, T_grid, Z_analytical,
                      color='blue', alpha=0.3, linewidth=0.5,
                      rstride=2, cstride=2)

    # Extract RL data
    s_vals = np.array([r['s'] for r in results])
    t_vals = np.array([r['t'] for r in results])

    if version == 'mean':
        z_vals = np.array([r['mean_reward'] for r in results])
        z_err = np.array([r['std_reward'] for r in results])
        title = 'Reward Landscape (Wireframe): Analytical vs RL Mean'
        filename = 'reward_landscape_3d_wireframe_mean.png'
    else:
        z_vals = np.array([r['max_reward'] for r in results])
        z_err = None
        title = 'Reward Landscape (Wireframe): Analytical vs RL Max'
        filename = 'reward_landscape_3d_wireframe_max.png'

    # Color points by error (difference from analytical)
    analytical_at_points = np.array([compute_reward_analytic(r['s'], r['t']) for r in results])
    errors = z_vals - analytical_at_points

    scatter = ax.scatter(s_vals, t_vals, z_vals,
                        c=errors, cmap='RdBu_r', s=40, alpha=0.9,
                        vmin=-0.05, vmax=0.05,
                        depthshade=True)

    # Error bars for mean version
    if version == 'mean' and z_err is not None:
        for i in range(len(s_vals)):
            if z_err[i] > 0.001:
                ax.plot([s_vals[i], s_vals[i]],
                       [t_vals[i], t_vals[i]],
                       [z_vals[i] - z_err[i], z_vals[i] + z_err[i]],
                       'k-', alpha=0.3, linewidth=1)

    ax.set_xlabel('s', fontsize=12, labelpad=10)
    ax.set_ylabel('t', fontsize=12, labelpad=10)
    ax.set_zlabel('Reward', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlim(S_MIN, S_MAX)
    ax.set_ylim(T_MIN, T_MAX)
    ax.set_zlim(0.5, 1.05)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Error (RL - Analytical)', fontsize=10)

    ax.view_init(elev=35, azim=45)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / filename}")


def find_contour_angle_at_radius_sa_side(r, target_log_eps=-10):
    """Find the angle where log(eps) = target_log_eps on SA-violating side.

    The -10 contour is just outside SA boundary (θ slightly > atan(2) ≈ 63.4°).
    """
    theta_sa = np.arctan(2)
    theta_low = theta_sa + 0.001
    theta_high = np.pi/2 - 0.001

    for _ in range(50):
        theta_mid = (theta_low + theta_high) / 2
        s = r * np.cos(theta_mid)
        t = r * np.sin(theta_mid)

        reward = compute_reward_analytic(s, t)
        eps = 1 - reward
        log_eps = np.log(max(eps, 1e-15))

        if log_eps < target_log_eps:
            theta_low = theta_mid  # Too close to HEC, need larger angle
        else:
            theta_high = theta_mid  # Too far from HEC, need smaller angle

    return theta_mid


def find_contour_angle_at_radius_mmi_side(r, target_log_eps=-10):
    """Find the angle where log(eps) = target_log_eps on MMI-violating side.

    At small θ (near θ=0), MMI is violated: u > 3(t-s) when t < s.
    As θ increases toward the SA boundary (~63.4°), we approach HEC and log(eps) → -∞.
    The -10 contour is at intermediate angles where log(eps) = -10.
    """
    theta_sa = np.arctan(2)
    theta_low = 0.001
    theta_high = theta_sa - 0.001

    # Check if -10 contour exists: need log_eps > -10 at one end and < -10 at other end
    # At theta_low (near 0°): log_eps ≈ -1 (far from HEC, high epsilon)
    # At theta_high (near SA boundary): log_eps → -∞ if inside HEC

    s_high, t_high = r * np.cos(theta_high), r * np.sin(theta_high)
    reward_high = compute_reward_analytic(s_high, t_high)

    if reward_high < 1.0 - 1e-10:
        # Not inside HEC at theta_high, check log_eps
        eps_high = 1 - reward_high
        log_eps_high = np.log(max(eps_high, 1e-15))
        if log_eps_high > target_log_eps:
            return None  # log_eps never reaches -10 on this side
    # else: inside HEC at theta_high, so log_eps crosses -10 somewhere

    # Binary search for the -10 contour
    for _ in range(50):
        theta_mid = (theta_low + theta_high) / 2
        s = r * np.cos(theta_mid)
        t = r * np.sin(theta_mid)

        reward = compute_reward_analytic(s, t)
        if reward >= 1.0 - 1e-15:
            # Inside HEC, need smaller angle (toward MMI violation)
            theta_high = theta_mid
        else:
            eps = 1 - reward
            log_eps = np.log(max(eps, 1e-15))
            if log_eps < target_log_eps:
                theta_high = theta_mid  # Too close to HEC, need smaller angle
            else:
                theta_low = theta_mid  # Too far from HEC, need larger angle

    return theta_mid


def compute_and_cache_analytical_surface(output_dir, n_angular=1200, n_radial=900):
    """
    Compute analytical reward surface and cache to file.
    Returns cached data if available with matching resolution.

    Grid structure (no overlap):
    - For small radii (r < r_contour_min): full angular range 0° to 90°
    - For larger radii (r >= r_contour_min where -10 contour exists):
      - Left part: 0° to θ_mmi (MMI-side boundary), last point set to z=-10
      - Right part: θ_sa (SA-side boundary) to 90°, first point set to z=-10
      - Middle part (inside HEC) is skipped
    """
    cache_file = output_dir / f"analytical_cache_{n_angular}x{n_radial}_v5.npz"

    # Check if cache exists
    if cache_file.exists():
        print(f"  Loading cached data from {cache_file.name}...")
        data = np.load(cache_file)
        return data

    print(f"  Computing analytical surface ({n_angular}x{n_radial} split grid)...")

    r_max = 1 / np.sqrt(3)
    r_contour_min = 0.454  # Radius where -10 contour starts

    # ========== Radial spacing ==========
    # Inner grid: from small r up to AND INCLUDING r_contour_min
    # Outer grid: from r_contour_min (inclusive) to r_max
    # They share the boundary at r_contour_min for smooth connection
    n_rad_inner = n_radial // 3
    n_rad_outer = n_radial - n_rad_inner
    r_inner = np.linspace(0.01, r_contour_min, n_rad_inner, endpoint=True)  # Include boundary
    r_outer = np.linspace(r_contour_min, r_max * 0.9999, n_rad_outer)  # Also starts at boundary

    print(f"    Radii: {n_rad_inner} inner (full angular, r≤{r_contour_min}) + {n_rad_outer} outer (split angular)")

    # ========== PART 1: Inner radii - full angular range ==========
    # Non-uniform angular spacing
    theta_sa = np.arctan(2)
    n_ang_sparse = n_angular // 3
    n_ang_dense = n_angular - n_ang_sparse
    theta_sparse = np.linspace(0.001, theta_sa, n_ang_sparse, endpoint=False)
    theta_dense = np.linspace(theta_sa, np.pi/2 - 0.001, n_ang_dense)
    theta_full = np.concatenate([theta_sparse, theta_dense])

    print(f"    Part 1 (inner, full angular): {len(r_inner)} radii x {len(theta_full)} angles")

    Theta_inner, R_inner = np.meshgrid(theta_full, r_inner)
    S_inner = R_inner * np.cos(Theta_inner)
    T_inner = R_inner * np.sin(Theta_inner)

    # ========== PART 2: Outer radii - split into left (MMI) and right (SA) ==========
    print(f"    Part 2 (outer, split): Finding -10 contour angles...")

    # Find contour angles for each outer radius
    contour_angles_sa = []
    contour_angles_mmi = []

    for r in r_outer:
        # SA side contour (always exists for outer radii)
        theta_sa_contour = find_contour_angle_at_radius_sa_side(r, target_log_eps=-10)
        contour_angles_sa.append(theta_sa_contour)

        # MMI side contour (may not exist for all radii)
        theta_mmi_contour = find_contour_angle_at_radius_mmi_side(r, target_log_eps=-10)
        contour_angles_mmi.append(theta_mmi_contour)  # None if doesn't exist

    contour_angles_sa = np.array(contour_angles_sa)

    # Count how many radii have MMI contour
    n_with_mmi = sum(1 for x in contour_angles_mmi if x is not None)
    print(f"    Found {n_with_mmi}/{len(r_outer)} radii with MMI-side -10 contour")

    # Build left part (0° to MMI contour) and right part (SA contour to 90°)
    n_ang_per_side = n_angular // 4  # Points per side for outer radii

    S_left_list, T_left_list, Z_left_boundary = [], [], []
    S_right_list, T_right_list, Z_right_boundary = [], [], []

    for i, r in enumerate(r_outer):
        theta_sa_c = contour_angles_sa[i]
        theta_mmi_c = contour_angles_mmi[i]

        # Right part: SA contour to 90°
        theta_right = np.linspace(theta_sa_c, np.pi/2 - 0.001, n_ang_per_side)
        for j, theta in enumerate(theta_right):
            S_right_list.append(r * np.cos(theta))
            T_right_list.append(r * np.sin(theta))
            Z_right_boundary.append(j == 0)  # First point is on -10 contour

        # Left part: 0° to MMI contour (if exists)
        if theta_mmi_c is not None:
            theta_left = np.linspace(0.001, theta_mmi_c, n_ang_per_side)
            for j, theta in enumerate(theta_left):
                S_left_list.append(r * np.cos(theta))
                T_left_list.append(r * np.sin(theta))
                Z_left_boundary.append(j == n_ang_per_side - 1)  # Last point is on -10 contour

    S_right = np.array(S_right_list)
    T_right = np.array(T_right_list)
    is_right_boundary = np.array(Z_right_boundary)

    S_left = np.array(S_left_list) if S_left_list else np.array([])
    T_left = np.array(T_left_list) if T_left_list else np.array([])
    is_left_boundary = np.array(Z_left_boundary) if Z_left_boundary else np.array([])

    print(f"    Right part (SA side): {len(S_right)} points ({np.sum(is_right_boundary)} on boundary)")
    print(f"    Left part (MMI side): {len(S_left)} points ({np.sum(is_left_boundary) if len(is_left_boundary) > 0 else 0} on boundary)")

    # ========== Compute rewards ==========
    print("    Computing rewards...")

    def compute_row(i, S_grid, T_grid):
        row = np.zeros(S_grid.shape[1])
        for j in range(S_grid.shape[1]):
            s, t = S_grid[i, j], T_grid[i, j]
            if 3*s**2 + 3*t**2 >= 1:
                row[j] = np.nan
            else:
                row[j] = compute_reward_analytic(s, t)
        return row

    def compute_1d(S, T):
        Z = np.zeros(len(S))
        for i in range(len(S)):
            s, t = S[i], T[i]
            if 3*s**2 + 3*t**2 >= 1:
                Z[i] = np.nan
            else:
                Z[i] = compute_reward_analytic(s, t)
        return Z

    # Inner grid (parallel)
    print("      Inner grid...")
    reward_rows = Parallel(n_jobs=-1, verbose=0)(
        delayed(compute_row)(i, S_inner, T_inner) for i in range(S_inner.shape[0])
    )
    Z_rewards_inner = np.array(reward_rows)

    # Right and left parts
    print("      Right part (SA side)...")
    Z_rewards_right = compute_1d(S_right, T_right)

    if len(S_left) > 0:
        print("      Left part (MMI side)...")
        Z_rewards_left = compute_1d(S_left, T_left)
    else:
        Z_rewards_left = np.array([])

    # ========== Convert to log(epsilon) ==========
    def to_log_eps(Z_rewards, clip_at=-10):
        epsilon = 1 - Z_rewards
        with np.errstate(divide='ignore', invalid='ignore'):
            log_eps = np.log(epsilon)
        Z = np.where(np.isnan(Z_rewards), np.nan, log_eps)
        Z = np.where(epsilon <= 1e-15, np.nan, Z)  # Inside HEC
        Z = np.where(log_eps < clip_at, np.nan, Z)  # Clip at -10
        return Z

    Z_analytical_inner = to_log_eps(Z_rewards_inner)
    Z_analytical_right = to_log_eps(Z_rewards_right)
    Z_analytical_left = to_log_eps(Z_rewards_left) if len(Z_rewards_left) > 0 else np.array([])

    # Set boundary points to z=-10
    Z_analytical_right[is_right_boundary] = -10.0
    if len(Z_analytical_left) > 0:
        Z_analytical_left[is_left_boundary] = -10.0

    # For inner grid, mark boundary at -10 contour (last valid point per column)
    for j in range(Z_analytical_inner.shape[1]):
        valid_mask = ~np.isnan(Z_analytical_inner[:, j])
        if np.any(valid_mask):
            last_valid_idx = np.where(valid_mask)[0][-1]
            if last_valid_idx < Z_analytical_inner.shape[0] - 1:
                if np.isnan(Z_analytical_inner[last_valid_idx + 1, j]):
                    Z_analytical_inner[last_valid_idx, j] = -10.0

    # Z_for_contour: unclipped for drawing contour lines
    epsilon_inner = np.maximum(1 - Z_rewards_inner, 1e-15)
    with np.errstate(divide='ignore', invalid='ignore'):
        Z_contour_inner = np.log(epsilon_inner)
    Z_contour_inner = np.where(np.isnan(Z_rewards_inner), np.nan, Z_contour_inner)

    # Reshape to 2D for surface plotting
    S_right_2d = S_right.reshape(len(r_outer), n_ang_per_side)
    T_right_2d = T_right.reshape(len(r_outer), n_ang_per_side)
    Z_analytical_right_2d = Z_analytical_right.reshape(len(r_outer), n_ang_per_side)

    # Left part may have different number of radii (only those with MMI contour)
    n_with_mmi_actual = sum(1 for x in contour_angles_mmi if x is not None)
    if n_with_mmi_actual > 0 and len(S_left) > 0:
        S_left_2d = S_left.reshape(n_with_mmi_actual, n_ang_per_side)
        T_left_2d = T_left.reshape(n_with_mmi_actual, n_ang_per_side)
        Z_analytical_left_2d = Z_analytical_left.reshape(n_with_mmi_actual, n_ang_per_side)
    else:
        S_left_2d = np.array([])
        T_left_2d = np.array([])
        Z_analytical_left_2d = np.array([])

    # Contour line data
    contour_sa_s = r_outer * np.cos(contour_angles_sa)
    contour_sa_t = r_outer * np.sin(contour_angles_sa)

    contour_mmi_r = np.array([r_outer[i] for i, x in enumerate(contour_angles_mmi) if x is not None])
    contour_mmi_theta = np.array([x for x in contour_angles_mmi if x is not None])
    contour_mmi_s = contour_mmi_r * np.cos(contour_mmi_theta) if len(contour_mmi_r) > 0 else np.array([])
    contour_mmi_t = contour_mmi_r * np.sin(contour_mmi_theta) if len(contour_mmi_r) > 0 else np.array([])

    # ========== Save to cache ==========
    print(f"    Saving cache to {cache_file.name}...")
    np.savez_compressed(cache_file,
                        # Inner grid (full angular range)
                        S_inner=S_inner, T_inner=T_inner,
                        Z_analytical_inner=Z_analytical_inner,
                        Z_contour_inner=Z_contour_inner,
                        # Right part (SA side: θ_sa to 90°)
                        S_right=S_right_2d, T_right=T_right_2d,
                        Z_analytical_right=Z_analytical_right_2d,
                        # Left part (MMI side: 0° to θ_mmi)
                        S_left=S_left_2d, T_left=T_left_2d,
                        Z_analytical_left=Z_analytical_left_2d,
                        # Contour lines for plotting
                        contour_sa_s=contour_sa_s, contour_sa_t=contour_sa_t,
                        contour_mmi_s=contour_mmi_s, contour_mmi_t=contour_mmi_t)

    total_points = S_inner.size + S_right.size + len(S_left)
    print(f"    Total: {total_points} points (no overlap)")

    return np.load(cache_file)


def load_surface_data(output_dir, n_angular=1200, n_radial=900):
    """Load cached surface data with inner, right, and left grids."""
    cache_file = output_dir / f"analytical_cache_{n_angular}x{n_radial}_v5.npz"

    if cache_file.exists():
        print(f"  Loading cached data from {cache_file.name}...")
        data = np.load(cache_file)
        return data
    else:
        # Compute and cache (this will create the file)
        return compute_and_cache_analytical_surface(output_dir, n_angular, n_radial)


def plot_3d_landscape_log(results, output_dir, version='mean'):
    """
    Create 3D plot with log(1-reward) scale.
    This puts analytical surface below since log(1-r) < 0 for r < 1.

    Uses v5 split grid (no overlap):
    - Inner grid: Full angular range for r < 0.454
    - Right grid: SA side (θ_sa to 90°) for r >= 0.454
    - Left grid: MMI side (0° to θ_mmi) for r >= 0.454 where it exists
    - -10 contour drawn on both SA side and MMI side
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Disable automatic z-ordering so we can control draw order explicitly
    ax.computed_zorder = False

    # Load or compute analytical surface (with caching)
    n_angular = 1200
    n_radial = 900
    data = load_surface_data(output_dir, n_angular, n_radial)

    # Z-ordering: constraint lines (zorder=0) < surfaces (zorder=10) < RL points (zorder=100)

    # Draw constraint boundaries FIRST on z=-10 base plane (lowest zorder)
    z_base = -10

    # 1. Disk boundary: s² + t² = 1/3 (from 3s² + 3t² + u² = 1 with u=0)
    theta = np.linspace(0, np.pi/2, 200)
    r_disk = 1/np.sqrt(3)
    s_disk = r_disk * np.cos(theta)
    t_disk = r_disk * np.sin(theta)
    ax.plot(s_disk, t_disk, z_base * np.ones_like(s_disk), 'k-', lw=2, zorder=0)

    # 2. SA boundary: t = 2s
    s_sa = np.linspace(0, 0.3, 100)
    t_sa = 2 * s_sa
    ax.plot(s_sa, t_sa, z_base * np.ones_like(s_sa), 'r-', lw=2, zorder=0)

    # 3. SSA boundary: u = 2t - s, with u = sqrt(1-3s²-3t²)
    # Solve: 1 - 3s² - 3t² = (2t - s)² => 7t² - 4st + 4s² - 1 = 0
    s_ssa = []
    t_ssa = []
    for s_val in np.linspace(0.001, 0.5, 200):
        a, b, c = 7, -4*s_val, 4*s_val**2 - 1
        disc = b**2 - 4*a*c
        if disc >= 0:
            t_val = (-b + np.sqrt(disc)) / (2*a)
            if t_val > 0 and t_val < 0.6:
                u_sq = 1 - 3*s_val**2 - 3*t_val**2
                if u_sq >= 0 and 2*t_val - s_val >= 0:
                    s_ssa.append(s_val)
                    t_ssa.append(t_val)
    if s_ssa:
        ax.plot(s_ssa, t_ssa, z_base * np.ones(len(s_ssa)), 'b-', lw=2, zorder=0)

    # 4. MMI boundary: u = 3(t - s)
    # Solve: 1 - 3s² - 3t² = 9(t - s)² => 12t² - 18st + 12s² - 1 = 0
    s_mmi = []
    t_mmi = []
    for s_val in np.linspace(0.001, 0.4, 200):
        a, b, c = 12, -18*s_val, 12*s_val**2 - 1
        disc = b**2 - 4*a*c
        if disc >= 0:
            t_val = (-b + np.sqrt(disc)) / (2*a)
            if t_val > s_val and t_val < 0.6:
                u_sq = 1 - 3*s_val**2 - 3*t_val**2
                if u_sq >= 0:
                    s_mmi.append(s_val)
                    t_mmi.append(t_val)
    if s_mmi:
        ax.plot(s_mmi, t_mmi, z_base * np.ones(len(s_mmi)), 'm-', lw=2, zorder=0)

    # Draw explicit -10 contour lines (still on base plane, behind surfaces)
    s_contour_sa = data['contour_sa_s']
    t_contour_sa = data['contour_sa_t']
    ax.plot(s_contour_sa, t_contour_sa, -10 * np.ones_like(s_contour_sa),
            'b--', lw=2, label=r'$\log(1-\mathrm{reward})=-10$', zorder=0)

    if 'contour_mmi_s' in data and data['contour_mmi_s'].size > 0:
        s_contour_mmi = data['contour_mmi_s']
        t_contour_mmi = data['contour_mmi_t']
        ax.plot(s_contour_mmi, t_contour_mmi, -10 * np.ones_like(s_contour_mmi),
                'b--', lw=2, zorder=0)  # Same style, no extra label

    # Plot surfaces (middle zorder, above lines but below data points)
    # Plot inner surface (full angular range, r < 0.454)
    surf1 = ax.plot_surface(data['S_inner'], data['T_inner'], data['Z_analytical_inner'],
                           color='lightblue', alpha=0.7,
                           linewidth=0, antialiased=True,
                           rstride=1, cstride=1,
                           label='Analytical', zorder=10)

    # Plot right surface (SA side: θ_sa to 90°, r >= 0.454)
    surf2 = ax.plot_surface(data['S_right'], data['T_right'], data['Z_analytical_right'],
                           color='lightblue', alpha=0.7,
                           linewidth=0, antialiased=True,
                           rstride=1, cstride=1, zorder=10)

    # Plot left surface (MMI side: 0° to θ_mmi, r >= 0.454) if it exists
    if 'S_left' in data and data['S_left'].size > 0:
        surf3 = ax.plot_surface(data['S_left'], data['T_left'], data['Z_analytical_left'],
                               color='lightblue', alpha=0.7,
                               linewidth=0, antialiased=True,
                               rstride=1, cstride=1, zorder=10)

    # Keep default background (grid and colored room panes)

    # Draw grid lines on analytical surface (thin gray, 20x20 density)
    # Grid lines stop at the log(1-reward)=-10 contour (blue dotted line)
    s_grid_vals = np.linspace(S_MIN, S_MAX, GRID_SIZE)  # 20 values
    t_grid_vals = np.linspace(T_MIN, T_MAX, GRID_SIZE)  # 20 values
    z_cutoff = -9.5  # Stop grid lines before they reach -10 boundary

    # Lines along t-direction (constant s)
    for s_val in s_grid_vals:
        t_line = np.linspace(T_MIN, T_MAX, 100)  # Fine resolution for smooth line
        s_line = np.full_like(t_line, s_val)
        z_line = np.array([np.log(max(1 - compute_reward_analytic(s_val, t), 1e-15))
                           if 3*s_val**2 + 3*t**2 < 1 else np.nan for t in t_line])
        # Don't draw below cutoff (stop at the -10 contour boundary)
        z_line = np.where(z_line < z_cutoff, np.nan, z_line)
        ax.plot(s_line, t_line, z_line, color='gray', linewidth=0.5, alpha=0.6, zorder=50)

    # Lines along s-direction (constant t)
    for t_val in t_grid_vals:
        s_line = np.linspace(S_MIN, S_MAX, 100)
        t_line = np.full_like(s_line, t_val)
        z_line = np.array([np.log(max(1 - compute_reward_analytic(s, t_val), 1e-15))
                           if 3*s**2 + 3*t_val**2 < 1 else np.nan for s in s_line])
        z_line = np.where(z_line < z_cutoff, np.nan, z_line)
        ax.plot(s_line, t_line, z_line, color='gray', linewidth=0.5, alpha=0.6, zorder=50)

    # Extract RL data points
    s_vals = np.array([r['s'] for r in results])
    t_vals = np.array([r['t'] for r in results])

    if version == 'mean':
        z_vals = np.array([r['mean_reward'] for r in results])
        z_err = np.array([r['std_reward'] for r in results])
        title = r'Reward Landscape: $\log(\epsilon)$ where $\epsilon=1-r$' + '\nAnalytical Surface vs RL Mean'
        filename = 'reward_landscape_3d_log_mean.png'
        point_label = 'RL Mean'
    else:
        z_vals = np.array([r['max_reward'] for r in results])
        z_err = None
        title = r'Reward Landscape: $\log(\epsilon)$ where $\epsilon=1-r$' + '\nAnalytical Surface vs RL Max'
        filename = 'reward_landscape_3d_log_max.png'
        point_label = 'RL Max'

    # Transform RL rewards to log(epsilon) where epsilon = 1-r
    epsilon_vals = 1 - z_vals  # Use epsilon directly
    z_vals_log = np.log(np.maximum(epsilon_vals, 1e-15))

    # Determine which points are inside HEC
    # HEC requires: SA (t <= 2s) and MMI (u <= 3(t-s))
    inside_hec = []
    for i, r in enumerate(results):
        s, t = r['s'], r['t']
        u_sq = 1 - 3*s**2 - 3*t**2
        u = np.sqrt(max(u_sq, 0))
        sa_satisfied = t <= 2*s
        mmi_satisfied = u <= 3*(t - s)
        inside_hec.append(sa_satisfied and mmi_satisfied)
    inside_hec = np.array(inside_hec)

    # Plot RL data points LAST with high zorder to ensure they appear above surface
    # Inside HEC: green, Outside HEC: red (no transparency)
    if np.any(inside_hec):
        ax.scatter(s_vals[inside_hec], t_vals[inside_hec], z_vals_log[inside_hec],
                   c='green', s=40, alpha=1.0, label='Inside HEC', depthshade=False, zorder=100)
    if np.any(~inside_hec):
        ax.scatter(s_vals[~inside_hec], t_vals[~inside_hec], z_vals_log[~inside_hec],
                   c='red', s=30, alpha=1.0, label='Outside HEC', depthshade=False, zorder=100)

    # Add error bars for mean version (transformed using epsilon)
    if version == 'mean' and z_err is not None:
        for i in range(len(s_vals)):
            if z_err[i] > 0.001:
                # Transform error bars using epsilon = 1-r
                eps_low = 1 - (z_vals[i] + z_err[i])  # higher reward = lower epsilon
                eps_high = 1 - (z_vals[i] - z_err[i])  # lower reward = higher epsilon
                z_low = np.log(max(eps_low, 1e-15))
                z_high = np.log(max(eps_high, 1e-15))
                err_color = 'green' if inside_hec[i] else 'red'
                ax.plot([s_vals[i], s_vals[i]],
                       [t_vals[i], t_vals[i]],
                       [z_low, z_high],
                       color=err_color, alpha=0.3, linewidth=1)

    # Labels (no title)
    ax.set_xlabel('s', fontsize=18, labelpad=10)
    ax.set_ylabel('t', fontsize=18, labelpad=10)
    ax.set_zlabel('')  # Remove label (was being cut off)
    ax.tick_params(axis='both', labelsize=14)

    # Set axis limits
    ax.set_xlim(S_MIN, S_MAX)
    ax.set_ylim(T_MIN, T_MAX)
    ax.set_zlim(-10, 0)

    # Add legend
    ax.legend(loc='upper left', fontsize=14)

    # Set viewing angle
    ax.view_init(elev=35, azim=45)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / filename}")

    # Also save PDF
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Disable automatic z-ordering so we can control draw order explicitly
    ax.computed_zorder = False

    # Z-ordering: constraint lines (zorder=0) < surfaces (zorder=10) < RL points (zorder=100)

    # Draw constraint boundaries FIRST on z=-10 base plane (lowest zorder)
    ax.plot(s_disk, t_disk, z_base * np.ones_like(s_disk), 'k-', lw=2, zorder=0)
    ax.plot(s_sa, t_sa, z_base * np.ones_like(s_sa), 'r-', lw=2, zorder=0)
    if s_ssa:
        ax.plot(s_ssa, t_ssa, z_base * np.ones(len(s_ssa)), 'b-', lw=2, zorder=0)
    if s_mmi:
        ax.plot(s_mmi, t_mmi, z_base * np.ones(len(s_mmi)), 'm-', lw=2, zorder=0)

    # Explicit -10 contour lines (SA side and MMI side) - dashed blue (behind surfaces)
    ax.plot(s_contour_sa, t_contour_sa, -10 * np.ones_like(s_contour_sa), 'b--', lw=2, zorder=0, label=r'$\log(1-\mathrm{reward})=-10$')
    if 'contour_mmi_s' in data and data['contour_mmi_s'].size > 0:
        ax.plot(s_contour_mmi, t_contour_mmi, -10 * np.ones_like(s_contour_mmi), 'b--', lw=2, zorder=0)

    # Plot surfaces (middle zorder, above lines but below data points)
    ax.plot_surface(data['S_inner'], data['T_inner'], data['Z_analytical_inner'],
                    color='lightblue', alpha=0.7,
                    linewidth=0, antialiased=True,
                    rstride=1, cstride=1, zorder=10)

    ax.plot_surface(data['S_right'], data['T_right'], data['Z_analytical_right'],
                    color='lightblue', alpha=0.7,
                    linewidth=0, antialiased=True,
                    rstride=1, cstride=1, zorder=10)

    if 'S_left' in data and data['S_left'].size > 0:
        ax.plot_surface(data['S_left'], data['T_left'], data['Z_analytical_left'],
                        color='lightblue', alpha=0.7,
                        linewidth=0, antialiased=True,
                        rstride=1, cstride=1, zorder=10)

    # Keep default background (grid and colored room panes)

    # Draw grid lines on analytical surface (thin gray, 20x20 density)
    for s_val in s_grid_vals:
        t_line = np.linspace(T_MIN, T_MAX, 100)
        s_line = np.full_like(t_line, s_val)
        z_line = np.array([np.log(max(1 - compute_reward_analytic(s_val, t), 1e-15))
                           if 3*s_val**2 + 3*t**2 < 1 else np.nan for t in t_line])
        z_line = np.where(z_line < z_cutoff, np.nan, z_line)
        ax.plot(s_line, t_line, z_line, color='gray', linewidth=0.5, alpha=0.6, zorder=50)

    for t_val in t_grid_vals:
        s_line = np.linspace(S_MIN, S_MAX, 100)
        t_line = np.full_like(s_line, t_val)
        z_line = np.array([np.log(max(1 - compute_reward_analytic(s, t_val), 1e-15))
                           if 3*s**2 + 3*t_val**2 < 1 else np.nan for s in s_line])
        z_line = np.where(z_line < z_cutoff, np.nan, z_line)
        ax.plot(s_line, t_line, z_line, color='gray', linewidth=0.5, alpha=0.6, zorder=50)

    # Plot RL data points LAST with high zorder to ensure they appear above surface
    if np.any(inside_hec):
        ax.scatter(s_vals[inside_hec], t_vals[inside_hec], z_vals_log[inside_hec],
                   c='green', s=40, alpha=1.0, depthshade=False, zorder=100)
    if np.any(~inside_hec):
        ax.scatter(s_vals[~inside_hec], t_vals[~inside_hec], z_vals_log[~inside_hec],
                   c='red', s=30, alpha=1.0, depthshade=False, zorder=100)

    if version == 'mean' and z_err is not None:
        for i in range(len(s_vals)):
            if z_err[i] > 0.001:
                # Use epsilon = 1-r for error bars
                eps_low = 1 - (z_vals[i] + z_err[i])
                eps_high = 1 - (z_vals[i] - z_err[i])
                z_low = np.log(max(eps_low, 1e-15))
                z_high = np.log(max(eps_high, 1e-15))
                err_color = 'green' if inside_hec[i] else 'red'
                ax.plot([s_vals[i], s_vals[i]],
                       [t_vals[i], t_vals[i]],
                       [z_low, z_high],
                       color=err_color, alpha=0.3, linewidth=1)

    ax.set_xlabel('s', fontsize=18, labelpad=10)
    ax.set_ylabel('t', fontsize=18, labelpad=10)
    ax.set_zlabel('')  # Remove label (was being cut off)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim(S_MIN, S_MAX)
    ax.set_ylim(T_MIN, T_MAX)
    ax.set_zlim(-10, 0)
    ax.view_init(elev=35, azim=45)
    ax.legend(loc='upper left', fontsize=14)

    plt.tight_layout()
    pdf_filename = filename.replace('.png', '.pdf')
    plt.savefig(output_dir / pdf_filename, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / pdf_filename}")


def main():
    # Paths
    script_dir = Path(__file__).parent
    base_results = Path('/resnick/groups/OoguriGroup/jaeha/entropyconeRL/results/experiment_for_3-3')
    output_dir = script_dir

    # Load from v6 only (dt=0.1, seed=999, 20 runs per point)
    results_dirs = [
        base_results / 'n3_grid_validation_v6',   # dt=0.1, seed=999, 20 runs
    ]

    print("=" * 60)
    print("3D Reward Landscape Plot")
    print("=" * 60)

    # Load RL results from v6
    print("\nLoading RL results from v6 (dt=0.1, seed=999, 20 runs/point)...")
    results = load_rl_results_combined(results_dirs)
    print(f"Loaded {len(results)} data points")

    # Show sample statistics
    if results:
        n_runs = [r.get('n_runs', 0) for r in results]
        print(f"Runs per point: min={min(n_runs)}, max={max(n_runs)}, typical={n_runs[0]}")

    # Generate plots
    print("\nGenerating 3D plots...")

    # Mean with error bars
    plot_3d_landscape(results, output_dir, version='mean')
    plot_3d_landscape_wireframe(results, output_dir, version='mean')

    # Max reward
    plot_3d_landscape(results, output_dir, version='max')
    plot_3d_landscape_wireframe(results, output_dir, version='max')

    # Log scale versions
    print("\nGenerating log-scale 3D plots with fixed z-ordering...")
    plot_3d_landscape_log(results, output_dir, version='mean')
    plot_3d_landscape_log(results, output_dir, version='max')

    print("\nDone!")


if __name__ == "__main__":
    main()
