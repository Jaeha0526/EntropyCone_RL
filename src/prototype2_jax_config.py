#!/usr/bin/env python3
"""
Configuration-based JAX-accelerated prototype for reward-only gradient following.
Loads experiment configuration from JSON files in configurations/prototype2/
"""

import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
import argparse
import sys

# Import the JAX-accelerated version
# Get the src directory path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

# Override the HECenv with optimized JAX version
import HECenv_parallel
from jax_optimization.HECenv_JAX import HECenv_JAX, create_optimized_sfromw_jax
HECenv_parallel.HECenv = HECenv_JAX

# Optimize scipy search with JAX (42x speedup)
from jax_optimization.scipy_search_optimized import (
    scipy_search_optimized,
    get_available_cpus_optimized,
    scipy_search_wrapper_optimized
)

# Store JAX function creator in module for scipy_search to access
HECenv_parallel.create_optimized_sfromw_jax = create_optimized_sfromw_jax

# Monkey-patch scipy functions
HECenv_parallel.scipy_search_original = HECenv_parallel.scipy_search  # Keep backup
HECenv_parallel.scipy_search = scipy_search_optimized
HECenv_parallel.get_available_cpus = get_available_cpus_optimized
HECenv_parallel.scipy_search_wrapper = scipy_search_wrapper_optimized

print("=" * 60)
print("✓ JAX-optimized HECenv loaded (334x speedup for RL training)")
print("✓ Scipy optimization patched with JAX (42x speedup)")
print("=" * 60)

# Now import run_stage which will use the JAX version
from HECenv_parallel import run_stage

# Import QP Safe Direction module
from qp_safe_direction import QPSafeDirection

# =============================================================================
# Path Resolution Helper
# =============================================================================
def get_repo_root():
    """Get the repository root directory (parent of src/)"""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(src_dir)

def resolve_data_path(relative_path):
    """Resolve a relative path from the repository root."""
    repo_root = get_repo_root()
    return os.path.join(repo_root, relative_path)

# =============================================================================
# Ray Loading Functions
# =============================================================================

# -- Primary ray source for paper (Section 3.4: N=3 MMI Finding) --

def load_n3_mmi_ray(ray_index, file_path=None):
    """Load a specific ray from the n=3 SA cone extremal rays for MMI finding experiment.

    This is the primary ray source for Section 3.4 of the paper.
    These are extremal rays of the SA (subadditivity) cone that may violate MMI.
    The experiment uses gradient-following to approach the MMI boundary.
    """
    if file_path is None:
        file_path = resolve_data_path("data/n3/sa_cone_rays.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"n=3 SA cone rays file not found: {file_path}")

    rays = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comment lines and empty lines
            if line and not line.startswith('#'):
                ray_coords = [float(x) for x in line.split()]
                rays.append(ray_coords)

    if ray_index < 1 or ray_index > len(rays):
        raise ValueError(f"n=3 MMI ray index {ray_index} out of range (1-{len(rays)})")

    return np.array(rays[ray_index - 1])

# -- Extended research ray sources (not used in paper) --
# Note: These require additional data files from the full entropyconeRL repository

def load_ray_from_file(ray_index, file_path=None):
    """Load a ray from converted rays file. [Extended research - not in paper]"""
    if file_path is None:
        file_path = resolve_data_path("data/n6/SA_cone_converted_rays.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Rays file not found: {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if ray_index < 1 or ray_index > len(lines):
        raise ValueError(f"Ray index {ray_index} out of range (1-{len(lines)})")
    ray_data = lines[ray_index - 1].strip().split()
    return np.array([float(x) for x in ray_data])

def load_new_ray(ray_index, file_path=None):
    """Load a ray from newly discovered rays file. [Extended research - not in paper]"""
    if file_path is None:
        file_path = resolve_data_path("extremal_rays/lp_ray_finder/converted_new_rays.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"New rays file not found: {file_path}. This data is not included in the clean paper repository.")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if ray_index < 1 or ray_index > len(lines):
        raise ValueError(f"New ray index {ray_index} out of range (1-{len(lines)})")
    ray_data = lines[ray_index - 1].strip().split()
    return np.array([float(x) for x in ray_data])

def load_genuine_new_ray(ray_index, file_path=None):
    """Load a ray from genuinely new rays file. [Extended research - not in paper]"""
    if file_path is None:
        file_path = resolve_data_path("extremal_rays/n6/new_extremal_rays/genuine_new_rays_converted_corrected.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Genuine new rays file not found: {file_path}. This data is not included in the clean paper repository.")
    rays = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ray_coords = [float(x) for x in line.split()]
                rays.append(ray_coords)
    if ray_index < 1 or ray_index > len(rays):
        raise ValueError(f"Genuine new ray index {ray_index} out of range (1-{len(rays)})")
    return np.array(rays[ray_index - 1])

def load_spurious_ray(ray_index, file_path=None):
    """Load a ray from spurious rays file. [Extended research - not in paper]"""
    if file_path is None:
        file_path = resolve_data_path("extremal_rays/n5/from_SSA_MMI/rays_not_in_true_hec.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Spurious rays file not found: {file_path}. This data is not included in the clean paper repository.")
    rays = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ray_coords = [float(x) for x in line.split()]
                rays.append(ray_coords)
    if ray_index < 1 or ray_index > len(rays):
        raise ValueError(f"Spurious ray index {ray_index} out of range (1-{len(rays)})")
    return np.array(rays[ray_index - 1])

def load_experiment4_ray(ray_index, file_path=None):
    """Load a ray from experiment4 S6-unique vertices file. [Extended research - not in paper]"""
    if file_path is None:
        file_path = resolve_data_path("extremal_rays/n5/from_complete_SA_MMI/experiment4/vertices_3M_final_s6_unique_20250924_200924.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Experiment4 rays file not found: {file_path}. This data is not included in the clean paper repository.")
    rays = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ray_coords = [float(x) for x in line.split()]
                rays.append(ray_coords)
    if ray_index < 1 or ray_index > len(rays):
        raise ValueError(f"Experiment4 ray index {ray_index} out of range (1-{len(rays)})")
    return np.array(rays[ray_index - 1])

def load_configuration(config_path):
    """Load experiment configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_reward_gradient(Svector, r, Sgrad, gradient_clip_norm=None, use_log_weighting=True):
    """
    Return gradient for optimization.
    
    Args:
        gradient_clip_norm: If provided, clip gradient to this maximum norm
        use_log_weighting: If True, use gradient to minimize log(1-r) [weighted by 1/(1-r)]
                          If False, use raw gradient of reward
    """
    Sgrad = np.array(Sgrad)
    
    if use_log_weighting:
        # Original: minimize log(1-r)
        # ∇[log(1-r)] = -∇r/(1-r)
        # Move in -∇[log(1-r)] direction = +∇r/(1-r)
        if r >= 0.999:  # Safety check
            print(f"Warning: r={r} too close to 1, using raw gradient instead")
            gradient = Sgrad
        else:
            gradient = Sgrad / (1 - r)
    else:
        # Just use raw gradient of reward (maximize r directly)
        gradient = Sgrad
    
    # Optional gradient clipping
    if gradient_clip_norm is not None:
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > gradient_clip_norm:
            gradient = gradient * (gradient_clip_norm / grad_norm)
            print(f"Gradient clipped from {grad_norm:.4f} to {gradient_clip_norm:.4f}")
    
    return gradient

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def main():
    parser = argparse.ArgumentParser(description='Configuration-based Entropy Cone Boundary Search with JAX')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file (e.g., configurations/prototype2/experiment.json)')
    parser.add_argument('--override-dir', type=str, default=None,
                       help='Override output directory from config')
    
    cmd_args = parser.parse_args()
    
    # Load configuration
    config = load_configuration(cmd_args.config)
    
    # Extract configurations
    training_config = config['training']
    experiment_config = config['experiment']
    
    # Load starting ray based on configuration
    if experiment_config['ray_source'] == 'original':
        prev_points = load_ray_from_file(experiment_config['ray_index'])
        print(f"Loaded original ray {experiment_config['ray_index']}")
        n_parties = 6  # Original rays are n=6
    elif experiment_config['ray_source'] == 'new':
        prev_points = load_new_ray(experiment_config['ray_index'])
        print(f"Loaded new ray {experiment_config['ray_index']}")
        n_parties = 6  # New rays are n=6
    elif experiment_config['ray_source'] == 'genuine_new':
        prev_points = load_genuine_new_ray(experiment_config['ray_index'])
        print(f"Loaded genuine new ray {experiment_config['ray_index']} (corrected S7 deduplication)")
        n_parties = 6  # Genuine new rays are n=6
    elif experiment_config['ray_source'] == 'spurious':
        prev_points = load_spurious_ray(experiment_config['ray_index'])
        print(f"Loaded spurious ray {experiment_config['ray_index']} (from SSA+MMI, not in true HEC)")
        n_parties = 5  # Spurious rays from SSA+MMI are n=5
    elif experiment_config['ray_source'] == 'experiment4':
        prev_points = load_experiment4_ray(experiment_config['ray_index'])
        print(f"Loaded experiment4 S6-unique ray {experiment_config['ray_index']} (from 3M enumeration)")
        n_parties = 5  # Experiment4 rays are n=5
    elif experiment_config['ray_source'] == 'n3_MMI_finding':
        prev_points = load_n3_mmi_ray(experiment_config['ray_index'])
        print(f"Loaded n=3 MMI finding ray {experiment_config['ray_index']} (SA cone ray for MMI approach)")
        n_parties = 3  # n=3 rays for MMI finding
    elif experiment_config['ray_source'] == 'custom':
        prev_points = np.array(experiment_config['custom_ray'])
        n_parties = experiment_config.get('n_parties', 3)
        print(f"Loaded custom ray with {len(prev_points)} components for n={n_parties}")
    else:
        raise ValueError(f"Unknown ray source: {experiment_config['ray_source']}")

    # Verify dimension matches expected size
    expected_dim = 2**n_parties - 1
    if len(prev_points) != expected_dim:
        raise ValueError(f"Ray dimension mismatch: got {len(prev_points)}, expected {expected_dim} for n={n_parties}")

    sum_components = sum(prev_points)
    
    # Set up args dictionary
    args = {}
    args["dir"] = cmd_args.override_dir if cmd_args.override_dir else experiment_config['output_dir']
    args["batch_size"] = training_config['batch_size']
    # Get device from environment config if specified, otherwise auto-detect
    env_config = config.get('environment', {})
    args["device"] = env_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    args["gpu_memory_fraction"] = training_config.get('gpu_memory_fraction', 0.95)
    args["seed"] = experiment_config.get('seed', None)  # Add seed support
    args["dt"] = training_config.get('dt', 1.1)
    args["nnn"] = n_parties  # Set based on ray source (5 for spurious, 6 for others)
    args["NN"] = training_config['NN']
    args["lr"] = training_config['lr']
    args["epsilon"] = training_config['epsilon']
    args["factor"] = training_config.get('factor', 0.7)
    args["keep_best_n"] = training_config.get('keep_best_n', 10)
    args["dS"] = training_config.get('dS', 0.00001)
    args["gradient_method"] = training_config.get('gradient_method', 'scipy')
    args["n_gradient_samples"] = training_config.get('n_gradient_samples', 100)
    args["gradient_cold_start"] = training_config.get('gradient_cold_start', False)
    args["gradient_distance_mode"] = training_config.get('gradient_distance_mode', 'fixed')
    args["gradient_n_repeats"] = training_config.get('gradient_n_repeats', 1)
    args["gradient_n_gpus"] = training_config.get('gradient_n_gpus', 1)
    args["hidden_dim"] = training_config['hidden_dim']
    args["num_layers"] = training_config['num_layers']
    args["rollout_length"] = training_config.get('rollout_length', 50)
    args["exploration_noise"] = training_config.get('exploration_noise', 0.15)
    args["normalize_weights"] = training_config.get('normalize_weights', True)
    args["enable_early_stopping"] = training_config.get('enable_early_stopping', True)
    args["early_stop_patience"] = training_config.get('early_stop_patience', 5)
    args["max_training_cycles"] = training_config['max_training_cycles']
    # Add precision support - default to float64 for accuracy
    args["precision"] = training_config.get('precision', 'float64')
    args["x_value"] = training_config.get('x_value', 1)
    
    # Warm-starting configuration
    args["warm_start"] = training_config.get('warm_start', True)
    args["model_save_path"] = os.path.join(args["dir"], "models")
    
    # Gradient following parameters
    Moves = experiment_config['moves']
    velocity = experiment_config['velocity']
    momentum = experiment_config['momentum']
    MODE = experiment_config.get('mode', 'reward_only')
    threshold = experiment_config.get('threshold', -20)  # Not used in reward_only mode
    gradient_clip_norm = experiment_config.get('gradient_clip_norm', None)  # Optional gradient clipping
    use_log_weighting = experiment_config.get('use_log_weighting', True)  # Default to True for backward compatibility
    
    # Early stopping parameters for trajectory (not for RL training)
    enable_trajectory_early_stop = experiment_config.get('enable_trajectory_early_stop', True)
    gradient_convergence_threshold = experiment_config.get('gradient_convergence_threshold', 1e-6)
    reward_plateau_threshold = experiment_config.get('reward_plateau_threshold', 1e-8)
    min_stages_before_convergence_check = experiment_config.get('min_stages_before_convergence_check', 5)
    min_stages_before_plateau_check = experiment_config.get('min_stages_before_plateau_check', 10)
    plateau_window_size = experiment_config.get('plateau_window_size', 5)

    # Safe direction mode parameters
    movement_mode = experiment_config.get('movement_mode', 'standard')
    safe_direction_config = experiment_config.get('safe_direction', {})

    # Initialize QP Safe Director if in safe_direction mode
    qp_director = None
    if movement_mode == 'safe_direction':
        known_facets_file = safe_direction_config.get('known_facets_file',
                                                       'data/n3/ray8_known_facets.txt')
        # Resolve relative path from repo root
        known_facets_file = resolve_data_path(known_facets_file)
        saturation_tolerance = safe_direction_config.get('saturation_tolerance', 1e-4)
        safety_factor = safe_direction_config.get('safety_factor', 0.9)
        verbose_qp = safe_direction_config.get('verbose', False)

        # Escape mode parameters
        use_escape_mode = safe_direction_config.get('use_escape_mode', False)
        min_increase_rate = safe_direction_config.get('min_increase_rate', 0.05)
        buffer_factor = safe_direction_config.get('buffer_factor', 1.5)
        escape_threshold_factor = safe_direction_config.get('escape_threshold_factor', 1.5)

        # Hybrid mode parameters
        use_hybrid_mode = safe_direction_config.get('use_hybrid_mode', False)
        hybrid_alpha = safe_direction_config.get('hybrid_alpha', 0.5)

        print(f"\nInitializing QP Safe Direction mode:")
        print(f"  Known facets file: {known_facets_file}")
        print(f"  Saturation tolerance: {saturation_tolerance}")
        print(f"  Safety factor: {safety_factor}")
        if use_hybrid_mode:
            print(f"  Hybrid mode: ENABLED")
            print(f"    Alpha: {hybrid_alpha:.2f} (1.0=pure escape, 0.0=pure maxmin)")
            print(f"    Min increase rate: {min_increase_rate}")
            print(f"    Buffer factor: {buffer_factor}")
            print(f"    Escape threshold factor: {escape_threshold_factor}")
        elif use_escape_mode:
            print(f"  Escape mode: ENABLED")
            print(f"    Min increase rate: {min_increase_rate}")
            print(f"    Buffer factor: {buffer_factor}")
            print(f"    Escape threshold factor: {escape_threshold_factor}")

        qp_director = QPSafeDirection(
            known_facets_file=known_facets_file,
            saturation_tolerance=saturation_tolerance,
            safety_factor=safety_factor,
            verbose=verbose_qp,
            dS=args["dS"],
            use_escape_mode=use_escape_mode,
            min_increase_rate=min_increase_rate,
            buffer_factor=buffer_factor,
            escape_threshold_factor=escape_threshold_factor,
            use_hybrid_mode=use_hybrid_mode,
            hybrid_alpha=hybrid_alpha
        )

    previous_gradient = None
    
    print("="*80)
    print("Configuration-based JAX-ACCELERATED Gradient Following")
    print("="*80)
    print(f"Configuration file: {cmd_args.config}")
    print(f"Using JAX-optimized HEC environment for ~334x speedup")
    print(f"GPU memory fraction: {args['gpu_memory_fraction']}")
    if args["seed"] is not None:
        print(f"Random seed: {args['seed']}")
    print(f"\nStarting {MODE} gradient following...")
    print(f"Ray source: {experiment_config['ray_source']}, index: {experiment_config.get('ray_index', 'N/A')}")
    print(f"Initial point: {prev_points}")
    print(f"Sum of components: {sum_components}")
    print(f"Parameters: momentum={momentum}, velocity={velocity}, moves={Moves}")
    print(f"Problem size: n={n_parties}, N={args['NN']}")
    print(f"Output directory: {args['dir']}")
    
    # Print early stopping settings
    if enable_trajectory_early_stop:
        print(f"Early stopping: ENABLED")
        print(f"  - Gradient convergence: < {gradient_convergence_threshold:.1e} after stage {min_stages_before_convergence_check}")
        print(f"  - Reward plateau: < {reward_plateau_threshold:.1e} variation over {plateau_window_size} stages after stage {min_stages_before_plateau_check}")
    else:
        print(f"Early stopping: DISABLED")
    
    # Initialize directories and files
    os.makedirs(args["dir"], exist_ok=True)
    os.makedirs(args["model_save_path"], exist_ok=True)
    results_file = os.path.join(args["dir"], "experiment_results.json")
    hyperparams_file = os.path.join(args["dir"], "hyperparameters.json")
    
    # Save configuration
    full_config = {
        "config_file": cmd_args.config,
        "training_args": training_config,
        "experiment_settings": experiment_config,
        "derived_settings": {
            "initial_ray": prev_points.tolist(),
            "initial_sum": sum_components,
            "jax_acceleration": True,
            "trajectory_early_stopping": {
                "enabled": enable_trajectory_early_stop,
                "gradient_convergence_threshold": gradient_convergence_threshold,
                "reward_plateau_threshold": reward_plateau_threshold,
                "min_stages_before_convergence_check": min_stages_before_convergence_check,
                "min_stages_before_plateau_check": min_stages_before_plateau_check,
                "plateau_window_size": plateau_window_size
            }
        }
    }
    
    with open(hyperparams_file, 'w') as f:
        json.dump(full_config, f, indent=4)
    
    # Initialize results
    all_results = [{
        "start_points": prev_points.tolist(),
        "sum_components": sum_components,
        "config": full_config
    }]
    
    # Initialize tracking lists
    stages = []
    positions = []
    rewards = []
    gradients = []
    reward_improvements = []
    
    overall_best_reward = -1.0
    overall_best_stage = -1
    
    print("\nStarting JAX-accelerated exploration...")
    print("Note: First stage will include JAX JIT compilation (~35s)")
    
    # Main gradient following loop
    for i in range(Moves):
        # Set the stage parameter
        args["stage"] = i
        args["target"] = torch.tensor(prev_points, dtype=torch.float32)
        args["name"] = f"{experiment_config.get('name_prefix', 'gradient_stage')}_{i}"
        
        # Set up warm-starting
        if i == 0:
            args["load_model_path"] = None
            args["load_optimizer_state"] = False
            print("Stage 0: Training from scratch (includes JAX JIT compilation)")
        else:
            prev_best_model_path = os.path.join(args["model_save_path"], f"stage_{i-1}_best.pth")
            args["load_model_path"] = prev_best_model_path
            args["load_optimizer_state"] = True
            print(f"Stage {i}: Continuing from BEST of stage {i-1}")
        
        # Set save paths
        args["save_model_path"] = os.path.join(args["model_save_path"], f"stage_{i}_final.pth")
        args["save_best_model_path"] = os.path.join(args["model_save_path"], f"stage_{i}_best.pth")
        
        print(f"\n=== Stage {i} ===")
        print(f"Current position sum: {sum(prev_points):.6f}")
        
        # Run training stage
        result_from_run_stage = run_stage(args)
        current_run_result = dict(result_from_run_stage)

        reward = np.array(current_run_result["best_optimized_reward"])
        S_grad = current_run_result["S_grad_from_best"]

        print(f"Reward: {reward:.6f}")
        print(f"Log(1-reward): {np.log(1-reward):.6f}")

        # If this is not the first stage, update the placeholder entry from previous stage
        if i > 0:
            # The placeholder was added as the last entry, replace it with full results
            all_results.pop()  # Remove placeholder
        
        # Track overall best
        if reward > overall_best_reward:
            overall_best_reward = reward
            overall_best_stage = i
            print(f"*** NEW OVERALL BEST! Stage {i}, Reward: {reward:.8f} ***")
        
        # Update tracking data
        stages.append(i)
        positions.append(prev_points.copy())
        rewards.append(reward)
        
        # Get gradient based on mode
        if MODE == "reward_only":
            gradient = get_reward_gradient(prev_points, reward, S_grad, gradient_clip_norm, use_log_weighting)
        else:
            raise NotImplementedError(f"Mode {MODE} not implemented")
        
        # Apply momentum
        if previous_gradient is None:
            previous_gradient = np.zeros_like(gradient)
        gradient = momentum * previous_gradient + (1 - momentum) * gradient
        previous_gradient = gradient.copy()
        
        gradients.append(gradient.copy())
        
        # Calculate improvement potential
        gradient_magnitude = np.linalg.norm(gradient)
        reward_improvement = gradient_magnitude / (1 - reward)
        reward_improvements.append(reward_improvement)
        
        print(f"Gradient magnitude: {gradient_magnitude:.6f}")
        print(f"Reward improvement potential: {reward_improvement:.6f}")
        
        # Check for early stopping conditions
        if enable_trajectory_early_stop:
            # Check for gradient convergence
            if i > min_stages_before_convergence_check and gradient_magnitude < gradient_convergence_threshold:
                print(f"\n*** EARLY STOP: Converged at stage {i}! ***")
                print(f"Gradient magnitude {gradient_magnitude:.2e} below threshold {gradient_convergence_threshold:.2e}")
                # Store final results before breaking
                stage_result = {
                    "stage": i,
                    "position": positions[i].tolist(),
                    "reward": float(reward),
                    "gradient": gradient.tolist(),
                    "gradient_magnitude": float(gradient_magnitude),
                    "reward_improvement": float(reward_improvement),
                    "sum_components": float(sum(positions[i])),
                    "log_1_minus_reward": float(np.log(1-reward)),
                    "early_stop_reason": "gradient_convergence"
                }
                stage_result.update(convert_to_serializable(current_run_result))
                all_results.append(stage_result)
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                break
            
            # Check for reward plateau
            if i > min_stages_before_plateau_check and len(rewards) >= plateau_window_size:
                recent_rewards = rewards[-plateau_window_size:]
                reward_variation = max(recent_rewards) - min(recent_rewards)
                if reward_variation < reward_plateau_threshold:
                    print(f"\n*** EARLY STOP: Reward plateau detected at stage {i}! ***")
                    print(f"Last {plateau_window_size} rewards variation: {reward_variation:.2e}")
                    print(f"Below threshold: {reward_plateau_threshold:.2e}")
                    print(f"May have found boundary.")
                    # Store final results before breaking
                    stage_result = {
                        "stage": i,
                        "position": positions[i].tolist(),
                        "reward": float(reward),
                        "gradient": gradient.tolist(),
                        "gradient_magnitude": float(gradient_magnitude),
                        "reward_improvement": float(reward_improvement),
                        "sum_components": float(sum(positions[i])),
                        "log_1_minus_reward": float(np.log(1-reward)),
                        "early_stop_reason": "reward_plateau"
                    }
                    stage_result.update(convert_to_serializable(current_run_result))
                    all_results.append(stage_result)
                    with open(results_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    break
        
        # Update position based on movement mode
        if movement_mode == 'standard':
            # Standard mode: direct gradient application with safety scaling
            movement = velocity * gradient

            # Find scale factor to ensure no coordinate drops below 10% of original value
            # (only constrain coordinates >= 1e-6 with negative movement)
            scale = 1.0
            for coord_idx in range(len(prev_points)):
                if prev_points[coord_idx] >= 1e-6 and movement[coord_idx] < 0:
                    max_decrease = 0.9 * prev_points[coord_idx]
                    required_scale = max_decrease / abs(movement[coord_idx])
                    scale = min(scale, required_scale)

            if scale < 1.0:
                print(f"Scaling movement by {scale:.4f} to maintain 10% minimum on large coordinates")

            # Apply scaled movement
            prev_points = prev_points + scale * movement

            # Log movement metadata
            movement_metadata = {
                'mode': 'standard',
                'velocity': velocity,
                'scale_factor': scale,
                'movement_distance': np.linalg.norm(scale * movement)
            }

        elif movement_mode == 'safe_direction':
            # Safe direction mode: QP-adjusted gradient with facet constraints
            # Compute safe direction and distance
            safe_direction, safe_distance, qp_metadata = qp_director.compute_safe_direction(
                prev_points, gradient
            )

            # Compute requested movement distance from velocity
            requested_distance = velocity * np.linalg.norm(gradient)

            # Use the smaller of safe_distance and requested_distance
            actual_distance = min(safe_distance, requested_distance)

            # Apply safe movement
            prev_points = prev_points + actual_distance * safe_direction

            # Log movement metadata
            movement_metadata = {
                'mode': 'safe_direction',
                'escape_mode_enabled': use_escape_mode,
                'escape_mode_successful': qp_metadata.get('escape_mode_successful', None),
                'fallback_occurred': qp_metadata.get('fallback_occurred', False),
                'fallback_reason': qp_metadata.get('fallback_reason', None),
                'qp_metadata': qp_metadata,
                'actual_movement': actual_distance,
                'requested_movement': requested_distance,
                'safe_distance': safe_distance
            }

            print(f"\n[Safe Direction Mode]")
            if use_hybrid_mode and 'hybrid_mode_successful' in qp_metadata:
                print(f"  Hybrid mode: {'SUCCESS' if qp_metadata['hybrid_mode_successful'] else 'FALLBACK'}")
                print(f"  Alpha: {qp_metadata.get('hybrid_alpha', hybrid_alpha):.2f}")
                if qp_metadata.get('hybrid_mode_successful', False):
                    # Show adaptive constraint logic
                    if 'maxmin_value' in qp_metadata:
                        print(f"  MaxMin value (max achievable): {qp_metadata['maxmin_value']:.6f}")
                    if 'escape_min_increase_used' in qp_metadata:
                        print(f"  Escape constraint (adaptive): {qp_metadata['escape_min_increase_used']:.6f}")

                    # Show gradient similarities
                    print(f"  MaxMin grad sim: {qp_metadata.get('maxmin_gradient_similarity', 0):.4f}")
                    print(f"  Escape grad sim: {qp_metadata.get('escape_gradient_similarity', 0):.4f}")
                    print(f"  Hybrid grad sim: {qp_metadata.get('hybrid_gradient_similarity', 0):.4f}")

                    # Show min increases (what each direction achieved)
                    if 'maxmin_min_increase' in qp_metadata:
                        print(f"  MaxMin min inc: {qp_metadata['maxmin_min_increase']:.6f}")
                    if 'escape_min_increase' in qp_metadata:
                        print(f"  Escape min inc: {qp_metadata['escape_min_increase']:.6f}")
                    if 'hybrid_min_increase' in qp_metadata:
                        print(f"  Hybrid min inc: {qp_metadata['hybrid_min_increase']:.6f}")
                elif qp_metadata.get('fallback_occurred', False):
                    print(f"  ⚠ Fallback reason: {qp_metadata.get('fallback_reason', 'unknown')}")
            elif use_escape_mode:
                print(f"  Escape mode: {'SUCCESS' if qp_metadata.get('escape_mode_successful', False) else 'FALLBACK'}")
                if qp_metadata.get('fallback_occurred', False):
                    print(f"  ⚠ Fallback reason: {qp_metadata.get('fallback_reason', 'unknown')}")
                if 'min_facet_increase' in qp_metadata:
                    print(f"  Min facet increase: {qp_metadata['min_facet_increase']:.4f}")
            print(f"  Gradient similarity: {qp_metadata['gradient_similarity']:.4f}")
            if 'n_saturated' in qp_metadata:
                print(f"  Saturated facets: {qp_metadata['n_saturated']}")
            if 'n_below_threshold' in qp_metadata:
                print(f"  Below threshold: {qp_metadata['n_below_threshold']}")
            if 'n_violating' in qp_metadata:
                print(f"  Constraints applied: {qp_metadata['n_violating']}")
            print(f"  Requested distance: {requested_distance:.6e}")
            print(f"  Safe distance: {safe_distance:.6e}")
            print(f"  Actual distance: {actual_distance:.6e} ({'velocity-limited' if actual_distance < safe_distance else 'safety-limited'})")
            print(f"  Max theoretical: {qp_metadata['max_safe_distance_raw']:.6e}")

        # Store stage results BEFORE movement
        stage_result = {
            "stage": i,
            "position": positions[i].tolist(),
            "reward": float(reward),
            "gradient": gradient.tolist(),
            "gradient_magnitude": float(gradient_magnitude),
            "reward_improvement": float(reward_improvement),
            "sum_components": float(sum(positions[i])),
            "log_1_minus_reward": float(np.log(1-reward)),
            "movement_metadata": movement_metadata
        }

        # Add detailed results from run_stage
        stage_result.update(convert_to_serializable(current_run_result))
        all_results.append(stage_result)

        # Save results after each stage
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Clip tiny coordinates to non-negative (for both modes)
        prev_points = np.maximum(prev_points, 0)

        # Log the new position immediately after movement (placeholder for next stage)
        # Only add placeholder if this is not the last move
        if i < Moves - 1:
            next_position_result = {
                "stage": i + 1,
                "position": prev_points.tolist(),
                "sum_components": float(sum(prev_points)),
                "note": "Position after movement - training results will be filled in next stage"
            }
            all_results.append(next_position_result)

            # Save again with the new position placeholder
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
        
        # Update hyperparameters file with latest stage
        full_config["experiment_settings"]["latest_completed_stage"] = i
        with open(hyperparams_file, 'w') as f:
            json.dump(full_config, f, indent=4)
        
        # Create updated plots with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))
        
        # Plot rewards
        ax1.plot(stages, rewards, 'b-', linewidth=2, label='Reward')
        ax1.axhline(y=overall_best_reward, color='r', linestyle='--', alpha=0.5, 
                   label=f'Best: {overall_best_reward:.6f} (Stage {overall_best_stage})')
        ax1.set_xlabel('Stage')
        ax1.set_ylabel('Reward')
        ax1.set_title('Reward Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot -log(1-reward)
        log_rewards = [-np.log(1 - r) if r < 1 else 20 for r in rewards]
        ax2.plot(stages, log_rewards, 'r-', linewidth=2)
        ax2.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Target (-log(1-r) = 20)')
        ax2.set_xlabel('Stage')
        ax2.set_ylabel('-log(1 - reward)')
        ax2.set_title('Convergence to Boundary')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        # Plot gradient magnitude
        ax3.plot(stages, [np.linalg.norm(g) for g in gradients], 'g-', linewidth=2)
        ax3.set_xlabel('Stage')
        ax3.set_ylabel('Gradient Magnitude')
        ax3.set_title('Gradient Magnitude Evolution')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args["dir"], "realtime_progress.png"), dpi=150)
        plt.close()
        
        # Create/update coordinate evolution plot after each stage
        positions_array = np.array(positions)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a heatmap of coordinate evolution
        # Transpose so coordinates are on y-axis and stages on x-axis
        im = ax.imshow(positions_array.T, aspect='auto', cmap='RdBu_r', 
                      vmin=-np.max(np.abs(positions_array)),
                      vmax=np.max(np.abs(positions_array)))
        
        ax.set_xlabel('Stage')
        ax.set_ylabel('Component Index')
        ax.set_title(f'Coordinate Evolution - Stage {i}')
        ax.set_xlim(-0.5, len(stages)-0.5)
        ax.set_xticks(range(0, len(stages), max(1, len(stages)//10)))
        ax.set_xticklabels(stages[::max(1, len(stages)//10)])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Component Value', rotation=270, labelpad=15)
        
        # Show only first 30 components for clarity
        if positions_array.shape[1] > 30:
            ax.set_ylim(29.5, -0.5)
            ax.set_yticks(range(0, 30, 5))
            ax.set_yticklabels(range(0, 30, 5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(args["dir"], "coordinate_evolution.png"), dpi=150)
        plt.close()
    
    # Add final position entry (after last movement) if we completed all moves
    if len(stages) == Moves:
        final_position_result = {
            "stage": Moves,
            "position": prev_points.tolist(),
            "sum_components": float(sum(prev_points)),
            "note": "Final position after all movements completed"
        }
        all_results.append(final_position_result)

        # Save results with final position
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Create final coordinate evolution plot (line plot version)
    print("\nCreating final coordinate evolution line plot...")
    positions_array = np.array(positions)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot a subset of coordinates
    num_coords_to_plot = min(10, positions_array.shape[1])
    for coord in range(num_coords_to_plot):
        ax.plot(stages, positions_array[:, coord], label=f'Coord {coord}', alpha=0.7)
    
    ax.set_xlabel('Stage')
    ax.set_ylabel('Coordinate Value')
    ax.set_title('Coordinate Evolution During Gradient Following (Line Plot)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args["dir"], "coordinate_evolution_lines.png"), dpi=150)
    plt.close()
    
    print("\n" + "="*80)
    print("Gradient following complete!")
    print(f"Best reward: {overall_best_reward:.8f} at stage {overall_best_stage}")
    print(f"Results saved to: {args['dir']}")
    print("="*80)

if __name__ == "__main__":
    main()