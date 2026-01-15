#!/usr/bin/env python3
"""
Configuration-based runner for HECenv_parallel RL training.

This script allows running HEC classification experiments using JSON configuration files,
making it easy to test different parameter combinations and classify new rays as inside/outside HEC.

Usage:
    python HECenv_parallel_config.py --config configurations/hecenv_parallel/newray_classification.json
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from datetime import datetime

# Import the JAX-accelerated version for speedup
# Get the src directory path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

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

from HECenv_parallel import *

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

# -- Primary ray source for paper (Section 4: N=6 SA Cone Classification) --

def load_sa_cone_ray(ray_index, file_path=None):
    """Load a specific ray from the SA cone extremal rays file (208 rays in binary ordering).

    This is the primary ray source for Section 4 of the paper.
    """
    if file_path is None:
        file_path = resolve_data_path("data/n6/SA_cone_converted_rays.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SA cone rays file not found: {file_path}")

    rays = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comment lines and empty lines
            if line and not line.startswith('#'):
                ray_coords = [float(x) for x in line.split()]
                rays.append(ray_coords)

    if ray_index < 1 or ray_index > len(rays):
        raise ValueError(f"Ray index {ray_index} out of range. Available: 1-{len(rays)}")

    return np.array(rays[ray_index - 1])

# -- Extended research ray sources (not used in paper) --
# Note: These require additional data files from the full entropyconeRL repository

def load_new_ray(ray_index, file_path=None):
    """Load a ray from newly discovered rays file. [Extended research - not in paper]"""
    if file_path is None:
        file_path = resolve_data_path("extremal_rays/lp_ray_finder/converted_new_rays.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"New rays file not found: {file_path}. This data is not included in the clean paper repository.")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if ray_index < 1 or ray_index > len(lines):
        raise ValueError(f"Ray index {ray_index} out of range. Available: 1-{len(lines)}")
    ray_coords = [float(x) for x in lines[ray_index - 1].strip().split()]
    return np.array(ray_coords)

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
        raise ValueError(f"Ray index {ray_index} out of range. Available: 1-{len(rays)}")
    return np.array(rays[ray_index - 1])

def load_original_ray(ray_index, file_path=None):
    """Load a ray from original extremal rays file. [Extended research - not in paper]"""
    if file_path is None:
        file_path = resolve_data_path("extremal_rays/n6/rays.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Original rays file not found: {file_path}. This data is not included in the clean paper repository.")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if ray_index < 0 or ray_index >= len(lines):
        raise ValueError(f"Ray index {ray_index} out of range. Available: 0-{len(lines)-1}")
    ray_coords = [float(x) for x in lines[ray_index].strip().split()]
    return np.array(ray_coords)

def load_spurious_ray(ray_index, file_path=None):
    """Load a ray from spurious rays file (not in true HEC). [Extended research - not in paper]"""
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
        raise ValueError(f"Ray index {ray_index} out of range. Available: 1-{len(rays)}")
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
        raise ValueError(f"Ray index {ray_index} out of range. Available: 1-{len(rays)}")
    return np.array(rays[ray_index - 1])

def load_config(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def setup_target_ray(config):
    """Setup target ray based on configuration"""
    ray_source = config['experiment']['target_ray_source']

    if ray_source == "new":
        ray_index = config['experiment']['target_ray_index']
        print(f"Loading new ray {ray_index}")
        target_ray = load_new_ray(ray_index)
        print(f"Target ray shape: {target_ray.shape}")
        print(f"First 5 coordinates: {target_ray[:5]}")
    elif ray_source == "genuine_new":
        ray_index = config['experiment']['target_ray_index']
        print(f"Loading genuine new ray {ray_index} (corrected S7 deduplication)")
        target_ray = load_genuine_new_ray(ray_index)
        print(f"Target ray shape: {target_ray.shape}")
        print(f"First 5 coordinates: {target_ray[:5]}")
    elif ray_source == "original":
        ray_index = config['experiment']['target_ray_index']
        print(f"Loading original ray {ray_index}")
        target_ray = load_original_ray(ray_index)
        print(f"Target ray shape: {target_ray.shape}")
        print(f"First 5 coordinates: {target_ray[:5]}")
    elif ray_source == "spurious":
        ray_index = config['experiment']['target_ray_index']
        print(f"Loading spurious ray {ray_index} (from SSA+MMI, not in true HEC)")
        target_ray = load_spurious_ray(ray_index)
        print(f"Target ray shape: {target_ray.shape}")
        print(f"First 5 coordinates: {target_ray[:5]}")
        print(f"Note: This ray is from n5 (31 dimensions)")
    elif ray_source == "experiment4":
        ray_index = config['experiment']['target_ray_index']
        print(f"Loading experiment4 S6-unique ray {ray_index} (from 3M enumeration)")
        target_ray = load_experiment4_ray(ray_index)
        print(f"Target ray shape: {target_ray.shape}")
        print(f"First 5 coordinates: {target_ray[:5]}")
        print(f"Note: This ray is from n5 experiment4 (8 S6-unique rays from 3M vertices)")
    elif ray_source == "sa_cone":
        ray_index = config['experiment']['target_ray_index']
        print(f"Loading SA cone ray {ray_index} (from SA cone extremal rays)")
        target_ray = load_sa_cone_ray(ray_index)
        print(f"Target ray shape: {target_ray.shape}")
        print(f"First 5 coordinates: {target_ray[:5]}")
        print(f"Note: SA cone contains 208 extremal rays (156 valid HEC, 52 spurious)")
    elif ray_source == "custom" or ray_source == "line_interpolation":
        # Handle custom w_optimal vector from configuration
        if 'w_optimal' in config['experiment']:
            print(f"Loading custom w_optimal vector from configuration")
            target_ray = np.array(config['experiment']['w_optimal'])
            print(f"Target ray shape: {target_ray.shape}")
            print(f"Sum of components: {np.sum(target_ray):.10f}")
            if len(target_ray) >= 5:
                print(f"First 5 coordinates: {target_ray[:5]}")
            else:
                print(f"All coordinates: {target_ray}")
            if 't_value' in config['experiment']:
                print(f"Interpolation parameter t = {config['experiment']['t_value']}")
            # For custom sources, use point_id if target_ray_index not provided
            if 'target_ray_index' not in config['experiment']:
                config['experiment']['target_ray_index'] = config['experiment'].get('point_id', 0)
        else:
            raise ValueError(f"w_optimal not found in configuration for ray_source='{ray_source}'")
    else:
        raise ValueError(f"Unknown ray source: {ray_source}")
    
    return target_ray

def run_hecenv_config_experiment(config_path):
    """Run HECenv_parallel experiment using configuration file"""
    
    # Load configuration
    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")
    print(f"Experiment: {config['experiment']['name']}")
    
    # Setup output directory
    output_dir = config['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Setup target ray
    target_ray = setup_target_ray(config)

    # Auto-detect n from ray dimension
    ray_dim = len(target_ray)
    # Solve: 2^n - 1 = ray_dim => n = log2(ray_dim + 1)
    n_detected = int(np.log2(ray_dim + 1))

    # Setup training parameters
    training_config = config['training']

    # Override config n if detecting from spurious or experiment4 rays (n=5)
    if config['experiment']['target_ray_source'] in ['spurious', 'experiment4']:
        if n_detected != 5:
            print(f"Warning: Ray dimension {ray_dim} doesn't match expected n=5")
        training_config['n'] = n_detected
        print(f"Auto-detected n={n_detected} from ray dimension")
    elif 'n' not in training_config:
        training_config['n'] = n_detected
        print(f"Auto-detected n={n_detected} from ray dimension")

    env_config = config.get('environment', {})  # Make environment optional with empty default
    
    # Setup device
    device = env_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup precision (default to float32 for backward compatibility)
    precision = env_config.get('precision', 'float32')
    print(f"Using precision: {precision}")
    
    # Setup seed early so it can be included in args
    seed = env_config.get('seed', 42)
    
    # Set target tensor dtype based on precision
    target_dtype = torch.float64 if precision == 'float64' else torch.float32
    
    # Create experiment arguments dictionary (matching run_stage expectations)
    args = {
        'stage': 0,  # Start from stage 0
        'dir': output_dir,
        'batch_size': training_config['batch_size'],
        'max_training_cycles': training_config.get('n_iter', 5000),
        'x_value': 1.0,  # Default value for coordinate perturbation
        'target': torch.tensor(target_ray, dtype=target_dtype),
        'device': device,
        'precision': precision,  # Pass precision to HECenv
        'dt': training_config.get('dt', 1.0),
        'nnn': training_config['n'],
        'NN': training_config['N'],
        'lr': training_config['lr'],
        'epsilon': training_config.get('epsilon', 0.1),
        'factor': training_config.get('factor', 0.7),
        'keep_best_n': training_config.get('keep_best_n', 5),
        'dS': training_config.get('dS', 0.001),
        'gradient_method': training_config.get('gradient_method', 'scipy'),
        'skip_gradient': training_config.get('skip_gradient', False),
        'n_gradient_samples': training_config.get('n_gradient_samples', 100),
        'name': config['experiment'].get('run_name', f"newray_{config['experiment']['target_ray_index']}_classification"),
        'hidden_dim': training_config.get('hidden_dim', 128),
        'num_layers': training_config.get('num_layers', 3),
        'rollout_length': training_config.get('rollout_length', 100),
        'exploration_noise': training_config.get('exploration_noise', 0.1),
        'normalize_weights': training_config.get('normalize_weights', True),
        'target_magnitude_factor': training_config.get('target_magnitude_factor', 0.5),
        'enable_early_stopping': training_config.get('enable_early_stopping', True),
        'early_stop_patience': training_config.get('early_stop_patience', 5),
        'plot_update_frequency': training_config.get('plot_update_frequency', 5),
        'gpu_memory_fraction': training_config.get('gpu_memory_fraction', 0.95),
        'seed': seed,  # Pass seed to HECenv
        'initial_weights': config['experiment'].get('initial_weights', None)  # Pass initial weights if provided
    }
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Using random seed: {seed}")
    
    # Save configuration and parameters
    timestamp = datetime.now().isoformat()
    experiment_info = {
        'timestamp': timestamp,
        'config_path': config_path,
        'config': config,
        'args': {k: v.tolist() if isinstance(v, (np.ndarray, torch.Tensor)) else str(v) if hasattr(v, '__str__') and not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v for k, v in args.items()},
        'target_ray_info': {
            'source': config['experiment']['target_ray_source'],
            'index': config['experiment']['target_ray_index'],
            'shape': target_ray.shape,
            'first_5_coords': target_ray[:5].tolist()
        }
    }
    
    with open(os.path.join(output_dir, 'experiment_info.json'), 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    print(f"Experiment info saved to: {os.path.join(output_dir, 'experiment_info.json')}")
    print("\nStarting JAX-accelerated HECenv_parallel RL training...")
    print("Using JAX-optimized HEC environment for ~334x speedup")
    target_type = {
        'new': 'New',
        'genuine_new': 'Genuine New',
        'original': 'Original',
        'spurious': 'Spurious (SSA+MMI)',
        'experiment4': 'Experiment4 S6-unique',
        'custom': 'Custom',
        'line_interpolation': 'Line Interpolation'
    }.get(config['experiment']['target_ray_source'], 'Unknown')
    print(f"Target: {target_type} ray {config['experiment']['target_ray_index']}")
    print(f"Parameters: n={training_config['n']}, N={training_config['N']}, batch_size={training_config['batch_size']}")
    print(f"Learning rate: {training_config['lr']}, Max iterations: {training_config.get('n_iter', 5000)}")
    
    # Run the experiment
    try:
        # Check if multi-run mode is enabled
        multi_run = training_config.get('multi_run', False)
        num_runs = training_config.get('num_runs', 1)

        if multi_run and num_runs > 1:
            print(f"\n{'='*60}")
            print(f"MULTI-RUN MODE: Running {num_runs} independent trials")
            print(f"{'='*60}\n")

            all_rewards = []
            all_results = []

            for run_idx in range(num_runs):
                print(f"\n{'─'*60}")
                print(f"RUN {run_idx + 1}/{num_runs}")
                print(f"{'─'*60}")

                # Update seed for each run to get different initialization
                run_args = args.copy()
                run_args['seed'] = args['seed'] + run_idx if args.get('seed') is not None else run_idx
                run_args['run_idx'] = run_idx + 1  # Pass run number (1-indexed)

                # Run the stage
                run_results = run_stage(run_args)

                # Extract best_optimized_reward
                best_reward = run_results.get('best_optimized_reward', 0)
                all_rewards.append(best_reward)
                all_results.append(run_results)

                # Save this run's results immediately (in case job times out)
                run_results_path = os.path.join(output_dir, f'results_run_{run_idx + 1}.json')
                with open(run_results_path, 'w') as f:
                    json.dump(run_results, f, indent=2)
                print(f"Run {run_idx + 1} results saved to: {run_results_path}")

                print(f"Run {run_idx + 1} best_optimized_reward: {best_reward:.6f}")

                # Compute progressive statistics after each run
                current_max = float(np.max(all_rewards))
                current_mean = float(np.mean(all_rewards))
                current_std = float(np.std(all_rewards))

                print(f"\n{'─'*60}")
                print(f"PROGRESSIVE STATISTICS (after {run_idx + 1}/{num_runs} runs)")
                print(f"{'─'*60}")
                print(f"Rewards so far: {[f'{r:.6f}' for r in all_rewards]}")
                print(f"Current max:  {current_max:.6f}")
                print(f"Current mean: {current_mean:.6f}")
                print(f"Current std:  {current_std:.6f}")
                print(f"{'─'*60}")

                # Save progressive results after each run
                progressive_results = {
                    'completed_runs': run_idx + 1,
                    'total_runs': num_runs,
                    'all_rewards': all_rewards,
                    'max_reward': current_max,
                    'mean_reward': current_mean,
                    'std_reward': current_std,
                    'timestamp': datetime.now().isoformat()
                }
                progressive_path = os.path.join(output_dir, 'progressive_results.json')
                with open(progressive_path, 'w') as f:
                    json.dump(progressive_results, f, indent=2)

            # Compute final statistics
            max_reward = float(np.max(all_rewards))
            mean_reward = float(np.mean(all_rewards))
            std_reward = float(np.std(all_rewards))

            print(f"\n{'='*60}")
            print(f"FINAL MULTI-RUN STATISTICS")
            print(f"{'='*60}")
            print(f"All rewards: {[f'{r:.6f}' for r in all_rewards]}")
            print(f"Max reward:  {max_reward:.6f}")
            print(f"Mean reward: {mean_reward:.6f}")
            print(f"Std reward:  {std_reward:.6f}")
            print(f"{'='*60}\n")

            # Create combined results with statistics
            results = all_results[0].copy()  # Use first run as base
            results['multi_run_statistics'] = {
                'all_rewards': all_rewards,
                'max_reward': max_reward,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'num_runs': num_runs
            }
            # Use max reward as the representative best_optimized_reward
            results['best_optimized_reward'] = max_reward

            # Save all individual run results
            for run_idx, run_result in enumerate(all_results):
                run_results_path = os.path.join(output_dir, f'results_run_{run_idx + 1}.json')
                with open(run_results_path, 'w') as f:
                    json.dump(run_result, f, indent=2)
                print(f"Run {run_idx + 1} results saved to: {run_results_path}")
        else:
            # Single run mode (original behavior)
            results = run_stage(args)

        # Save results
        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_path}")

        # Determine if ray is inside or outside HEC
        final_reward = results.get('best_optimized_reward', 0)
        convergence_threshold = 1e-3  # Same threshold used in HECenv_parallel

        if final_reward > convergence_threshold:
            classification = "INSIDE HEC"
            print(f"\n🎯 RAY CLASSIFICATION: {classification}")
            print(f"   Final reward: {final_reward:.6f} > {convergence_threshold}")
            if multi_run and num_runs > 1:
                print(f"   Multi-run max: {results['multi_run_statistics']['max_reward']:.6f}")
                print(f"   Multi-run mean: {results['multi_run_statistics']['mean_reward']:.6f} ± {results['multi_run_statistics']['std_reward']:.6f}")
            print(f"   The RL agent successfully learned to reach this ray from random weights.")
        else:
            classification = "OUTSIDE HEC (or boundary)"
            print(f"\n❌ RAY CLASSIFICATION: {classification}")
            print(f"   Final reward: {final_reward:.6f} ≤ {convergence_threshold}")
            if multi_run and num_runs > 1:
                print(f"   Multi-run max: {results['multi_run_statistics']['max_reward']:.6f}")
                print(f"   Multi-run mean: {results['multi_run_statistics']['mean_reward']:.6f} ± {results['multi_run_statistics']['std_reward']:.6f}")
            print(f"   The RL agent could not reliably reach this ray from random weights.")

        # Save classification result
        classification_result = {
            'ray_source': config['experiment']['target_ray_source'],
            'ray_index': config['experiment']['target_ray_index'],
            'classification': classification,
            'final_reward': final_reward,
            'convergence_threshold': convergence_threshold,
            'timestamp': timestamp
        }

        # Add multi-run statistics if available
        if multi_run and num_runs > 1:
            classification_result['multi_run_statistics'] = results['multi_run_statistics']
        
        classification_path = os.path.join(output_dir, 'classification.json')
        with open(classification_path, 'w') as f:
            json.dump(classification_result, f, indent=2)
        
        print(f"Classification saved to: {classification_path}")
        
        return classification_result
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        error_info = {
            'error': str(e),
            'timestamp': timestamp,
            'config_path': config_path
        }
        
        error_path = os.path.join(output_dir, 'error.json')
        with open(error_path, 'w') as f:
            json.dump(error_info, f, indent=2)
        
        raise

def process_config_directory(config_dir, output_base_dir=None):
    """Process all JSON config files in a directory sequentially"""
    import glob
    from pathlib import Path
    import time
    
    # Find all JSON files in the directory
    config_pattern = os.path.join(config_dir, "*.json")
    config_files = sorted(glob.glob(config_pattern))
    
    if not config_files:
        print(f"No JSON configuration files found in {config_dir}")
        return []
    
    print(f"\nFound {len(config_files)} configuration files in {config_dir}")
    print("Will process them sequentially using JAX-accelerated HECenv")
    print("="*60)
    
    results = []
    start_time = time.time()
    
    for idx, config_path in enumerate(config_files, 1):
        print(f"\n[{idx}/{len(config_files)}] Processing: {os.path.basename(config_path)}")
        print("-"*40)
        
        try:
            # Load config to potentially override output directory
            config = load_config(config_path)
            
            # Override output directory if specified
            if output_base_dir:
                config_name = Path(config_path).stem
                config['experiment']['output_dir'] = os.path.join(output_base_dir, config_name)
                
                # Save modified config temporarily
                temp_config_path = f"/tmp/temp_config_{os.getpid()}_{idx}.json"
                with open(temp_config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Run with modified config
                result = run_hecenv_config_experiment(temp_config_path)
                
                # Clean up temp file
                os.remove(temp_config_path)
            else:
                # Run with original config path
                result = run_hecenv_config_experiment(config_path)
            
            result['config_file'] = os.path.basename(config_path)
            results.append(result)
            
            print(f"✅ Completed: {result['classification']} (reward: {result['final_reward']:.6f})")
            
        except Exception as e:
            print(f"❌ Error processing {config_path}: {e}")
            results.append({
                'config_file': os.path.basename(config_path),
                'classification': 'ERROR',
                'final_reward': -1,
                'error': str(e)
            })
    
    # Print summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r.get('classification') != 'ERROR')
    inside_count = sum(1 for r in results if 'INSIDE' in r.get('classification', ''))
    outside_count = sum(1 for r in results if 'OUTSIDE' in r.get('classification', ''))
    
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Total configs processed: {len(config_files)}")
    print(f"Successful: {successful}/{len(config_files)}")
    print(f"Inside HEC: {inside_count}")
    print(f"Outside HEC: {outside_count}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average time per config: {total_time/len(config_files):.1f} seconds")
    
    # Save batch summary if output directory specified
    if output_base_dir:
        summary_path = os.path.join(output_base_dir, 'batch_summary.json')
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config_directory': config_dir,
            'total_configs': len(config_files),
            'successful': successful,
            'inside_hec': inside_count,
            'outside_hec': outside_count,
            'total_time_seconds': total_time,
            'results': results
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nBatch summary saved to: {summary_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run HECenv_parallel RL experiment with configuration file(s)')
    
    # Make arguments mutually exclusive
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', help='Path to single configuration JSON file')
    group.add_argument('--config-dir', help='Directory containing multiple configuration JSON files')
    
    parser.add_argument('--output-dir', help='Override output directory from config')
    
    args = parser.parse_args()
    
    if args.config_dir:
        # Process all configs in directory
        results = process_config_directory(args.config_dir, args.output_dir)
        
        if results:
            print("\nFinal Results Summary:")
            for r in results:
                print(f"  {r['config_file']}: {r['classification']} (reward: {r['final_reward']:.6f})")
    else:
        # Process single config file (original behavior)
        config = load_config(args.config)
        
        if args.output_dir:
            config['experiment']['output_dir'] = args.output_dir
        
        # Run experiment
        result = run_hecenv_config_experiment(args.config)
        
        print(f"\nExperiment completed successfully!")
        print(f"Classification: {result['classification']}")
        print(f"Final reward: {result['final_reward']:.6f}")

if __name__ == "__main__":
    main()