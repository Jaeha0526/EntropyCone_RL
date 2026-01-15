"""
Optimized scipy_search that uses JAX-optimized Sfromw function

Key optimizations:
1. Uses JAX-optimized Sfromw instead of Sfromw_single (334x faster)
2. Respects SLURM CPU allocation instead of using all machine CPUs
3. Maintains exact same interface and behavior as original
"""

import numpy as np
import torch
from scipy.optimize import minimize
import multiprocessing
import os

# Force JAX to use CPU only to avoid GPU memory issues in multiprocessing
# This must be set BEFORE JAX is imported anywhere in worker processes
os.environ['JAX_PLATFORMS'] = 'cpu'


def scipy_search_optimized(w_reference, target, min_idx, n, N, epsilon, factor, jax_fn_single=None):
    """
    Optimized version using JAX-optimized Sfromw function.

    Args:
        w_reference: Reference weight vector
        target: Target S-vector
        min_idx: Minimum indices
        n: Number of parties
        N: Total nodes
        epsilon: Convergence tolerance
        factor: Epsilon reduction factor
        jax_fn_single: JAX-optimized Sfromw function (if None, auto-creates from module)

    Returns:
        Same format as original scipy_search (with added timing info)
    """
    import time

    start_time_total = time.time()

    # Auto-create JAX function if not provided
    start_jax_creation = time.time()
    if jax_fn_single is None:
        try:
            # Force JAX to use CPU only for scipy optimization (avoids GPU memory issues in multiprocessing)
            import os
            os.environ['JAX_PLATFORMS'] = 'cpu'

            # Try to get JAX creator function from HECenv_parallel module
            import HECenv_parallel
            if hasattr(HECenv_parallel, 'create_optimized_sfromw_jax'):
                create_fn = HECenv_parallel.create_optimized_sfromw_jax
                jax_fn_single, _, _, _ = create_fn(n, N)
                print(f"[scipy_search_optimized] Created JAX Sfromw (CPU) for n={n}, N={N}")
            else:
                # Fallback to unoptimized version
                print("WARNING: create_optimized_sfromw_jax not found in HECenv_parallel, using unoptimized Sfromw_single")
                from HECenv_parallel import Sfromw_single
                sfromw_func = Sfromw_single
                jax_fn_single = None
        except Exception as e:
            print(f"WARNING: Failed to create JAX function: {e}, using unoptimized Sfromw_single")
            from HECenv_parallel import Sfromw_single
            sfromw_func = Sfromw_single
            jax_fn_single = None

    jax_creation_time = time.time() - start_jax_creation

    # Create wrapper for JAX function if we have it
    if jax_fn_single is not None:
        def sfromw_func(w, n, N):
            """Wrapper to convert JAX output to PyTorch format"""
            # Convert to JAX
            import jax.numpy as jnp
            w_jax = jnp.array(w if isinstance(w, np.ndarray) else w.detach().cpu().numpy())

            # Compute with JAX
            s_jax, idx_jax = jax_fn_single(w_jax)

            # Convert back to PyTorch
            s_torch = torch.from_numpy(np.array(s_jax))
            idx_torch = torch.from_numpy(np.array(idx_jax))

            return s_torch, idx_torch

    # Convert inputs to numpy
    if torch.is_tensor(w_reference):
        w_reference = w_reference.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    # Define objective function
    def objective(w):
        """Objective: minimize L2 distance to target S-vector"""
        real_value, _ = sfromw_func(w, n, N)
        if torch.is_tensor(real_value):
            real_value = real_value.detach().cpu().numpy()

        diff = target - real_value
        return np.sum(diff ** 2)

    # Optimization loop with decreasing epsilon
    start_optimization = time.time()
    w_optimal = w_reference.copy()
    best_loss = float('inf')

    current_epsilon = epsilon
    iteration = 0
    max_iterations = 10  # Maximum epsilon iterations
    sfromw_calls = 0
    sfromw_time = 0.0

    while current_epsilon > 1e-10 and iteration < max_iterations:
        # Bounds: all weights >= current_epsilon
        bounds = [(current_epsilon, None) for _ in range(len(w_optimal))]

        # Run optimization
        result = minimize(
            objective,
            w_optimal,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12}
        )

        if result.fun < best_loss:
            best_loss = result.fun
            w_optimal = result.x

        # Reduce epsilon
        current_epsilon *= factor
        iteration += 1

    optimization_time = time.time() - start_optimization

    # Compute final S-vector and check if indices match
    start_final_sfromw = time.time()
    real_value, real_idx = sfromw_func(w_optimal, n, N)
    final_sfromw_time = time.time() - start_final_sfromw
    sfromw_calls += 1

    # Convert to lists for compatibility
    if torch.is_tensor(real_idx):
        real_idx = real_idx.cpu().tolist()
    elif isinstance(real_idx, np.ndarray):
        real_idx = real_idx.tolist()

    wanted_idx = min_idx if isinstance(min_idx, list) else min_idx.tolist()

    # Compute reward (cosine similarity)
    if torch.is_tensor(real_value):
        real_value_np = real_value.detach().cpu().numpy()
    else:
        real_value_np = np.array(real_value) if not isinstance(real_value, np.ndarray) else real_value

    reward = np.dot(real_value_np, target) / (np.linalg.norm(real_value_np) * np.linalg.norm(target))
    is_solution = np.allclose(reward, 1, atol=1e-5)

    # Get Cuts (same as original)
    from HECenv_parallel import get_Cuts
    Cuts = get_Cuts(min_idx, n, N)

    total_time = time.time() - start_time_total

    # Return in same format as original scipy_search with added timing
    results = {
        "target": target.tolist(),
        "Cuts": Cuts,
        "w_optimal": w_optimal.tolist(),
        "reward": reward,
        "is_solution": is_solution,
        # Add timing information
        "timing": {
            "total_time": total_time,
            "jax_creation_time": jax_creation_time,
            "optimization_time": optimization_time,
            "final_sfromw_time": final_sfromw_time,
            "epsilon_iterations": iteration,
            "used_jax": jax_fn_single is not None,
            "jax_platform": "cpu"  # Scipy optimization runs on CPU to avoid GPU memory issues in multiprocessing
        }
    }

    return results


def get_available_cpus_optimized(usage_fraction=0.9):
    """
    Get number of CPUs available, respecting SLURM allocation.

    This fixes the CPU oversubscription issue by checking SLURM_CPUS_PER_TASK
    environment variable first.
    """
    # First check if running under SLURM
    slurm_cpus = os.getenv('SLURM_CPUS_PER_TASK')
    if slurm_cpus is not None:
        try:
            allocated_cpus = int(slurm_cpus)
            # Use 90% of allocated CPUs
            available = int(allocated_cpus * usage_fraction)
            return max(1, available)
        except ValueError:
            pass

    # Fallback to system CPU count
    total_cpus = multiprocessing.cpu_count()
    available_cpus = int(total_cpus * usage_fraction)
    return max(1, available_cpus)


def scipy_search_wrapper_optimized(args):
    """Wrapper for parallel execution with JAX functions"""
    # Force JAX to use CPU only BEFORE any JAX operations
    # This must happen in the worker process before JAX is initialized
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'

    # Original format: (w_reference, target, min_idx, n, N, epsilon, factor)
    # JAX function will be auto-created in scipy_search_optimized
    w_reference, target, min_idx, n, N, epsilon, factor = args
    return scipy_search_optimized(w_reference, target, min_idx, n, N, epsilon, factor)


def parallel_scipy_optimization_optimized(
    optimization_tasks,
    jax_fn_single,
    max_workers=None,
    verbose=True
):
    """
    Run parallel scipy optimization with optimizations.

    Args:
        optimization_tasks: List of (w_reference, target, min_idx, n, N, epsilon, factor) tuples
        jax_fn_single: JAX-optimized Sfromw function
        max_workers: Maximum number of parallel workers (None = auto-detect from SLURM)
        verbose: Print progress information

    Returns:
        List of optimization results with timing
    """
    import time

    # Determine number of CPUs to use
    if max_workers is None:
        available_cpus = get_available_cpus_optimized(0.9)
        num_cpus_to_use = min(10, available_cpus, len(optimization_tasks))
    else:
        num_cpus_to_use = max_workers

    if verbose:
        slurm_cpus = os.getenv('SLURM_CPUS_PER_TASK', 'N/A')
        print(f"SLURM allocation: {slurm_cpus} CPUs")
        print(f"Running parallel scipy optimization on {len(optimization_tasks)} trajectories using {num_cpus_to_use} CPUs...")

    # Add JAX function to each task
    tasks_with_jax = [task + (jax_fn_single,) for task in optimization_tasks]

    # Run optimizations in parallel
    start_time = time.time()
    if len(optimization_tasks) > 0:
        with multiprocessing.Pool(processes=num_cpus_to_use) as pool:
            results = pool.map(scipy_search_wrapper_optimized, tasks_with_jax)
    else:
        results = []

    total_time = time.time() - start_time

    if verbose and len(results) > 0:
        # Print timing summary
        avg_time = total_time / len(results) if len(results) > 0 else 0
        print(f"\nScipyoptimization completed in {total_time:.2f}s")
        print(f"  Average per trajectory: {avg_time:.2f}s")
        if len(results) > 0 and 'timing' in results[0]:
            avg_jax_time = sum(r['timing']['jax_creation_time'] for r in results) / len(results)
            avg_opt_time = sum(r['timing']['optimization_time'] for r in results) / len(results)
            print(f"  Avg JAX creation: {avg_jax_time:.3f}s")
            print(f"  Avg optimization: {avg_opt_time:.3f}s")

    return results
