"""
JAX optimization modules for HEC environment.

This package provides JAX-accelerated implementations for:
- Sfromw computation (~334x speedup over pure NumPy)
- Scipy optimization (~42x speedup with JAX gradients)
"""

from .HECenv_JAX import (
    HECenv_JAX,
    HECenv_JAX_optimized,
    create_optimized_sfromw_jax,
    Sfromw_JAX_optimized,
    torch_to_jax,
    jax_to_torch,
)

from .scipy_search_optimized import (
    scipy_search_optimized,
    get_available_cpus_optimized,
    scipy_search_wrapper_optimized,
    parallel_scipy_optimization_optimized,
)

__all__ = [
    'HECenv_JAX',
    'HECenv_JAX_optimized',
    'create_optimized_sfromw_jax',
    'Sfromw_JAX_optimized',
    'torch_to_jax',
    'jax_to_torch',
    'scipy_search_optimized',
    'get_available_cpus_optimized',
    'scipy_search_wrapper_optimized',
    'parallel_scipy_optimization_optimized',
]
