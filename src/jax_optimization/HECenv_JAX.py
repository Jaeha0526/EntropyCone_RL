"""
HECenv_parallel_JAX_optimized_final_v2.py

This version properly uses JAX autodiff for gradients while avoiding NaN issues.
The key is to use JAX's grad function correctly and handle the tensor conversions properly.
"""

from collections import defaultdict
from typing import Optional
import os
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
import json
import numpy as np
from scipy.optimize import minimize
from torchrl.envs import ParallelEnv
from torchrl.envs.transforms import GrayScale
import time

from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import multiprocessing

# JAX imports
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
from functools import partial

# DLPack conversion utilities
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


#########################
# DLPack conversion utilities
#########################

def torch_to_jax(tensor):
    """Convert PyTorch tensor to JAX array via DLPack (zero-copy when possible)"""
    return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor.contiguous()))

def jax_to_torch(array, device='cuda'):
    """Convert JAX array to PyTorch tensor via DLPack (zero-copy when possible)"""
    dlpack_capsule = jax_dlpack.to_dlpack(array)
    tensor = torch_dlpack.from_dlpack(dlpack_capsule)
    return tensor.to(device)


#########################
# Optimized JAX functions
#########################

def combination_to_index_np(i, j, N):
    """Convert node pair to edge index (NumPy version for pre-computation)"""
    a, b = min(i, j), max(i, j)
    return a*(N-1) - a*(a-1)//2 + b - a - 1

def precompute_cut_matrices(n, N):
    """
    Pre-compute matrices that represent which edges are in each cut.
    Returns a matrix of shape (2^n-1, 2^(N-n-1), N*(N-1)//2)
    """
    num_edges = (N * (N - 1)) // 2
    num_i = 2**n - 1
    num_j = 2**(N-n-1)
    
    cut_matrix = np.zeros((num_i, num_j, num_edges), dtype=np.bool_)
    
    for i in range(1, 2**n):
        # Generate W configuration for this i
        W = []
        ihere = i
        for j_bit in range(1, n+1):
            if ihere % 2 == 1:
                W.append(j_bit)
            ihere = ihere // 2
        
        for j in range(num_j):
            # Generate Where configuration
            Where = W.copy()
            jhere = j
            for k in range(N-n-1):
                if jhere % 2 == 1:
                    Where.append(N-k-1)
                jhere = jhere // 2
            
            # Mark edges in the cut
            inside = set(Where)
            outside = set(range(1, N+1)) - inside
            
            for out_node in outside:
                for in_node in inside:
                    # Convert to 0-based indices
                    edge_idx = combination_to_index_np(out_node-1, in_node-1, N)
                    cut_matrix[i-1, j, edge_idx] = True
    
    return cut_matrix

def create_optimized_sfromw_jax(n, N, dtype=jnp.float32):
    """Create optimized JAX function using matrix operations
    
    Args:
        n: Number of parties
        N: Total nodes including holographic
        dtype: JAX dtype to use (jnp.float32 or jnp.float64), defaults to jnp.float32
    """
    # Pre-compute cut matrix at function creation time
    cut_matrix = precompute_cut_matrices(n, N)
    # Convert to specified dtype for JAX operations
    cut_matrix_jax = jnp.array(cut_matrix, dtype=dtype)
    
    @jit
    def sfromw_single(w):
        """Process single weight vector using matrix multiplication"""
        # Ensure w is 1D
        w = w.flatten()
        
        # Compute all cut weights at once: (num_i, num_j)
        cut_weights = jnp.dot(cut_matrix_jax, w)
        
        # Find minimum across j dimension
        s_vector = jnp.min(cut_weights, axis=1)
        min_indices = jnp.argmin(cut_weights, axis=1)
        
        return s_vector, min_indices
    
    # Create batched version
    sfromw_batch = jit(vmap(sfromw_single))
    
    # Create versions with gradient computation using value_and_grad
    def sfromw_with_vjp_single(w, v):
        """Compute S-vector and vector-Jacobian product for single input"""
        def f(w):
            s, _ = sfromw_single(w)
            return s
        
        s, vjp_fn = jax.vjp(f, w)
        grad_w = vjp_fn(v)[0]
        return s, grad_w
    
    def sfromw_with_vjp_batch(w, v):
        """Compute S-vector and vector-Jacobian product for batch input"""
        return vmap(sfromw_with_vjp_single)(w, v)
    
    # JIT compile the VJP versions
    sfromw_vjp_single = jit(sfromw_with_vjp_single)
    sfromw_vjp_batch = jit(sfromw_with_vjp_batch)
    
    # Keep the old gradient functions for compatibility
    grad_sfromw_single = sfromw_vjp_single
    grad_sfromw_batch = sfromw_vjp_batch
    
    return sfromw_single, sfromw_batch, grad_sfromw_single, grad_sfromw_batch


#########################
# Import remaining functions from original
#########################

from HECenv_parallel import (
    Ws_from_index,
    combination_to_index,
    _reset,
    _make_spec,
    make_composite_from_td,
    _set_seed,
    gen_params,
    get_Cuts,
    scipy_search,
    gradient_search,
)


#########################
# Optimized autograd function for JAX integration
#########################

class OptimizedJAXSfromwFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, n, N, jax_fn_single, jax_fn_batch, grad_fn_single, grad_fn_batch):
        # Store for backward
        ctx.save_for_backward(w)
        ctx.n = n
        ctx.N = N
        ctx.grad_fn_single = grad_fn_single
        ctx.grad_fn_batch = grad_fn_batch
        ctx.w_shape = w.shape
        ctx.device = w.device
        
        # Convert to JAX
        w_np = w.detach().cpu().numpy()
        w_jax = jnp.array(w_np)
        
        # Compute with JAX
        if w.dim() > 1:
            s_jax, idx_jax = jax_fn_batch(w_jax)
        else:
            s_jax, idx_jax = jax_fn_single(w_jax)
        
        # Convert back to PyTorch
        s_np = np.array(s_jax)
        idx_np = np.array(idx_jax)
        s_torch = torch.from_numpy(s_np).to(device=w.device)
        idx_torch = torch.from_numpy(idx_np).to(device=w.device)
        
        return s_torch, idx_torch
    
    @staticmethod
    def backward(ctx, grad_output, grad_idx):
        if grad_output is None:
            return None, None, None, None, None, None, None
        
        w, = ctx.saved_tensors
        
        # Convert to JAX
        w_np = w.detach().cpu().numpy()
        w_jax = jnp.array(w_np)
        
        # Convert grad_output to JAX (this is the vector for VJP)
        grad_output_np = grad_output.detach().cpu().numpy()
        v_jax = jnp.array(grad_output_np)
        
        # Compute vector-Jacobian product
        if w.dim() > 1:
            _, grad_jax = ctx.grad_fn_batch(w_jax, v_jax)
        else:
            _, grad_jax = ctx.grad_fn_single(w_jax, v_jax)
        
        # Convert gradient to PyTorch
        grad_np = np.array(grad_jax)
        grad_w = torch.from_numpy(grad_np).to(device=ctx.device)
        
        # Reshape if needed to match original w shape
        if grad_w.shape != ctx.w_shape:
            grad_w = grad_w.reshape(ctx.w_shape)
        
        return grad_w, None, None, None, None, None, None


#########################
# JAX-optimized Sfromw function
#########################

def Sfromw_JAX_optimized(w, n, N, jax_fn_single, jax_fn_batch, grad_fn_single, grad_fn_batch):
    """
    JAX-optimized version of Sfromw that maintains exact output compatibility
    """
    # Handle tensor conversion
    if torch.is_tensor(n):
        n_scalar = n.flatten()[0].item() if n.numel() > 1 else n.item()
    else:
        n_scalar = n
    
    if torch.is_tensor(N):
        N_scalar = N.flatten()[0].item() if N.numel() > 1 else N.item()
    else:
        N_scalar = N
    
    # Use autograd function for proper gradient support
    if w.requires_grad:
        return OptimizedJAXSfromwFunction.apply(w, n_scalar, N_scalar, 
                                                jax_fn_single, jax_fn_batch,
                                                grad_fn_single, grad_fn_batch)
    else:
        # Direct computation without gradient tracking
        w_np = w.detach().cpu().numpy()
        w_jax = jnp.array(w_np)
        
        if w.dim() > 1:
            s_jax, idx_jax = jax_fn_batch(w_jax)
        else:
            s_jax, idx_jax = jax_fn_single(w_jax)
        
        s_np = np.array(s_jax)
        idx_np = np.array(idx_jax)
        s_torch = torch.from_numpy(s_np).to(device=w.device)
        idx_torch = torch.from_numpy(idx_np).to(device=w.device)
        
        return s_torch, idx_torch


#########################
# JAX-optimized step function
#########################

def _step_jax_optimized(tensordict, jax_fn_single, jax_fn_batch, grad_fn_single, grad_fn_batch):
    """
    JAX-optimized step function that maintains exact compatibility
    """
    # Get all the inputs
    n = tensordict["params", "n"]
    N = tensordict["params", "N"]
    w = tensordict["w"]
    wdot = tensordict["action"]
    dt = tensordict["params", "dt"]
    Svector_target = tensordict["params", "target"]
    
    # Update w with the action
    dt_unsqueezed = dt.unsqueeze(-1) if dt.dim() < wdot.dim() else dt
    w = w + wdot * dt_unsqueezed
    w = torch.clamp(w, min=0.0)
    
    # Optional: Normalize w vector
    normalize_weights = tensordict["params"].get("normalize_weights", False)
    if torch.is_tensor(normalize_weights):
        normalize_weights = normalize_weights.item() if normalize_weights.dim() == 0 else normalize_weights[0].item()
    if normalize_weights:
        if torch.is_tensor(N):
            N_val = int(N.flatten()[0].item())
        else:
            N_val = N
        num_weights = (N_val * (N_val - 1)) // 2

        # Get target_magnitude_factor from params
        target_mag_factor = tensordict["params"].get("target_magnitude_factor", 0.5)
        if torch.is_tensor(target_mag_factor):
            target_mag_factor = target_mag_factor.item() if target_mag_factor.dim() == 0 else target_mag_factor[0].item()

        target_magnitude = target_mag_factor * (num_weights ** 0.5)

        w_norm = torch.norm(w, dim=-1, keepdim=True)
        w = w / torch.clamp(w_norm, min=1e-8) * target_magnitude
    
    # Compute current S vector using JAX optimization
    Svector_current, min_idx = Sfromw_JAX_optimized(w, n, N, 
                                                     jax_fn_single, jax_fn_batch,
                                                     grad_fn_single, grad_fn_batch)
    
    # Fix dimensions if needed
    if Svector_target.dim() > Svector_current.dim():
        if Svector_target.dim() == 3 and Svector_current.dim() == 2:
            Svector_target = Svector_target[:, 0, :]
    
    # Compute difference
    Svector_difference = Svector_target - Svector_current
    
    # Compute reward (exactly as in original)
    eps = 1e-6
    if Svector_current.dim() > 1:
        Svector_target_norm = Svector_target / (torch.norm(Svector_target, dim=-1, keepdim=True) + eps)
        Svector_current_norm = Svector_current / (torch.norm(Svector_current, dim=-1, keepdim=True) + eps)
        reward = torch.sum(Svector_target_norm * Svector_current_norm, dim=-1)
        done = reward < 1e-3
        reward = reward.unsqueeze(-1)
    else:
        reward = torch.dot(Svector_target, Svector_current) / (torch.norm(Svector_target) * torch.norm(Svector_current) + eps)
        done = reward < 1e-3
        reward = reward.unsqueeze(-1)
    
    # Build output TensorDict
    out = TensorDict({
        "w": w,
        "Svector_current": Svector_current,
        "Svector_difference": Svector_difference,
        "params": tensordict["params"],
        "reward": reward,
        "min_idx": min_idx,
        "done": done,
    }, batch_size=tensordict.batch_size)
    
    return out


#########################
# Optimized HECenv class
#########################

class HECenv_JAX_optimized(EnvBase):
    """JAX-optimized HEC environment with proper gradient computation"""
    
    metadata = {}
    batch_locked = False
    _optimized_jax_version = True  # Identifier for debug checking
    
    def __init__(self, dt, target=None, n=6, N=10, seed=None, device="cpu", precision="float32", initial_weights=None, **kwargs):
        self.n = n
        self.N = N
        self.dt = dt
        self.target = target
        self.device = device
        self.precision = precision
        self.initial_weights = initial_weights  # Store initial weights if provided

        # Set JAX precision if using float64
        if precision == "float64":
            import jax
            jax.config.update("jax_enable_x64", True)
            dtype = jnp.float64
            print("JAX 64-bit mode enabled for high precision")
        else:
            dtype = jnp.float32
        
        # Pre-create optimized JAX functions with specified precision
        print(f"Pre-computing cut matrices for n={n}, N={N} with {precision} precision...")
        start = time.time()
        (self.jax_fn_single, self.jax_fn_batch, 
         self.grad_fn_single, self.grad_fn_batch) = create_optimized_sfromw_jax(n, N, dtype=dtype)
        print(f"Pre-computation complete in {time.time() - start:.1f}s")
        
        # Initialize attributes
        self.num_weights = (N * (N - 1)) // 2
        
        # Generate initial params and setup specs  
        td_params = self.gen_params(dt=dt, target=target, n=n, N=N, device=device, precision=precision)
        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        
        # Set seed
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
            print(f"No seed provided, using {seed}")
        self.set_seed(seed)
    
    # DO NOT override step() - let TorchRL handle it
    
    def _step(self, tensordict):
        """Use the JAX-optimized step function"""
        return _step_jax_optimized(tensordict, 
                                  self.jax_fn_single, self.jax_fn_batch,
                                  self.grad_fn_single, self.grad_fn_batch)
    
    # Use all original methods without modification
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec
    _reset = _reset
    _set_seed = _set_seed


# Use the optimized version as the default
HECenv_JAX = HECenv_JAX_optimized