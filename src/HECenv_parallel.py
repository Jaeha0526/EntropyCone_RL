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
from scipy.optimize import minimize  # Using scipy with analytical gradients for 10x speedup
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
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
from matplotlib import pyplot as plt
import multiprocessing
from functools import partial
import tempfile

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


#########################
# CPU allocation helpers
#########################

def get_available_cpus(usage_fraction=0.9):
    """
    Get the number of CPUs available for parallel processing.
    
    Args:
        usage_fraction: Fraction of total CPUs to use (default 0.9 for 90%)
    
    Returns:
        Number of CPUs to use
    """
    total_cpus = multiprocessing.cpu_count()
    available_cpus = int(total_cpus * usage_fraction)
    return max(1, available_cpus)  # Ensure at least 1 CPU


def scipy_search_wrapper(args):
    """
    Wrapper function for scipy_search to work with multiprocessing.
    Unpacks arguments and calls scipy_search.
    """
    w_reference, target, min_idx, n, N, epsilon, factor = args
    return scipy_search(w_reference, target, min_idx, n, N, epsilon, factor)


#########################
# Related to the step
#########################

def _step(tensordict):
        """
        Step the environment.
        """
        # print("="*50)
        # print("Step called")
        # print(f"[DEBUG] tensordict: {tensordict}")
        n = tensordict["params", "n"]
        N = tensordict["params", "N"]
        w = tensordict["w"]
        wdot = tensordict["action"]
        dt = tensordict["params", "dt"]
        Svector_target = tensordict["params", "target"]
        
        # print("="*50)
        # print("Debugging step")
        # print(f"[DEBUG] w shape: {w.shape}")
        # print(f"[DEBUG] wdot shape: {wdot.shape}")
        # print(f"[DEBUG] dt shape: {dt.shape}")
        # print(f"[DEBUG] Svector_target shape: {Svector_target.shape}")
        # print(f"[DEBUG] n: {n}")
        # print(f"[DEBUG] N: {N}")
        # print("="*50)
            
        # Update w with the action
        # Unsqueeze dt to match dimensions for broadcasting: [batch_size] → [batch_size, 1]
        # This allows proper pairwise multiplication with wdot [batch_size, N*(N-1)/2]
        dt_unsqueezed = dt.unsqueeze(-1)
        # print(f"[DEBUG] dt_unsqueezed shape: {dt_unsqueezed.shape}")
        
        # Multiply each batch element of wdot with corresponding dt
        w = w + wdot * dt_unsqueezed
        # Enforce positivity of w
        w = torch.clamp(w, min=0.0)
        
        # Optional: Normalize w vector to target magnitude for component scale ~0.5
        normalize_weights = tensordict["params"].get("normalize_weights", False)
        # Handle both batched and non-batched normalize_weights
        if torch.is_tensor(normalize_weights):
            normalize_weights = normalize_weights.item() if normalize_weights.dim() == 0 else normalize_weights[0].item()
        if normalize_weights:
            # Calculate target magnitude: target_magnitude_factor * sqrt(N*(N-1)/2)
            if torch.is_tensor(N):
                N_val = int(N.flatten()[0].item())  # Take first value from batched N
            else:
                N_val = N
            num_weights = (N_val * (N_val - 1)) // 2

            # Get target_magnitude_factor from params
            target_mag_factor = tensordict["params"].get("target_magnitude_factor", 0.5)
            if torch.is_tensor(target_mag_factor):
                target_mag_factor = target_mag_factor.item() if target_mag_factor.dim() == 0 else target_mag_factor[0].item()

            target_magnitude = target_mag_factor * (num_weights ** 0.5)

            w_norm = torch.norm(w, dim=-1, keepdim=True)
            w = w / torch.clamp(w_norm, min=1e-8) * target_magnitude  # Scale to target magnitude
        # print(f"[DEBUG] Updated w shape: {w.shape}")
        
        # Compute current S vector (Sfromw already handles tensor conversion internally)
        Svector_current, min_idx = Sfromw(w, n, N)
        
        # If target has more dimensions than current, reshape target
        if Svector_target.dim() > Svector_current.dim():
            if Svector_target.dim() == 3 and Svector_current.dim() == 2:
                # Take just the first slice if target is redundantly batched
                Svector_target = Svector_target[:, 0, :]
            
        # print(f"[DEBUG] Svector_current shape: {Svector_current.shape}")
        # print(f"[DEBUG] Svector_target shape after fix: {Svector_target.shape}")
        
        # Compute difference and reward
        Svector_difference = Svector_target - Svector_current
        
        eps = 1e-6
        # For batched inputs, we need to compute the reward for each batch
        if Svector_current.dim() > 1:
            # Normalize vectors for cosine similarity
            Svector_target_norm = Svector_target / (torch.norm(Svector_target, dim=-1, keepdim=True) + eps)
            Svector_current_norm = Svector_current / (torch.norm(Svector_current, dim=-1, keepdim=True) + eps)
            
            # Compute dot product along the last dimension
            reward = torch.sum(Svector_target_norm * Svector_current_norm, dim=-1)
            done = reward < 1e-3
            # Add extra dimension to match reward spec
            reward = reward.unsqueeze(-1)
        else:
            # Non-batched case
            reward = torch.dot(Svector_target, Svector_current) / (torch.norm(Svector_target) * torch.norm(Svector_current) + eps)
            done = reward < 1e-3
            # Add extra dimension to match reward spec
            reward = reward.unsqueeze(-1)
        
        out = TensorDict({
            "w": w,
            "Svector_current": Svector_current,
            "Svector_difference": Svector_difference,
            "params": tensordict["params"],
            "reward": reward,
            "min_idx": min_idx,
            "done": done,
        }, tensordict.shape)
        # print(f"[DEBUG] out['reward'] : {out['reward']}")
        # print(f"[DEBUG] out['done'] : {out['done']}")
        # print(f"[DEBUG] out['w'] shape: {out['w'].shape}")
        # print(f"[DEBUG] out['Svector_current'] shape: {out['Svector_current'].shape}")
        # print(f"[DEBUG] out['Svector_difference'] shape: {out['Svector_difference'].shape}")
        # print(f"[DEBUG] out['params'] shape: {out['params'].shape}")
        # print("="*50)
        return out
    
def Sfromw(w, n, N):
    """
    Fully vectorized conversion from w-space to s-space.
    Processes all batch elements in parallel on GPU.
    """
    # Convert n and N to Python integers if they are tensors
    if torch.is_tensor(n):
        n_val = int(n.item()) if n.dim() == 0 else int(n[0].item())
    else:
        n_val = n
        
    if torch.is_tensor(N):
        N_val = int(N.item()) if N.dim() == 0 else int(N[0].item())
    else:
        N_val = N
    
    # Handle both batched and non-batched inputs
    if torch.is_tensor(w) and w.dim() > 1:
        batch_shape = w.shape[:-1]
        batch_size = torch.prod(torch.tensor(batch_shape)).item()
        device = w.device
        
        # Flatten batch dimensions for vectorized processing
        w_flat = w.view(batch_size, -1)  # [batch_size, n_weights]
        
        # Preallocate outputs
        Svector = torch.zeros(batch_size, 2**n_val-1, device=device)
        min_indices = torch.zeros(batch_size, 2**n_val-1, dtype=torch.long, device=device)
        
        # Pre-compute ALL configurations for maximum vectorization
        num_i_configs = 2**n_val - 1  # 63 for n=6
        num_j_configs = 2**(N_val-n_val-1)  # 2 for N=8, n=6
        total_configs = num_i_configs * num_j_configs  # 126 total
        
        # Pre-allocate for all configurations
        all_CWs = torch.zeros(batch_size, num_i_configs, num_j_configs, device=device)
        
        # Generate all configurations upfront (this part is unavoidably sequential but tiny)
        config_indices = []
        for i in range(1, 2**n_val):
            # Generate W configuration for this i
            W = []
            ihere = i
            for j_bit in range(1, n_val+1):
                if ihere % 2 == 1:
                    W.append(j_bit)
                ihere = ihere // 2
            
            for j in range(num_j_configs):
                # Generate Where configuration
                Where = W.copy()
                jhere = j
                for k in range(N_val-n_val-1):
                    if jhere % 2 == 1:
                        Where.append(N_val-k-1)
                    jhere = jhere // 2
                
                config_indices.append((i-1, j, Where))
        
        # Vectorized computation for ALL configurations
        for i_idx, j_idx, Where in config_indices:
            all_CWs[:, i_idx, j_idx] = CWfromwW_vectorized(w_flat, Where, N_val)
        
        # Vectorized minimum finding across j dimension for all i and batches
        min_vals, min_idxs = torch.min(all_CWs, dim=2)  # [batch_size, num_i_configs]
        Svector = min_vals
        min_indices = min_idxs
        
        # Reshape back to original batch dimensions
        Svector = Svector.view(*batch_shape, 2**n_val-1)
        min_indices = min_indices.view(*batch_shape, 2**n_val-1)
        
        return Svector, min_indices
    else:
        # Non-batched case - use single version
        return Sfromw_single(w, n_val, N_val)

def Sfromw_single(w, n, N):
    """
    Non-batched version of Sfromw that processes a single example.
    """
    # Determine device based on w input
    device = w.device if torch.is_tensor(w) else torch.device('cpu') 
    Svector = torch.zeros(2**n-1, device=device)
    min_indices = torch.zeros(2**n-1, dtype=torch.long, device=device)
    
    # Add timing for debugging
    # start_sfromw = time.time()
    # print(f"[DEBUG_Sfromw_single] N: {N}")
    # print(f"[DEBUG_Sfromw_single] n: {n}")
    # print(f"[DEBUG_Sfromw_single] shape of w: {w.shape}")
    for i in range(1, 2**n):  # decide which I we are interested in
        W = []
        ihere = i
        for j in range(1, n+1):
            if ihere % 2 == 1:
                W.append(j)
            ihere = ihere // 2
        
        CWs = []
        for j in range(2**(N-n-1)):
            Where = W.copy()
            jhere = j
            for k in range(N-n-1):
                if jhere % 2 == 1:
                    Where.append(N-k-1)
                jhere = jhere // 2
            # print(f"[DEBUG] Where: {Where}")
            CWs.append(CWfromwW_single(w, Where, N, n))
        
        # Stack CWs into a tensor and use torch.min with keepdim=True to get both value and index
        CWs_tensor = torch.stack(CWs)
        min_val, min_idx = torch.min(CWs_tensor, dim=0)
        Svector[i-1] = min_val
        min_indices[i-1] = min_idx
        
    return Svector, min_indices

def Ws_from_index(index, n, N):
    # print(f"[DEBUG] index: {index}")
    # print(f"[DEBUG] index.shape: {index.shape}")
    # print(f"[DEBUG] n: {n}")
    # print(f"[DEBUG] N: {N}")
    if len(index) != 2**n-1:
        raise ValueError(f"index must be of length 2**n-1, but got {len(index)}")
    Ws = []
    for i in range(1,2**n):
        W = []
        ihere = i
        for j in range(1, n+1):
            if ihere % 2 == 1:
                W.append(j)
            ihere = ihere // 2
        
        windex = index[i-1]
        for k in range(N-n-1):
            if windex % 2 == 1:
                W.append(N-k-1)
            windex = windex // 2
        Ws.append(W)
            
    return Ws

def CWfromwW_vectorized(w_batch, W, N):
    """
    Fully vectorized version that processes all batch elements simultaneously.
    
    Args:
        w_batch: [batch_size, n_weights] - batch of weight vectors
        W: list - configuration indices
        N: int - total number of elements
    
    Returns:
        torch.Tensor: [batch_size] - CW values for all batch elements
    """
    device = w_batch.device
    batch_size = w_batch.shape[0]
    
    outside = [i-1 for i in range(1, N+1) if i not in W]
    inside = [i-1 for i in range(1, N+1) if i in W]
    
    # Pre-compute all indices for vectorized access
    indices = []
    for i in outside:
        for j in inside:
            indices.append(combination_to_index([i, j], N))
    
    if not indices:  # Handle empty case
        return torch.zeros(batch_size, device=device)
    
    # Vectorized computation: sum all relevant weights at once
    indices_tensor = torch.tensor(indices, device=device)
    CW = w_batch[:, indices_tensor].sum(dim=1)  # [batch_size, len(indices)] -> [batch_size]
    
    return CW

def CWfromwW(w, W, N, n):
    """
    Convert the state vector from the w-space to the s-space.
    Handles batched inputs.
    """
    # Convert N to Python integer if it's a tensor
    N_val = int(N.item()) if torch.is_tensor(N) else N
    # print(f"[DEBUG_CWfromwW] N: {N}")
    # print(f"[DEBUG_CWfromwW] N_val: {N_val}")
    # print(f"[DEBUG_CWfromwW] shape of w: {w.shape}")
    
    # Handle batch dimensions
    if torch.is_tensor(w) and w.dim() > 1:
        # Process each batch separately
        batch_size = list(w.shape[:-1])
        result = torch.zeros(*batch_size, device=w.device)
        
        if len(batch_size) == 1:
            for b in range(batch_size[0]):
                result[b] = CWfromwW_single(w[b], W, N_val, n)
        else:
            # Flatten batch dimensions for processing
            flat_batch_size = torch.prod(torch.tensor(batch_size)).item()
            flat_w = w.view(flat_batch_size, -1)
            flat_result = torch.zeros(flat_batch_size, device=w.device)
            
            for b in range(flat_batch_size):
                flat_result[b] = CWfromwW_single(flat_w[b], W, N_val, n)
            
            # Reshape back to original batch dimensions
            result = flat_result.view(*batch_size)
        
        return result
    else:
        # Non-batched case
        return CWfromwW_single(w, W, N_val, n)

def CWfromwW_single(w, W, N, n):
    """
    Non-batched version of CWfromwW that processes a single example.
    """
    # Determine device based on w input
    device = w.device if torch.is_tensor(w) else torch.device('cpu') 
    # print(f"[DEBUG_CWfromwW_single] N: {N}")
    outside = [i-1 for i in range(1, N+1) if i not in W]
    inside = [i-1 for i in range(1, N+1) if i in W]
    
    # Initialize CW as a tensor
    CW = torch.tensor(0.0, device=device)
    
    # Check if w is a tensor before indexing
    is_w_tensor = torch.is_tensor(w)
    # print(f"[DEBUG_CWfromwW_single] shape of w: {w.shape}")
    
    for i in outside:
        for j in inside:
            idx = combination_to_index([i, j], N)
            if is_w_tensor:
                # Perform tensor addition without .item()
                CW = CW + w[idx]
            else:
                # If w is not a tensor (e.g., list or ndarray), handle accordingly
                # Assuming it supports indexing like w[idx]
                CW = CW + torch.tensor(w[idx], device=device) # Convert to tensor if not already
    
    return CW
    
def combination_to_index(comb, N):
    """
    Convert a combination (unordered pair) to its lexicographic index
    
    Parameters:
    comb (tuple/list): The combination (e.g., [1,3])
    
    Returns:
    int: The lexicographic index (0-based) [1,3] -> 2
    """
    # Ensure the combination is sorted
    a, b = sorted(comb)
    
    # Formula for lexicographic index of combination (a,b) where a < b
    # This calculates how many combinations come before this one
    return int(a*(N-1) - a*(a-1)//2 + b - a - 1)

#########################
# Setup reset function
#########################
def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty():
        # if no ``tensordict`` is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input ``tensordict`` contains all the relevant
        # parameters to get started.
        # print("[DEBUG-RESET] No tensordict provided, generating params...")
        tensordict = self.gen_params(self.dt, self.target, self.n, self.N, batch_size=self.batch_size, device=self.device, precision=self.precision)
    else:
        pass
        # print("[DEBUG-RESET] Tensordict provided:")
        # print(f"[DEBUG-RESET] Shape: {tensordict.shape}")
        # if 'params' in tensordict and 'target' in tensordict['params']:
            # print(f"[DEBUG-RESET] Params target shape: {tensordict['params', 'target'].shape}")

    # Extract parameters
    N = tensordict["params", "N"]
    n = tensordict["params", "n"]
    Svector_target = tensordict["params", "target"]
    # print(f"[DEBUG-RESET] Extracted Svector_target shape: {Svector_target.shape}")
        
    # print(f"[DEBUG] N: {N}")
    # print(f"[DEBUG] n: {n}")
    # print(f"[DEBUG] batch_size: {self.batch_size}")
    # print(f"[DEBUG] tensordict.shape: {tensordict.shape}")
    
    # Initialize random weights with correct precision
    dtype = torch.float64 if self.precision == "float64" else torch.float32
    w = torch.rand(*tensordict.batch_size, (self.N*(self.N-1))//2, generator=self.rng, device=self.device, dtype=dtype)

    # Override with custom initial weights if provided
    if hasattr(self, 'initial_weights') and self.initial_weights is not None:
        w_init = torch.tensor(self.initial_weights, device=self.device, dtype=dtype)

        # Validate dimension compatibility with N
        expected_dim = (self.N * (self.N - 1)) // 2
        actual_dim = w_init.shape[-1] if w_init.dim() > 0 else 1
        if actual_dim != expected_dim:
            raise ValueError(
                f"initial_weights dimension mismatch: got {actual_dim}, expected {expected_dim} "
                f"for N={self.N} (N*(N-1)/2 = {self.N}*{self.N-1}/2 = {expected_dim})"
            )

        # Expand to batch size if needed
        if w_init.dim() == 1:
            w_init = w_init.unsqueeze(0).expand(*tensordict.batch_size, -1)
        w = w_init
        print(f"[INIT] Using custom initial weights (shape: {w.shape}, non-zero: {torch.count_nonzero(w).item()})")

    # Optional: Normalize initial weights to target magnitude for component scale ~0.5
    normalize_weights = tensordict["params"].get("normalize_weights", False)
    # Handle both batched and non-batched normalize_weights
    if torch.is_tensor(normalize_weights):
        normalize_weights = normalize_weights.item() if normalize_weights.dim() == 0 else normalize_weights[0].item()
    if normalize_weights:
        # Calculate target magnitude: target_magnitude_factor * sqrt(N*(N-1)/2)
        N_val = int(self.N)
        num_weights = (N_val * (N_val - 1)) // 2

        # Get target_magnitude_factor from params
        target_mag_factor = tensordict["params"].get("target_magnitude_factor", 0.5)
        if torch.is_tensor(target_mag_factor):
            target_mag_factor = target_mag_factor.item() if target_mag_factor.dim() == 0 else target_mag_factor[0].item()

        target_magnitude = target_mag_factor * (num_weights ** 0.5)

        w_norm = torch.norm(w, dim=-1, keepdim=True)
        w = w / torch.clamp(w_norm, min=1e-8) * target_magnitude  # Scale to target magnitude
    
    # Compute initial S vector - Sfromw already handles tensor conversion internally
    Svector_current, min_idx = Sfromw(w, n, N)
    
    # Compute difference
    Svector_difference = Svector_target - Svector_current
    
    out = TensorDict(
        {
            "w": w,
            "Svector_current": Svector_current,
            "min_idx": min_idx,
            "Svector_difference": Svector_difference,
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
    )
    
    # print(f"[DEBUG-RESET] Final out['params', 'target'] shape: {out['params', 'target'].shape}")
    return out

#########################
# Setup _make_spec
#########################
def _make_spec(self, td_params):
    # Under the hood, this will populate self.output_spec["observation"]
    self.observation_spec = Composite(
        w=Unbounded(
            shape=((self.N*(self.N-1))//2,),
            dtype=torch.float32,
        ),
        Svector_current=Unbounded(
            shape=(2**self.n-1,),
            dtype=torch.float32,
        ),
        Svector_difference=Unbounded(
            shape=(2**self.n-1,),
            dtype=torch.float32,
        ),
        min_idx=Unbounded(
            shape=(2**self.n-1,),
            dtype=torch.int64,
        ),
        # we need to add the ``params`` to the observation specs, as we want
        # to pass it at each step during a rollout
        params=make_composite_from_td(td_params["params"]),
        shape=(),
    )
    # since the environment is stateless, we expect the previous output as input.
    # For this, ``EnvBase`` expects some state_spec to be available
    self.state_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec when
    # `self.action_spec = spec` will be called supported
    self.action_spec = Bounded(
        low=-1,
        high=1,
        shape=((self.N*(self.N-1))//2,),
        dtype=torch.float32,
    )
    self.reward_spec = Unbounded(shape=(*td_params.shape, 1))


def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = Composite(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else Unbounded(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

#########################
# Setup _set_seed
#########################
def _set_seed(self, seed: Optional[int]):
    # Create a generator on the correct device
    rng = torch.Generator(device=self.device)
    rng.manual_seed(seed)
    self.rng = rng
    
#########################
# Setup the gen_params
#########################
def gen_params(dt=0.01, target=None, n=3, N=5, batch_size=None, device="cpu", normalize_weights=False, target_magnitude_factor=0.5, precision="float32"):
        """
        Generate the parameters for the HEC problem.

        Args:
            target_magnitude_factor: Factor for weight normalization target magnitude (default: 0.5)
                                   target_magnitude = target_magnitude_factor * sqrt(num_weights)
            precision: "float32" or "float64" for tensor precision
        """
        # print("generating params...")
        # print(f"[DEBUG-GEN] Input target: {target}")
        # print(f"[DEBUG-GEN] Input target shape: {target.shape if torch.is_tensor(target) else 'not tensor'}")
        # print(f"[DEBUG-GEN] Input batch_size: {batch_size}")
        # print(f"[DEBUG-GEN] Input dt: {dt}, n: {n}, N: {N}")
        # print(f"[DEBUG-GEN] Input device: {device}")
        
        if batch_size is None:
            batch_size = []
        
        # Set dtype based on precision parameter
        dtype = torch.float64 if precision == "float64" else torch.float32
            
        # Create parameters for the TensorDict
        # Start with empty dictionaries to fill
        batched_params = {}
        
        # Handle all parameters with batching
        if batch_size:
            # Convert dt to batched tensor if it's a scalar
            if not torch.is_tensor(dt):
                # Add device=device
                batched_params["dt"] = torch.full(batch_size, dt, device=device)
            elif dt.dim() == 0:
                batched_params["dt"] = dt.expand(batch_size[0]).to(device)
            else:
                batched_params["dt"] = dt.to(device)
                
            # Convert n to batched tensor if it's a scalar
            if not torch.is_tensor(n):
                # Add device=device
                batched_params["n"] = torch.full(batch_size, n, dtype=torch.int64, device=device)
            elif n.dim() == 0:
                batched_params["n"] = n.expand(batch_size[0]).to(device)
            else:
                batched_params["n"] = n.to(device)
                
            # Convert N to batched tensor if it's a scalar
            if not torch.is_tensor(N):
                # Add device=device
                batched_params["N"] = torch.full(batch_size, N, dtype=torch.int64, device=device)
            elif N.dim() == 0:
                batched_params["N"] = N.expand(batch_size[0]).to(device)
            else:
                batched_params["N"] = N.to(device)
            
            # Add normalize_weights parameter (broadcast to batch size)
            if batch_size:
                batched_params["normalize_weights"] = torch.full(batch_size, normalize_weights, dtype=torch.bool, device=device)
            else:
                batched_params["normalize_weights"] = torch.tensor(normalize_weights, dtype=torch.bool, device=device)

            # Add target_magnitude_factor parameter (broadcast to batch size)
            if batch_size:
                batched_params["target_magnitude_factor"] = torch.full(batch_size, target_magnitude_factor, dtype=torch.float32, device=device)
            else:
                batched_params["target_magnitude_factor"] = torch.tensor(target_magnitude_factor, dtype=torch.float32, device=device)

            # Handle target
            if target is not None:
                # If target is not a tensor, convert it
                if not torch.is_tensor(target):
                    # Add device=device and dtype
                    batched_params["target"] = torch.tensor(target, device=device, dtype=dtype).expand(batch_size[0], -1)
                    # print(f"[DEBUG-GEN] Converted target to batched tensor: {batched_params['target'].shape}")
                # If target is a 1D tensor, expand it
                elif target.dim() == 1:
                    # Move target to device before expanding
                    batched_params["target"] = target.to(device).unsqueeze(0).expand(batch_size[0], -1)
                    # print(f"[DEBUG-GEN] Expanded 1D target: {batched_params['target'].shape}")
                # If target already has batch dimensions
                elif target.dim() > 1:
                    # Ensure target is on the correct device
                    target = target.to(device)
                    # Handle mismatched batch dimensions
                    if list(target.shape[:-1]) != batch_size:
                        # Reduce dimensions if needed
                        if target.dim() > 2:
                            target = target[0]
                            # print(f"[DEBUG-GEN] Reduced dimensions: {target.shape}")
                        # Expand to match batch size
                        batched_params["target"] = target.expand(batch_size[0], -1)
                        # print(f"[DEBUG-GEN] Expanded target: {batched_params['target'].shape}")
                    else:
                        batched_params["target"] = target
        else:
            # For non-batched case, ensure parameters are tensors on the correct device
            batched_params = {
                # Add device=device
                "dt": torch.tensor(dt, device=device) if not torch.is_tensor(dt) else dt.to(device),
                "n": torch.tensor(n, dtype=torch.int64, device=device) if not torch.is_tensor(n) else n.to(device),
                "N": torch.tensor(N, dtype=torch.int64, device=device) if not torch.is_tensor(N) else N.to(device),
                "normalize_weights": torch.tensor(normalize_weights, dtype=torch.bool, device=device),
                "target_magnitude_factor": torch.tensor(target_magnitude_factor, dtype=torch.float32, device=device)
            }
            if target is not None:
                # Add device=device and dtype
                batched_params["target"] = torch.tensor(target, device=device, dtype=dtype) if not torch.is_tensor(target) else target.to(device=device, dtype=dtype)
        
        # Create the TensorDict with correct shape
        params_shape = batch_size if batch_size else []
        td = TensorDict(
            {
                # Ensure inner TensorDict tensors are on the correct device
                "params": TensorDict(batched_params, params_shape, device=device)
            },
            params_shape,
            device=device
        )
        
        # print(f"[DEBUG-GEN] Final TensorDict shape: {td.shape}")
        # print(f"[DEBUG-GEN] Final params dt shape: {td['params', 'dt'].shape}")
        # print(f"[DEBUG-GEN] Final params n shape: {td['params', 'n'].shape}")
        # print(f"[DEBUG-GEN] Final params N shape: {td['params', 'N'].shape}")
        # if 'target' in td['params']:
        #     print(f"[DEBUG-GEN] Final params target shape: {td['params', 'target'].shape}")
        
        return td
    

class HECenv(EnvBase):
    """
    A simple environment for the HECCone problem.
    """
    metadata = {}
    batch_locked = False
    def __init__(self, dt, target=None, n=3, N=5, seed=None, device="cpu", precision="float32", initial_weights=None):
        self.n = n
        self.N = N
        self.dt = dt
        self.precision = precision
        
        # Set dtype based on precision
        dtype = torch.float64 if precision == "float64" else torch.float32
        
        # Handle the target
        if target is None:
            # Default target with ones
            self.target = torch.ones(2**n-1, device=device, dtype=dtype)
        elif isinstance(target, (int, float)):
            # Scalar value
            self.target = torch.ones(2**n-1, device=device, dtype=dtype) * target
        elif torch.is_tensor(target) and target.dim() == 1:
            # Already a 1D tensor - convert to correct dtype
            self.target = target.to(device=device, dtype=dtype)
        else:
            # Convert to tensor if not already
            self.target = torch.tensor(target, device=device, dtype=dtype)

        self.device = device
        self.initial_weights = initial_weights  # Store initial weights if provided
        # Pass device and precision to gen_params
        td_params = self.gen_params(dt, self.target, n, N, device=device, precision=self.precision)

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
            print(f"No seed provided, using {seed}")
        self.set_seed(seed)
        
    
    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed
    
#########################
# Regarding the local search
#########################
def get_Cuts(min_idx, n, N):
    """
    min_idx: the idicies describing the W configuration
    Ws_from_index: the function that converts the min_idx to the W ( including I )
    Cut: index of w that cut the W and W^c
    """
    Cuts = []
    for W in Ws_from_index(min_idx, n, N):
        Cut = []
        outside = [i-1 for i in range(1, N+1) if i not in W]
        inside = [i-1 for i in range(1, N+1) if i in W]
        for o in outside:
            for i in inside:
                Cut.append(combination_to_index([o, i], N))
        Cuts.append(Cut)
    return Cuts



def scipy_search( w_reference, target, min_idx, n, N, epsilon, factor):
    """
    w_reference: the reference weights
    target: the target S-vector
    min_idx: wanted min_idx
    n: the number of variables
    N: the number of elements in the W configuration
    """
    
    Cuts = get_Cuts(min_idx, n, N)

    # Create coefficient matrix A
    n_vars = len(w_reference)  # 10 variables
    A = np.zeros((len(Cuts), n_vars))
    for i, cut in enumerate(Cuts):
        for j in cut:
            A[i, j] += 1

    # Define optimization objective: minimize squared deviation from target equations
    def objective(w):
        Aw = A @ w
        return -np.dot(Aw, target) / (np.linalg.norm(Aw) * np.linalg.norm(target))
    
    # Optimized objective with analytical gradient for 10x speedup
    target_norm = np.linalg.norm(target)
    
    def objective_and_gradient(w):
        Aw = A @ w
        Aw_norm = np.linalg.norm(Aw)
        
        if Aw_norm == 0:
            return 0.0, np.zeros_like(w)
        
        dot_product = np.dot(Aw, target)
        obj = -dot_product / (Aw_norm * target_norm)
        
        # Analytical gradient
        grad_term1 = -A.T @ target / (Aw_norm * target_norm)
        grad_term2 = dot_product * A.T @ Aw / (Aw_norm**3 * target_norm)
        grad = grad_term1 + grad_term2
        
        return obj, grad

    # Initial guess (start with reference solution)
    initial_guess = w_reference.copy()

    # Constraints: each element must be within 0.01 of reference value
    explore = True
    
    while explore:
        print(f"[scipy_search] Exploring with epsilon: {epsilon}")
        bounds = [(max(0, ref-epsilon), ref+epsilon) for ref in w_reference]

        # Solve the optimization problem using scipy with analytical gradient
        result = minimize(
            objective_and_gradient,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            jac=True,  # Function returns both objective and gradient
            options={'ftol': 1e-10, 'gtol': 1e-10}
        )

        # Get the optimized weights
        w_optimal = result.x    

        # Check if we found a solution that satisfies equations
        reward = np.dot(A @ w_optimal, target) / (np.linalg.norm(A @ w_optimal) * np.linalg.norm(target))
        is_solution = np.allclose(reward, 1, atol=1e-5)

        print("Optimization successful:", result.success)
        # print("\nReference weights:", [f"{w:.4f}" for w in w_reference])
        # print("\nOptimal weights:", [f"{w:.4f}" for w in w_optimal])
        
        real_value, real_idx = Sfromw_single(w_optimal, n, N)
        # print(f"\nreal values: {[f'{v:.4f}' for v in real_value.tolist()]}")
        real_idx = real_idx.cpu().tolist() if torch.is_tensor(real_idx) else [i.item() if torch.is_tensor(i) else i for i in real_idx]
        wanted_idx = min_idx
        # print(f"\nreal idx: {real_idx}")
        # print(f"\nwanted idx: {wanted_idx}")
        
        if np.array_equal(real_idx, wanted_idx):
            print(f"explored within the same w configuration.")
            explore = False
        
            # print(f"\nachieved values: {[f'{v:.4f}' for v in A @ w_optimal]}")
            # print(f"\ntarget: {[f'{v:.4f}' for v in target]}")
            print(f"\nreward: {reward:.6f}")
            print("Found solution satisfies equations:", is_solution)
        else:
            print(f"explored out of the same w configuration with epsilon: {epsilon}")
            epsilon *= factor
            
            # Add lower bound for epsilon - if too small, use original w as optimized value
            if epsilon < 1e-7:
                print(f"epsilon reached lower bound (1e-7), using original w_reference as w_optimal")
                w_optimal = w_reference.copy()
                reward = np.dot(A @ w_optimal, target) / (np.linalg.norm(A @ w_optimal) * np.linalg.norm(target))
                explore = False
            
        print("")
        
        # Store results in JSON file
    results = {
        "target": target.tolist(),
        "Cuts": Cuts,
        "w_optimal": w_optimal.tolist(),
        "reward": reward,
        "is_solution": is_solution,
    }
    
    return results


def gradient_search_worker(args):
    """
    Worker function for parallel gradient computation.
    """
    i, w_reference, S_target, dS, min_idx, n, N, epsilon, factor, base_reward = args
    dS_vector = np.zeros(len(S_target))
    dS_vector[i] = dS
    S_target_perturbed = S_target + dS_vector
    results = scipy_search(w_reference, S_target_perturbed, min_idx, n, N, epsilon, factor)
    return (results["reward"] - base_reward) / dS


def gradient_search(S_target, w_reference, min_idx, n, N, dS=0.001, epsilon=0.1, factor=0.9):
    """
    S_target: the target S-vector
    w_reference: the reference weights
    n: the number of variables
    N: the number of elements in the W configuration
    
    Now with PARALLEL computation of gradients!
    """
    Cuts = get_Cuts(min_idx, n, N)
    base_results = scipy_search(w_reference, S_target, min_idx, n, N, epsilon, factor)
    base_reward = base_results["reward"]
    
    # Determine number of CPUs to use
    available_cpus = get_available_cpus(0.9)
    num_cpus_to_use = min(21, available_cpus)
    
    # Prepare tasks for parallel processing
    n_dims = len(S_target)
    gradient_tasks = []
    for i in range(n_dims):
        gradient_tasks.append((i, w_reference, S_target, dS, min_idx, n, N, epsilon, factor, base_reward))
    
    print(f"Computing gradient in parallel: {n_dims} dimensions using {num_cpus_to_use} CPUs...")
    
    # Process in batches if needed
    S_grad = []
    batch_size = num_cpus_to_use
    
    for batch_start in range(0, n_dims, batch_size):
        batch_end = min(batch_start + batch_size, n_dims)
        batch_tasks = gradient_tasks[batch_start:batch_end]
        
        with multiprocessing.Pool(processes=min(num_cpus_to_use, len(batch_tasks))) as pool:
            batch_results = pool.map(gradient_search_worker, batch_tasks)
        
        S_grad.extend(batch_results)
        
        if batch_end < n_dims:
            print(f"  Completed batch {batch_start//batch_size + 1}/{(n_dims + batch_size - 1)//batch_size}")
    
    return S_grad


def gradient_search_rl_worker(args):
    """
    Worker function for parallel RL-based gradient computation.
    Each worker runs a mini RL training on a perturbed target S+dS[i].
    """
    i, S_target, dS, model_path, base_args = args

    dS_vector = np.zeros(len(S_target))
    dS_vector[i] = dS
    S_target_perturbed = S_target + dS_vector

    modified_args = base_args.copy()
    modified_args["target"] = torch.tensor(S_target_perturbed, dtype=modified_args.get("target").dtype)
    modified_args["load_model_path"] = model_path
    modified_args["load_optimizer_state"] = False
    modified_args["stage"] = -1
    modified_args["dir"] = os.path.join("/tmp/claude", f"gradient_rl_dim_{i}")
    os.makedirs(modified_args["dir"], exist_ok=True)
    modified_args["name"] = f"gradient_dim_{i}"
    modified_args["disable_plotting"] = True
    modified_args.pop("save_model_path", None)
    modified_args.pop("save_best_model_path", None)

    print(f"[gradient_search_rl] Computing gradient for dimension {i}/{len(S_target)}")

    try:
        result = run_stage(modified_args)
        reward = result["best_optimized_reward"]
        print(f"[gradient_search_rl] Dimension {i}: reward = {reward:.6f}")
        return reward
    except Exception as e:
        print(f"[gradient_search_rl] Error in dimension {i}: {e}")
        return 0.0


def gradient_search_rl(S_target, best_model_state, base_reward, n, N, dS, base_args):
    """
    RL-based gradient computation using warm-started mini RL trainings.

    Args:
        S_target: target S-vector
        best_model_state: dict with policy weights from main training
        base_reward: reward from main training
        n, N: problem parameters
        dS: perturbation size
        base_args: full args dict from main training

    Returns:
        S_grad: gradient array computed via RL exploration
    """
    print("\n[gradient_search_rl] Starting RL-based gradient computation...")
    print(f"[gradient_search_rl] Base reward: {base_reward:.6f}")
    print(f"[gradient_search_rl] Perturbation dS: {dS}")

    temp_model_file = tempfile.mktemp(suffix=".pth", dir="/tmp/claude")
    os.makedirs("/tmp/claude", exist_ok=True)
    torch.save(best_model_state, temp_model_file)
    print(f"[gradient_search_rl] Saved best model to: {temp_model_file}")

    n_dims = len(S_target)
    print(f"[gradient_search_rl] Computing {n_dims} dimensions using sequential GPU computation...")

    S_grad = []

    for i in range(n_dims):
        dS_vector = np.zeros(len(S_target))
        dS_vector[i] = dS
        S_target_perturbed = S_target + dS_vector

        modified_args = base_args.copy()
        modified_args["target"] = torch.tensor(S_target_perturbed, dtype=modified_args.get("target").dtype)
        modified_args["load_model_path"] = temp_model_file
        modified_args["load_optimizer_state"] = False
        modified_args["stage"] = -1
        modified_args["dir"] = os.path.join("/tmp/claude", f"gradient_rl_dim_{i}")
        os.makedirs(modified_args["dir"], exist_ok=True)
        modified_args["name"] = f"gradient_dim_{i}"
        modified_args["disable_plotting"] = True
        modified_args.pop("save_model_path", None)
        modified_args.pop("save_best_model_path", None)
        modified_args["skip_gradient"] = True

        print(f"[gradient_search_rl] Computing gradient for dimension {i+1}/{n_dims}")

        try:
            result = run_stage(modified_args)
            reward = result["best_optimized_reward"]
            S_grad.append((reward - base_reward) / dS)
            print(f"[gradient_search_rl] Dimension {i+1}/{n_dims}: reward = {reward:.6f}")
        except Exception as e:
            print(f"[gradient_search_rl] Error in dimension {i}: {e}")
            S_grad.append(0.0)

    try:
        os.remove(temp_model_file)
        print(f"[gradient_search_rl] Cleaned up temp model file")
    except Exception as e:
        print(f"[gradient_search_rl] Warning: Could not remove temp file: {e}")

    print(f"[gradient_search_rl] Gradient computation complete!")
    print(f"[gradient_search_rl] Gradient magnitude: {np.linalg.norm(S_grad):.6f}")

    return S_grad


def gradient_search_rl_v2_worker(args):
    """
    Worker function for parallel RL-based gradient computation (v2 with random sampling).
    Each worker runs a mini RL training on a perturbed target S + dS * direction.
    """
    sample_idx, S_target, direction_vector, dS, model_path, base_args = args

    S_target_perturbed = S_target + dS * direction_vector

    modified_args = base_args.copy()
    modified_args["target"] = torch.tensor(S_target_perturbed, dtype=modified_args.get("target").dtype)
    modified_args["load_model_path"] = model_path
    modified_args["load_optimizer_state"] = False
    modified_args["stage"] = -1
    modified_args["dir"] = os.path.join("/tmp/claude", f"gradient_rl_v2_sample_{sample_idx}")
    os.makedirs(modified_args["dir"], exist_ok=True)
    modified_args["name"] = f"gradient_v2_sample_{sample_idx}"
    modified_args["disable_plotting"] = True
    modified_args.pop("save_model_path", None)
    modified_args.pop("save_best_model_path", None)

    print(f"[gradient_search_rl_v2] Computing sample {sample_idx}")

    try:
        result = run_stage(modified_args)
        reward = result["best_optimized_reward"]
        print(f"[gradient_search_rl_v2] Sample {sample_idx}: reward = {reward:.6f}")
        return reward
    except Exception as e:
        print(f"[gradient_search_rl_v2] Error in sample {sample_idx}: {e}")
        return 0.0


def _gpu_worker_single_sample(args):
    """Worker function for multi-GPU parallel gradient sampling.

    Each worker runs on a specific GPU and computes reward for one sample.
    """
    (sample_idx, gpu_id, S_target_perturbed, modified_args_template,
     cold_start, temp_model_file, n_repeats, base_seed, base_reward) = args

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Re-import torch after setting CUDA_VISIBLE_DEVICES
    import torch

    modified_args = modified_args_template.copy()
    modified_args["target"] = torch.tensor(S_target_perturbed, dtype=torch.float64)
    modified_args["load_model_path"] = None if cold_start else temp_model_file
    modified_args["load_optimizer_state"] = False
    modified_args["stage"] = -1
    modified_args["dir"] = os.path.join("/tmp/claude", f"gradient_rl_v2_sample_{sample_idx}_gpu{gpu_id}")
    os.makedirs(modified_args["dir"], exist_ok=True)
    modified_args["name"] = f"gradient_v2_sample_{sample_idx}"
    modified_args["disable_plotting"] = True
    modified_args.pop("save_model_path", None)
    modified_args.pop("save_best_model_path", None)
    modified_args["skip_gradient"] = True

    try:
        if n_repeats == 1:
            result = run_stage(modified_args)
            reward = result["best_optimized_reward"]
            print(f"[GPU {gpu_id}] Sample {sample_idx}: reward = {reward:.6f}")
        else:
            repeat_rewards = []
            for rep in range(n_repeats):
                modified_args["dir"] = os.path.join("/tmp/claude", f"gradient_rl_v2_sample_{sample_idx}_rep_{rep}_gpu{gpu_id}")
                os.makedirs(modified_args["dir"], exist_ok=True)
                modified_args["name"] = f"gradient_v2_sample_{sample_idx}_rep_{rep}"
                modified_args["seed"] = base_seed + rep * 1000 + sample_idx
                result = run_stage(modified_args)
                rep_reward = result["best_optimized_reward"]
                repeat_rewards.append(rep_reward)
                print(f"[GPU {gpu_id}] Sample {sample_idx}, repeat {rep+1}/{n_repeats}: reward = {rep_reward:.6f}")
            reward = max(repeat_rewards)
            print(f"[GPU {gpu_id}] Sample {sample_idx}: MAX reward = {reward:.6f}")
        return (sample_idx, reward)
    except Exception as e:
        print(f"[GPU {gpu_id}] Error in sample {sample_idx}: {e}")
        return (sample_idx, base_reward)


def gradient_search_rl_v2(S_target, best_model_state, base_reward, n, N, dS, n_samples, base_args, cold_start=False, distance_mode='fixed', n_repeats=1, n_gpus=1):
    """
    RL-based gradient computation using random sampling on hyperplane orthogonal to S.
    Uses linear regression for robust gradient estimation.

    Args:
        S_target: target S-vector (numpy array)
        best_model_state: dict with policy weights from main training
        base_reward: reward from main training
        n, N: problem parameters
        dS: perturbation size (max distance for uniform modes)
        n_samples: number of random directions to sample
        base_args: full args dict from main training
        cold_start: if True, start each sample from scratch (unbiased but slower)
                   if False, warm start from trained model (faster but potentially biased)
        distance_mode: 'fixed' (default), 'uniform_full' [0, dS], or 'uniform_half' [dS/2, dS]
        n_repeats: number of times to run RL for each sample, take max reward (default 1)
        n_gpus: number of GPUs to use for parallel sampling (default 1 = sequential)

    Returns:
        S_grad: gradient array computed via linear regression on RL samples
    """
    print("\n[gradient_search_rl_v2] Starting RL-based gradient computation (v2 with random sampling)...")
    print(f"[gradient_search_rl_v2] Base reward: {base_reward:.6f}")
    print(f"[gradient_search_rl_v2] Perturbation dS: {dS}")
    print(f"[gradient_search_rl_v2] Distance mode: {distance_mode}")
    print(f"[gradient_search_rl_v2] Number of samples: {n_samples}")
    print(f"[gradient_search_rl_v2] Repeats per sample: {n_repeats} ({'max of repeats' if n_repeats > 1 else 'single run'})")
    print(f"[gradient_search_rl_v2] GPUs: {n_gpus} ({'parallel' if n_gpus > 1 else 'sequential'})")
    print(f"[gradient_search_rl_v2] Cold start: {cold_start} ({'unbiased, slower' if cold_start else 'warm start, faster'})")

    temp_model_file = tempfile.mktemp(suffix=".pth", dir="/tmp/claude")
    os.makedirs("/tmp/claude", exist_ok=True)
    torch.save(best_model_state, temp_model_file)
    print(f"[gradient_search_rl_v2] Saved best model to: {temp_model_file}")

    n_dims = len(S_target)
    print(f"[gradient_search_rl_v2] S-vector dimension: {n_dims}")

    S_unit = S_target / np.linalg.norm(S_target)
    print(f"[gradient_search_rl_v2] Computing orthonormal basis for hyperplane orthogonal to S...")

    random_matrix = np.random.randn(n_dims, n_dims)
    random_matrix[:, 0] = S_unit
    Q, R = np.linalg.qr(random_matrix)
    orthogonal_basis = Q[:, 1:]
    print(f"[gradient_search_rl_v2] Orthogonal basis shape: {orthogonal_basis.shape} (should be {n_dims} x {n_dims-1})")

    print(f"[gradient_search_rl_v2] Sampling {n_samples} random directions on hyperplane...")
    directions_in_basis = []
    directions_in_full_space = []
    distances = []  # Store actual distance for each sample

    for i in range(n_samples):
        valid_direction = False
        resample_count = 0

        while not valid_direction:
            coeffs = np.random.randn(n_dims - 1)
            direction = orthogonal_basis @ coeffs
            direction = direction / np.linalg.norm(direction)

            dot_with_S = np.dot(direction, S_unit)
            if abs(dot_with_S) > 1e-10:
                print(f"[gradient_search_rl_v2] WARNING: Sample {i} not orthogonal to S (dot = {dot_with_S})")

            # Sample distance based on mode
            if distance_mode == 'fixed':
                distance = dS
            elif distance_mode == 'uniform_full':
                distance = np.random.uniform(0, dS)
            elif distance_mode == 'uniform_half':
                distance = np.random.uniform(dS/2, dS)
            else:
                raise ValueError(f"Unknown distance_mode: {distance_mode}")

            S_target_perturbed = S_target + distance * direction

            if np.all(S_target_perturbed >= 0):
                valid_direction = True
            else:
                resample_count += 1
                if resample_count % 10 == 0:
                    print(f"[gradient_search_rl_v2] Sample {i}: resampled {resample_count} times (negative components)")

        if resample_count > 0:
            print(f"[gradient_search_rl_v2] Sample {i}: accepted after {resample_count} resamples")

        coeffs_for_normalized_direction = orthogonal_basis.T @ direction
        directions_in_basis.append(coeffs_for_normalized_direction)
        directions_in_full_space.append(direction)
        distances.append(distance)

    distances = np.array(distances)
    if distance_mode != 'fixed':
        print(f"[gradient_search_rl_v2] Distance statistics: mean={distances.mean():.6f}, std={distances.std():.6f}, min={distances.min():.6f}, max={distances.max():.6f}")

    base_seed = base_args.get("seed", 42) or 42

    if n_gpus > 1:
        # Multi-GPU parallel execution
        print(f"[gradient_search_rl_v2] Using MULTI-GPU parallel computation with {n_gpus} GPUs...")
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)

        # Prepare args template (without tensor to avoid pickling issues)
        args_template = base_args.copy()
        args_template.pop("target", None)  # Will be set per-sample

        # Prepare all sample tasks
        all_tasks = []
        for i in range(n_samples):
            S_target_perturbed = S_target + distances[i] * directions_in_full_space[i]
            gpu_id = i % n_gpus  # Round-robin GPU assignment
            task = (i, gpu_id, S_target_perturbed.tolist(), args_template,
                    cold_start, temp_model_file, n_repeats, base_seed, base_reward)
            all_tasks.append(task)

        # Process in batches of n_gpus
        rewards = [None] * n_samples
        for batch_start in range(0, n_samples, n_gpus):
            batch_end = min(batch_start + n_gpus, n_samples)
            batch_tasks = all_tasks[batch_start:batch_end]
            print(f"[gradient_search_rl_v2] Processing batch {batch_start//n_gpus + 1}: samples {batch_start+1}-{batch_end}")

            with mp.Pool(processes=len(batch_tasks)) as pool:
                batch_results = pool.map(_gpu_worker_single_sample, batch_tasks)

            for sample_idx, reward in batch_results:
                rewards[sample_idx] = reward
                print(f"[gradient_search_rl_v2] Sample {sample_idx+1}/{n_samples}: reward = {reward:.6f}")

    else:
        # Sequential single-GPU execution (original behavior)
        print(f"[gradient_search_rl_v2] Using sequential GPU computation (no multiprocessing)...")

        rewards = []
        for i in range(n_samples):
            S_target_perturbed = S_target + distances[i] * directions_in_full_space[i]

            modified_args = base_args.copy()
            modified_args["target"] = torch.tensor(S_target_perturbed, dtype=modified_args.get("target").dtype)
            # Cold start: train from scratch (unbiased); Warm start: use trained model (faster but biased)
            modified_args["load_model_path"] = None if cold_start else temp_model_file
            modified_args["load_optimizer_state"] = False
            modified_args["stage"] = -1
            modified_args["dir"] = os.path.join("/tmp/claude", f"gradient_rl_v2_sample_{i}")
            os.makedirs(modified_args["dir"], exist_ok=True)
            modified_args["name"] = f"gradient_v2_sample_{i}"
            modified_args["disable_plotting"] = True
            modified_args.pop("save_model_path", None)
            modified_args.pop("save_best_model_path", None)
            modified_args["skip_gradient"] = True

            print(f"[gradient_search_rl_v2] Computing sample {i+1}/{n_samples}...")

            try:
                if n_repeats == 1:
                    # Single run (default behavior)
                    result = run_stage(modified_args)
                    reward = result["best_optimized_reward"]
                    print(f"[gradient_search_rl_v2] Sample {i+1}/{n_samples}: reward = {reward:.6f}")
                else:
                    # Multiple repeats - run n_repeats times and take max
                    repeat_rewards = []
                    for rep in range(n_repeats):
                        modified_args["dir"] = os.path.join("/tmp/claude", f"gradient_rl_v2_sample_{i}_rep_{rep}")
                        os.makedirs(modified_args["dir"], exist_ok=True)
                        modified_args["name"] = f"gradient_v2_sample_{i}_rep_{rep}"
                        # Use different seed for each repeat to ensure independent runs
                        modified_args["seed"] = base_seed + rep * 1000 + i
                        result = run_stage(modified_args)
                        rep_reward = result["best_optimized_reward"]
                        repeat_rewards.append(rep_reward)
                        print(f"[gradient_search_rl_v2] Sample {i+1}/{n_samples}, repeat {rep+1}/{n_repeats} (seed={modified_args['seed']}): reward = {rep_reward:.6f}")
                    reward = max(repeat_rewards)
                    print(f"[gradient_search_rl_v2] Sample {i+1}/{n_samples}: MAX reward = {reward:.6f} (from {repeat_rewards})")
                rewards.append(reward)
            except Exception as e:
                print(f"[gradient_search_rl_v2] Error in sample {i}: {e}")
                rewards.append(base_reward)

    try:
        os.remove(temp_model_file)
        print(f"[gradient_search_rl_v2] Cleaned up temp model file")
    except Exception as e:
        print(f"[gradient_search_rl_v2] Warning: Could not remove temp file: {e}")

    sampling_data = {
        "base_S": S_target.tolist(),
        "base_reward": float(base_reward),
        "dS": float(dS),
        "distance_mode": distance_mode,
        "n_samples": n_samples,
        "n_repeats": n_repeats,
        "samples": []
    }

    for i in range(n_samples):
        S_perturbed = S_target + distances[i] * directions_in_full_space[i]
        sampling_data["samples"].append({
            "sample_index": i,
            "S_perturbed": S_perturbed.tolist(),
            "direction": directions_in_full_space[i].tolist(),
            "distance": float(distances[i]),
            "reward": float(rewards[i]),
            "reward_change": float(rewards[i] - base_reward)
        })

    output_dir = base_args.get("dir", ".")
    stage_num = base_args.get("stage", 0)
    sampling_file = os.path.join(output_dir, f"gradient_samples_stage_{stage_num}.json")

    with open(sampling_file, 'w') as f:
        json.dump(sampling_data, f, indent=2)
    print(f"[gradient_search_rl_v2] Saved sampling data to: {sampling_file}")

    print(f"\n[gradient_search_rl_v2] Performing linear regression...")
    reward_changes = np.array([r - base_reward for r in rewards])

    # For variable distance modes, use: reward_change = distance * (gradient · direction)
    # So we fit: y = X @ gradient where X_i = distance_i * direction_i, y_i = reward_change_i
    # For fixed mode (backward compatible): scale by 1/dS
    if distance_mode == 'fixed':
        A = np.array(directions_in_basis)
        b = reward_changes / dS
    else:
        # Variable distance: include distance in design matrix
        A = np.array(directions_in_basis) * distances[:, np.newaxis]
        b = reward_changes

    print(f"[gradient_search_rl_v2] Design matrix A shape: {A.shape}")
    print(f"[gradient_search_rl_v2] Response vector b shape: {b.shape}")
    print(f"[gradient_search_rl_v2] Reward change statistics: mean={np.mean(reward_changes):.6f}, std={np.std(reward_changes):.6f}")

    A_with_intercept = np.column_stack([A, np.ones(len(A))])
    params, residuals, rank, s = np.linalg.lstsq(A_with_intercept, b, rcond=None)
    gradient_coeffs = params[:-1]
    intercept = params[-1]

    ss_res = residuals[0] if len(residuals) > 0 else np.sum((b - A_with_intercept @ params)**2)
    ss_tot = np.sum((b - np.mean(b))**2)
    r2_with_intercept = 1 - (ss_res / ss_tot)

    b_pred_no_intercept = A @ gradient_coeffs
    ss_res_no_intercept = np.sum((b - b_pred_no_intercept)**2)
    r2_no_intercept = 1 - (ss_res_no_intercept / ss_tot)

    print(f"[gradient_search_rl_v2] Linear regression complete!")
    print(f"[gradient_search_rl_v2] Matrix rank: {rank}/{n_dims}")
    print(f"[gradient_search_rl_v2] Intercept: {intercept:.6e}")
    print(f"[gradient_search_rl_v2] R² (with intercept): {r2_with_intercept:.6f}")
    print(f"[gradient_search_rl_v2] R² (no intercept): {r2_no_intercept:.6f}")
    if len(residuals) > 0:
        print(f"[gradient_search_rl_v2] Residual: {residuals[0]:.6f}")

    gradient_in_full_space = orthogonal_basis @ gradient_coeffs

    gradient_dot_S = np.dot(gradient_in_full_space, S_unit)
    print(f"[gradient_search_rl_v2] Gradient · S = {gradient_dot_S:.10f} (should be ~0)")
    print(f"[gradient_search_rl_v2] Gradient magnitude: {np.linalg.norm(gradient_in_full_space):.6f}")

    sampling_data["gradient_computed"] = gradient_in_full_space.tolist()
    sampling_data["gradient_magnitude"] = float(np.linalg.norm(gradient_in_full_space))
    sampling_data["gradient_dot_S"] = float(gradient_dot_S)
    sampling_data["linear_regression"] = {
        "matrix_rank": int(rank),
        "residuals": float(residuals[0]) if len(residuals) > 0 else None,
        "intercept": float(intercept),
        "r2_with_intercept": float(r2_with_intercept),
        "r2_no_intercept": float(r2_no_intercept)
    }

    with open(sampling_file, 'w') as f:
        json.dump(sampling_data, f, indent=2)
    print(f"[gradient_search_rl_v2] Updated sampling data with gradient: {sampling_file}")

    return gradient_in_full_space.tolist()


# #########################
# # Test the environment
# #########################
# print("="*50)
# print("Test the environment")
# target = torch.tensor([1.0]*7)
# print(f"[DEBUG-TEST] Original target: {target.shape}")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"[DEBUG-TEST] Device: {device}")
# env = HECenv(dt=0.01, target=target, n=3, N=5, device=device)
# print(f"[DEBUG-TEST] After creating env, env.target shape: {env.target.shape}")


# #########################
# # Wrap the environment
# #########################
# # env = TransformedEnv(
# #     env,
# #     # Unsqueeze the observations that we will concatenate
# #     UnsqueezeTransform(
# #         dim=-1,
# #         in_keys=["w", "Svector_current", "Svector_difference"],
# #         in_keys_inv=["w", "Svector_current", "Svector_difference"],
# #     ),
# # )
# env = TransformedEnv(env)

# # Now add a CatTensors transform to combine your vectors into a single observation:
# cat_transform = CatTensors(
#     in_keys=["w", "Svector_current", "Svector_difference"],
#     out_key="observation",
#     del_keys=False
# )
# env.append_transform(cat_transform)


# #########################
# # Check the environment
# #########################
# print("="*50)
# print("Check the environment specs")
# check_env_specs(env)
# print("="*50)
# print("="*50)
# print("="*50)


# #########################
# # Test the rollout
# #########################
# print("reset (batch size of 3)")
# batch_size = 5 # number of environments to be executed in batch
# print(f"[DEBUG-TEST] Creating params with batch_size={batch_size}")
# # Pass device to gen_params
# params = env.gen_params(dt=0.01, target=env.target, n=3, N=5, batch_size=[batch_size], device=device)
# print(f"[DEBUG-TEST] After gen_params, params['params', 'target'] shape: {params['params', 'target'].shape}")
# # Pass device to gen_params when resetting within rollout
# td = env.reset(env.gen_params(dt=0.01, target=env.target, n=3, N=5, batch_size=[batch_size], device=device))
# print(f"[DEBUG-TEST] After reset, td['params', 'target'] shape: {td['params', 'target'].shape}")
# # print("reset (batch size of 3)", td)
# td = env.rand_step(td)
# # print("rand step (batch size of 3)", td)

# rollout = env.rollout(
#     3,
#     auto_reset=False,  # we're executing the reset out of the ``rollout`` call
#     # Pass device to gen_params
#     tensordict=env.reset(env.gen_params(dt=0.01, target=target, n=3, N=5, batch_size=[batch_size], device=device)),
# )
# print("="*50)
# print("Test the rollout finished")
# print("="*50)
# print("rollout of len 3 (batch size of 5):", rollout)

def run_stage(args):
    """
    Run a single stage of the experiment.
    Returns the results dictionary for this stage.
    
    Args:
        stage (int): The stage number to run
        dir (str): Directory path for saving plots and results
    """
    # GPU memory management for multiple processes
    if args["device"] == "cuda":
        torch.cuda.empty_cache()
        # Allow more memory per process for better GPU utilization
        gpu_memory_fraction = args.get("gpu_memory_fraction", 0.95)  # Default to 0.95 if not specified
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
    
    #########################
    # Setting up the experiment
    #########################
    stage = args["stage"]
    dir = args["dir"]
    batch_size = args["batch_size"]
    max_training_cycles = args["max_training_cycles"]
    x_value = args["x_value"]
    target = args["target"]
    device = args["device"]
    dt = args["dt"]
    nnn = args["nnn"]
    NN = args["NN"]
    lr = args["lr"]
    epsilon = args["epsilon"]
    factor = args["factor"]
    keep_best_n = args["keep_best_n"]
    dS = args["dS"]
    name = args["name"]
    
    
    #########################
    # Setting up the environment
    #########################
    # Get precision parameter if provided (default to float32)
    precision = args.get("precision", "float32")
    # Get seed if provided
    seed = args.get("seed", None)
    # Get initial_weights if provided
    initial_weights = args.get("initial_weights", None)
    env = HECenv(dt=dt, target=target, n=nnn, N=NN, device=device, precision=precision, seed=seed, initial_weights=initial_weights)
    env = TransformedEnv(env)
    
    # Create a new CatTensors transform for this stage
    cat_transform = CatTensors(
        in_keys=["w", "Svector_current", "Svector_difference"],
        out_key="observation",
        del_keys=False
    )
    env.append_transform(cat_transform)
    
    #########################
    # Define policy
    #########################
    # Use seed from args if provided, otherwise use 0 for backward compatibility
    if seed is not None:
        torch.manual_seed(seed)
        env.set_seed(seed)
        print(f"Using seed {seed} for network initialization")
    else:
        torch.manual_seed(0)
        env.set_seed(0)
        print(f"No seed provided, using default seed 0")

    # Calculate correct action dimension for this problem size
    action_dim = (NN * (NN - 1)) // 2
    
    # Get network architecture hyperparameters
    hidden_dim = args.get("hidden_dim", 64)  # Default: 64
    num_layers = args.get("num_layers", 3)   # Default: 3 hidden layers
    
    # Build network dynamically based on hyperparameters
    layers = []
    layers.append(nn.LazyLinear(hidden_dim))
    layers.append(nn.Tanh())
    
    # Add hidden layers
    for _ in range(num_layers - 1):
        layers.append(nn.LazyLinear(hidden_dim))
        layers.append(nn.Tanh())
    
    # Output layer
    layers.append(nn.LazyLinear(action_dim))  # FIXED: Output correct number of action components
    
    net = nn.Sequential(*layers)
    
    # Move network to the correct device and set dtype based on precision
    net.to(device)
    if precision == "float64":
        net = net.double()  # Convert network to float64

    net.train()
    policy = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["action"],
    )

    #########################
    # Model loading (warm-start)
    #########################
    load_model_path = args.get("load_model_path", None)
    if load_model_path:
        if not os.path.exists(load_model_path):
            raise FileNotFoundError(f"Required model file not found: {load_model_path}")
        
        print(f"Loading model from: {load_model_path}")
        try:
            checkpoint = torch.load(load_model_path, map_location=device)
            policy.load_state_dict(checkpoint['policy_state_dict'])
            print(f"Successfully loaded model from {load_model_path}")
            
            # Print model info for verification
            if 'best_reward' in checkpoint:
                print(f"  - Previous best reward: {checkpoint['best_reward']:.6f}")
            if 'iteration' in checkpoint:
                print(f"  - Found at iteration: {checkpoint['iteration']}")
            if 'stage' in checkpoint:
                print(f"  - From stage: {checkpoint['stage']}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load required model from {load_model_path}: {e}")

    #########################
    # Train the policy and plot the results
    #########################
    # Add early stopping parameters
    early_stop_patience = args.get("early_stop_patience", 10)  # Default: 10 training cycles
    enable_early_stopping = args.get("enable_early_stopping", True)  # Default: enabled
    
    max_iter = max_training_cycles
    pbar = tqdm.tqdm(range(max_iter))
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_training_cycles)
    
    # Load optimizer state if explicitly requested (for true resuming)
    load_optimizer_state = args.get("load_optimizer_state", False)  # Default: False (fresh optimizer)
    if load_model_path and load_optimizer_state:
        try:
            checkpoint = torch.load(load_model_path, map_location=device)
            if 'optimizer_state_dict' in checkpoint:
                optim.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"  - Optimizer state also loaded from {load_model_path}")
            else:
                print(f"  - No optimizer state found in {load_model_path}, using fresh optimizer")
        except Exception as e:
            print(f"  - Warning: Could not load optimizer state: {e}, using fresh optimizer")
    elif load_model_path:
        print(f"  - Using fresh optimizer (load_optimizer_state=False)")
    logs = defaultdict(list)

    best_rewards_all = [0]*keep_best_n
    best_min_idxs_all = [None]*keep_best_n
    best_ws_all = [None]*keep_best_n
    
    # Timing variables
    timing_logs = defaultdict(list)
    
    # Early stopping variables
    last_best_rewards = None
    no_improvement_count = 0
    early_stopped = False
    
    # Best model tracking
    best_model_state = None
    best_model_reward = -float('inf')
    save_best_model_path = args.get("save_best_model_path", None)
    
    for iteration in pbar:
        start_total = time.time()
        
        # 1. Environment reset timing
        start_reset = time.time()
        # Get normalize_weights hyperparameter
        normalize_weights = args.get("normalize_weights", False)
        target_magnitude_factor = args.get("target_magnitude_factor", 0.5)
        # Convert target to correct dtype based on precision
        target_dtype = torch.float64 if precision == "float64" else torch.float32
        init_td = env.reset(env.gen_params(dt=dt, target=target.to(device=device, dtype=target_dtype), n=nnn, N=NN, batch_size=[batch_size], device=device, normalize_weights=normalize_weights, target_magnitude_factor=target_magnitude_factor, precision=precision))
        timing_logs["reset_time"].append(time.time() - start_reset)

        # 2. Rollout timing (this includes Sfromw computations)
        start_rollout = time.time()
        rollout_length = args.get("rollout_length", 100)  # Default: 100 timesteps
        
        # Add exploration noise during training
        exploration_noise = args.get("exploration_noise", 0.1)  # Default: 0.1 std noise
        if exploration_noise > 0:
            # Create exploration policy wrapper
            def exploration_policy(td):
                td_with_action = policy(td)
                # Add Gaussian noise to actions for exploration
                noise = torch.randn_like(td_with_action["action"]) * exploration_noise
                td_with_action["action"] = torch.clamp(td_with_action["action"] + noise, -1, 1)
                return td_with_action
            rollout = env.rollout(rollout_length, exploration_policy, tensordict=init_td, auto_reset=False)
        else:
            rollout = env.rollout(rollout_length, policy, tensordict=init_td, auto_reset=False)
        rollout_time = time.time() - start_rollout
        timing_logs["rollout_time"].append(rollout_time)
        
        # 3. Backward pass timing
        start_backward = time.time()
        traj_return = rollout["next", "reward"].mean()
        (-traj_return).backward()
        timing_logs["backward_time"].append(time.time() - start_backward)
        
        # 4. Optimizer step timing
        start_optim = time.time()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        timing_logs["optim_time"].append(time.time() - start_optim)
        
        # Get the best reward and its index from the batch
        last_rewards = rollout[..., -1]["next", "reward"]
        best_reward_idx = torch.argmax(last_rewards)
        best_reward = last_rewards[best_reward_idx].item()
                
        # Get corresponding min_idx, w for the best performing trajectory
        best_min_idx = rollout[..., -1]["next", "min_idx"][best_reward_idx]
        best_w = rollout[..., -1]["next", "w"][best_reward_idx]
        # print(f"[DEBUG-TEST] best_min_idx: {best_min_idx}")
        # print(f"[DEBUG-TEST] best reward: {best_reward}")
        
        # Update best reward and min_idx if current reward is better
        if best_reward > best_rewards_all[-1]:
            best_rewards_all.append(best_reward)
            best_min_idxs_all.append(best_min_idx.cpu().tolist() if torch.is_tensor(best_min_idx) else best_min_idx)
            best_ws_all.append(best_w.cpu().tolist() if torch.is_tensor(best_w) else best_w)
            
            rewards_tensor = torch.tensor(best_rewards_all)
            sorted_indices = torch.argsort(rewards_tensor, descending=True)
            keep_indices = sorted_indices[:keep_best_n]
            
            best_rewards_all = [best_rewards_all[i] for i in keep_indices]
            best_min_idxs_all = [best_min_idxs_all[i] for i in keep_indices]
            best_ws_all = [best_ws_all[i] for i in keep_indices]
            
            # Track the absolute best model state during training
            if best_reward > best_model_reward:
                best_model_reward = best_reward
                best_model_state = {
                    'policy_state_dict': policy.state_dict().copy(),
                    'optimizer_state_dict': optim.state_dict().copy(),
                    'stage': stage,
                    'iteration': iteration,
                    'best_reward': float(best_reward),
                    'args': args
                }
                print(f"[DEBUG] New best model found at iteration {iteration} with reward {best_reward:.6f}")
                print(f"[DEBUG] best_model_state created successfully")
        
        # Early stopping check
        if enable_early_stopping:
            current_best_rewards = tuple(sorted(best_rewards_all, reverse=True))
            if last_best_rewards is not None and current_best_rewards == last_best_rewards:
                no_improvement_count += 1
                if no_improvement_count >= early_stop_patience:
                    print(f"\nEarly stopping at training cycle {iteration+1}: No improvement for {early_stop_patience} cycles")
                    early_stopped = True
                    break
            else:
                no_improvement_count = 0
            last_best_rewards = current_best_rewards
        
        # Total iteration timing
        timing_logs["total_time"].append(time.time() - start_total)
        
        # Calculate and display timing stats every 10 iterations
        if iteration % 10 == 0 and iteration > 0:
            avg_times = {k: np.mean(v[-10:]) for k, v in timing_logs.items()}
            timing_str = f"Reset:{avg_times['reset_time']:.3f}s Rollout:{avg_times['rollout_time']:.3f}s Backward:{avg_times['backward_time']:.3f}s Optim:{avg_times['optim_time']:.3f}s Total:{avg_times['total_time']:.3f}s"
            
        # Display the best reward and timing info
        if iteration % 10 == 0:
            early_stop_info = f"no_improve: {no_improvement_count}/{early_stop_patience}" if enable_early_stopping else ""
            pbar.set_description(
                f"best: {best_rewards_all[0]:4.4f}, "
                f"current: {best_reward:4.4f}, "
                f"{early_stop_info}, "
                f"times- R:{timing_logs['rollout_time'][-1]:.2f}s T:{timing_logs['total_time'][-1]:.2f}s"
            )
        else:
            early_stop_info = f"no_improve: {no_improvement_count}/{early_stop_patience}" if enable_early_stopping else ""
            pbar.set_description(
                f"best: {best_rewards_all[0]:4.4f}, "
                f"current: {best_reward:4.4f}, "
                f"{early_stop_info}"
            )
        
        logs["best_reward"].append(best_reward)
        logs["best_min_idx"].append(best_min_idx.cpu() if torch.is_tensor(best_min_idx) else best_min_idx)
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
        # logs["last_current_Svector"].append(rollout[..., -1]["next", "Svector_current"])
        # logs["last_w"].append(rollout[..., -1]["next", "w"])
        # logs["test"].append(rollout[..., -1])
        
        # Real-time plot update every 20 cycles (configurable)
        plot_update_frequency = args.get("plot_update_frequency", 3)
        if iteration % plot_update_frequency == 0 and iteration > 0:
            # Create real-time plot with current progress
            plt.figure(figsize=(12, 5))
            
            # Left subplot: Best reward over iterations
            plt.subplot(1, 2, 1)
            plt.plot(logs["best_reward"], marker='o', markersize=3)
            plt.title(f"Best Reward Progress (Cycle {iteration+1}/{max_training_cycles})")
            plt.xlabel("Training Iteration")
            plt.ylabel("Best Reward")
            plt.grid(True)
            
            # Right subplot: Best reward colored by S-vector configuration
            plt.subplot(1, 2, 2)
            if len(logs["best_min_idx"]) > 0:
                min_idx_strs = ['_'.join(map(str, idx.cpu().tolist() if torch.is_tensor(idx) else idx)) if torch.is_tensor(idx) or isinstance(idx, list) else str(idx) for idx in logs["best_min_idx"]]
                unique_configs, counts = np.unique(min_idx_strs, return_counts=True)
                color_map = matplotlib.colormaps['tab20'](np.linspace(0, 1, len(unique_configs)))
                config_to_color = dict(zip(unique_configs, color_map))
                colors = [config_to_color[s] for s in min_idx_strs]
                
                x_values = range(len(logs["best_reward"]))
                y_values = logs["best_reward"]
                for i in range(len(x_values) - 1):
                    plt.plot(x_values[i:i+2], y_values[i:i+2], color=colors[i], marker='o', markersize=3, linestyle='-')
                if len(x_values) > 0:
                    plt.plot(x_values[-1], y_values[-1], color=colors[-1], marker='o', markersize=3)
                
                # Add legend for top 3 most common configurations (simplified for real-time)
                top3_indices = np.argsort(counts)[-3:]
                legend_elements = [plt.scatter([], [], c=[config_to_color[unique_configs[i]]],
                                            label=f'Config {unique_configs[i]}')
                                 for i in top3_indices]
                plt.legend(handles=legend_elements, fontsize=8)
            
            plt.title("Best Reward by S-vector Configuration")
            plt.xlabel("Training Iteration")
            plt.ylabel("Best Reward")
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save real-time plot (overwrites previous)
            run_suffix = f"_run{args.get('run_idx', '')}" if args.get('run_idx') is not None else ""
            realtime_plot_path = os.path.join(dir, f'training_plot_n{nnn}_N{NN}_lr{lr}_batch{batch_size}_{name}{run_suffix}.png')
            plt.savefig(realtime_plot_path)
            plt.close()
            
            print(f"Updated real-time plot at iteration {iteration+1}")
        
        scheduler.step()
    
    # Print final timing summary
    actual_iterations = len(logs["best_reward"])
    print(f"\n========== TRAINING SUMMARY for Stage {stage} ==========")
    print(f"VECTORIZED VERSION WITH EARLY STOPPING")
    print(f"Training cycles: {actual_iterations}/{max_iter} ({actual_iterations/max_iter*100:.1f}%)")
    print(f"Early stopped: {'Yes' if early_stopped else 'No'}")
    if early_stopped:
        print(f"Stopped after {no_improvement_count} training cycles without improvement (patience: {early_stop_patience})")
    print(f"Convergence efficiency: {actual_iterations/max_iter:.3f}")
    print(f"\n========== TIMING BREAKDOWN ==========")
    for key, times in timing_logs.items():
        avg_time = np.mean(times)
        total_time = np.sum(times)
        print(f"{key:15s}: avg={avg_time:.3f}s, total={total_time:.1f}s, {total_time/np.sum(timing_logs['total_time'])*100:.1f}%")
    
    # for log in logs["test"]:
    #     eps = 1e-6
    #     Svector_target = log['params','target']
    #     Svector_current = log['next','Svector_current']
    #     Svector_target_norm = Svector_target / (torch.norm(Svector_target, dim=-1, keepdim=True) + eps)
    #     Svector_current_norm = Svector_current / (torch.norm(Svector_current, dim=-1, keepdim=True) + eps)
                
    #     # Compute dot product along the last dimension
    #     dot = torch.sum(Svector_target_norm * Svector_current_norm, dim=-1)
    #     print("="*50)
    #     print(f"w: {log['next','w'][0]}\n")
    #     print(f"Svector_current: {Svector_current[0]}\n")
    #     print(f"Svector_target: {Svector_target[0]}\n")
    #     print(f"dot: {dot}\n")
    #     print(f"current norm: {Svector_current_norm}\n")
    #     print(f"target norm: {Svector_target_norm}\n")
    #     print(f"reward: {log['next','reward'][0]}\n")
    #     print("="*50)

    env.close()
    
    def plot():
        is_ipython = "inline" in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(logs["best_reward"])
        plt.title("returns")
        plt.xlabel("iteration")
        plt.subplot(1, 2, 2)
        
        # Convert min_idx tensors to integers before making strings
        # Convert min_idx tensors to unique string representations
        min_idx_strs = ['_'.join(map(str, idx.cpu().tolist() if torch.is_tensor(idx) else idx)) if torch.is_tensor(idx) or isinstance(idx, list) else str(idx) for idx in logs["best_min_idx"]]
        # min_idx_items = [idx.item() if torch.is_tensor(idx) else idx for idx in logs["best_min_idx"]]
        # min_idx_strs = [str(item) for item in min_idx_items]

        # Get unique configurations and their counts
        unique_configs, counts = np.unique(min_idx_strs, return_counts=True)

        # Create a color map for unique configurations using the updated API
        color_map = matplotlib.colormaps['tab20'](np.linspace(0, 1, len(unique_configs)))
        config_to_color = dict(zip(unique_configs, color_map))

        # Get colors for each point
        colors = [config_to_color[s] for s in min_idx_strs]
        
        # Use plt.plot with markers instead of plt.scatter to connect points
        # Also, apply colors individually point-by-point
        x_values = range(len(logs["best_reward"]))
        y_values = logs["best_reward"]
        for i in range(len(x_values) - 1):
             plt.plot(x_values[i:i+2], y_values[i:i+2], color=colors[i], marker='o', markersize=5, linestyle='-')
        # Plot the last point marker
        if len(x_values) > 0:
             plt.plot(x_values[-1], y_values[-1], color=colors[-1], marker='o', markersize=5)

        # Add legend for top 5 most common configurations
        top5_indices = np.argsort(counts)[-5:]
        legend_elements = [plt.scatter([], [], c=[config_to_color[unique_configs[i]]],
                                    label=f'Config {unique_configs[i]} ({counts[i]} times)')
                         for i in top5_indices]
        plt.legend(handles=legend_elements)

        plt.title("Best Reward by S-vector Configuration")
        plt.xlabel("iteration")
        if is_ipython:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        plt.show()
        
        # Save the plot
        run_suffix = f"_run{args.get('run_idx', '')}" if args.get('run_idx') is not None else ""
        file_path = os.path.join(dir, f'training_plot_n{nnn}_N{NN}_lr{lr}_batch{batch_size}_{name}{run_suffix}.png')
        plt.savefig(file_path)

    # ========== CHECKPOINT SAVE ==========
    # Save intermediate results after RL training completes, before ranking/scipy optimization
    # This allows recovery if the job times out during the post-training phase
    def to_json_serializable(obj):
        """Convert tensors, numpy arrays, and other objects to JSON-serializable format."""
        if obj is None:
            return None
        if torch.is_tensor(obj):
            return obj.cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [to_json_serializable(x) for x in obj]
        if isinstance(obj, (int, float, str, bool)):
            return obj
        return list(obj) if hasattr(obj, '__iter__') else obj

    checkpoint_data = {
        "checkpoint_type": "post_rl_training",
        "best_rewards_all": [float(r) if r is not None else None for r in best_rewards_all],
        "best_min_idxs_all": [to_json_serializable(idx) for idx in best_min_idxs_all],
        "best_ws_all": [to_json_serializable(w) for w in best_ws_all],
        "best_optimized_reward": float(max(best_rewards_all)) if best_rewards_all else 0.0,
        "target": to_json_serializable(target),
        "training_summary": {
            "actual_iterations": len(logs["best_reward"]),
            "max_iterations": max_iter,
            "early_stopped": early_stopped,
            "keep_best_n": keep_best_n
        },
        "status": "rl_training_complete"
    }

    run_suffix = f"_run{args.get('run_idx', '')}" if args.get('run_idx') is not None else ""
    checkpoint_path = os.path.join(dir, f'checkpoint_post_rl{run_suffix}.json')
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"\n[CHECKPOINT] Saved post-RL training checkpoint to: {checkpoint_path}")
    print(f"[CHECKPOINT] Best reward so far: {checkpoint_data['best_optimized_reward']:.10f}")
    # ========== END CHECKPOINT SAVE ==========

    print(f"target: {target}")
    Cuts = []
    for i in range(keep_best_n):
        if best_min_idxs_all[i] is not None:
            print(f"=====Rank {i+1} among all======")
            print(f"reward: {best_rewards_all[i]}")
            print(f"min_idx: {best_min_idxs_all[i]}")
            print(f"W configuration: {Ws_from_index(best_min_idxs_all[i], nnn, NN)}")
            Cuts.append(get_Cuts(best_min_idxs_all[i], nnn, NN))
            print(f"Cut: {Cuts[-1]}")
        else:
            print(f"=====Rank {i+1} among all======")
            print(f"reward: {best_rewards_all[i]}")
            print(f"min_idx: None (no solution found)")
            Cuts.append([])
    
    print(f"========Try to solve the problem with the cuts========\n")
        
    # Convert target to numpy for scipy operations
    target_numpy = target.cpu().numpy() if torch.is_tensor(target) else np.array(target.tolist())
    results = {}
    results["search_from_best_n"] = []
    
    # Step 1: Run scipy optimization for all keep_best_n trajectories in PARALLEL
    # Prepare data for parallel processing
    optimization_tasks = []
    valid_indices = []
    
    for i in range(keep_best_n):
        if best_min_idxs_all[i] is not None and best_ws_all[i] is not None:
            # Best performing trajectory values (reference solution)
            # Convert to numpy if tensor, otherwise use as-is
            if torch.is_tensor(best_ws_all[i]):
                w_reference = best_ws_all[i].cpu().numpy()
            else:
                w_reference = np.array(best_ws_all[i])
            
            optimization_tasks.append((w_reference, target_numpy, best_min_idxs_all[i], nnn, NN, epsilon, factor))
            valid_indices.append(i)
    
    # Determine number of CPUs to use
    available_cpus = get_available_cpus(0.9)
    num_cpus_to_use = min(10, available_cpus, len(optimization_tasks))
    
    print(f"Running parallel scipy optimization on {len(optimization_tasks)} trajectories using {num_cpus_to_use} CPUs...")

    # Run optimizations in parallel with timing
    scipy_start_time = time.time()
    if len(optimization_tasks) > 0:
        with multiprocessing.Pool(processes=num_cpus_to_use) as pool:
            parallel_results = pool.map(scipy_search_wrapper, optimization_tasks)
    else:
        parallel_results = []
    scipy_end_time = time.time()

    # Calculate scipy timing statistics
    scipy_total_time = scipy_end_time - scipy_start_time
    scipy_avg_time = scipy_total_time / len(optimization_tasks) if len(optimization_tasks) > 0 else 0.0
    print(f"Scipy optimization completed in {scipy_total_time:.2f}s (avg {scipy_avg_time:.3f}s per trajectory)")
    
    # Reconstruct results maintaining original order
    parallel_idx = 0
    for i in range(keep_best_n):
        if i in valid_indices:
            result = parallel_results[parallel_idx]
            # Don't compute gradient yet - just set empty list
            result["S_grad"] = []
            results["search_from_best_n"].append(result)
            parallel_idx += 1
        else:
            # Skip None entries
            results["search_from_best_n"].append({"reward": 0, "S_grad": []})
    
    # Step 2: Find the best trajectory
    results["best_index"] = max(range(len(results["search_from_best_n"])), 
                               key=lambda i: results["search_from_best_n"][i]["reward"])
    
    # Step 3: Compute gradient ONLY for the best trajectory
    gradient_start_time = time.time()
    if results["best_index"] < keep_best_n and best_ws_all[results["best_index"]] is not None:
        # Get the best trajectory's weights
        if torch.is_tensor(best_ws_all[results["best_index"]]):
            best_w_reference = best_ws_all[results["best_index"]].cpu().numpy()
        else:
            best_w_reference = np.array(best_ws_all[results["best_index"]])

        # Choose gradient computation method based on configuration
        skip_gradient = args.get("skip_gradient", False)
        gradient_method = args.get("gradient_method", "scipy")

        if skip_gradient:
            print("\n[run_stage] Skipping gradient computation (skip_gradient=True)")
            S_grad = [0.0] * len(target_numpy)
        elif gradient_method == "rl":
            # RL-based gradient: warm-start mini RL trainings for each dimension
            if best_model_state is None:
                print("[WARNING] best_model_state is None, falling back to scipy gradient method")
                S_grad = gradient_search(target_numpy, best_w_reference, best_min_idxs_all[results["best_index"]],
                                        nnn, NN, dS, epsilon, factor)
            else:
                print(f"\n[run_stage] Using RL-based gradient computation")
                base_reward = results["search_from_best_n"][results["best_index"]]["reward"]
                S_grad = gradient_search_rl(target_numpy, best_model_state, base_reward,
                                           nnn, NN, dS, args)
        elif gradient_method == "rl_v2":
            # RL-based gradient v2: random sampling on hyperplane + linear regression
            if best_model_state is None:
                print("[WARNING] best_model_state is None, falling back to scipy gradient method")
                S_grad = gradient_search(target_numpy, best_w_reference, best_min_idxs_all[results["best_index"]],
                                        nnn, NN, dS, epsilon, factor)
            else:
                print(f"\n[run_stage] Using RL-based gradient computation v2 (random sampling + linear regression)")
                base_reward = results["search_from_best_n"][results["best_index"]]["reward"]
                n_samples = args.get("n_gradient_samples", 100)
                cold_start = args.get("gradient_cold_start", False)
                distance_mode = args.get("gradient_distance_mode", "fixed")
                n_repeats = args.get("gradient_n_repeats", 1)
                n_gpus = args.get("gradient_n_gpus", 1)
                S_grad = gradient_search_rl_v2(target_numpy, best_model_state, base_reward,
                                              nnn, NN, dS, n_samples, args, cold_start=cold_start,
                                              distance_mode=distance_mode, n_repeats=n_repeats, n_gpus=n_gpus)
        else:
            # Original scipy-based gradient: assumes fixed W configurations
            print(f"\n[run_stage] Using scipy-based gradient computation")
            S_grad = gradient_search(target_numpy, best_w_reference, best_min_idxs_all[results["best_index"]],
                                    nnn, NN, dS, epsilon, factor)

        # Update the S_grad for the best trajectory
        results["search_from_best_n"][results["best_index"]]["S_grad"] = S_grad

    gradient_end_time = time.time()
    gradient_total_time = gradient_end_time - gradient_start_time
    gradient_method_used = args.get("gradient_method", "scipy")  # Get from args to avoid NameError
    print(f"\n========== GRADIENT COMPUTATION TIMING ==========")
    print(f"Gradient method: {gradient_method_used}")
    print(f"Total gradient time: {gradient_total_time:.1f}s ({gradient_total_time/60:.1f} minutes)")
    if gradient_method_used == "rl_v2":
        n_samples = args.get("n_gradient_samples", 100)
        avg_per_sample = gradient_total_time / n_samples if n_samples > 0 else 0
        print(f"Samples: {n_samples}, Avg per sample: {avg_per_sample:.1f}s")
    print(f"=================================================")
    
    results["name"] = name
    results["x_value"] = x_value
    # Convert final best rewards list items if they might still be tensors (though unlikely now)
    results["best_rewards_all"] = [r if isinstance(r, (int, float)) else r.item() for r in best_rewards_all]
    results["best_min_idxs_all"] = best_min_idxs_all # Already lists of ints
    results["best_ws_all"] = best_ws_all # Already lists of floats
    results["best_optimized_reward"] = results["search_from_best_n"][results["best_index"]]["reward"]
    results["S_grad_from_best"] = results["search_from_best_n"][results["best_index"]]["S_grad"]
    results["stage"] = stage  # Add stage number to results
    results["gradient_method"] = args.get("gradient_method", "scipy")  # Record which gradient method was used
    
    # Add training logs for plotting
    results["training_logs"] = {
        "best_reward": logs["best_reward"],
        "best_min_idx": [idx.cpu().tolist() if torch.is_tensor(idx) else idx for idx in logs["best_min_idx"]],
        "return": logs["return"],
        "last_reward": logs["last_reward"]
    }
    
    # Add early stopping metrics
    actual_iterations = len(logs["best_reward"])
    results["early_stopped"] = early_stopped
    results["actual_iterations"] = actual_iterations
    results["max_iterations"] = max_iter
    results["convergence_efficiency"] = actual_iterations / max_iter if max_iter > 0 else 1.0
    results["early_stop_patience_used"] = early_stop_patience
    results["no_improvement_count_final"] = no_improvement_count
    
    # Save timing information to results
    results["timing_stats"] = {
        "avg_reset_time": float(np.mean(timing_logs["reset_time"])),
        "avg_rollout_time": float(np.mean(timing_logs["rollout_time"])),
        "avg_backward_time": float(np.mean(timing_logs["backward_time"])),
        "avg_optim_time": float(np.mean(timing_logs["optim_time"])),
        "avg_total_time": float(np.mean(timing_logs["total_time"])),
        "total_rollout_time": float(np.sum(timing_logs["rollout_time"])),
        "total_time": float(np.sum(timing_logs["total_time"])),
        "rollout_percentage": float(np.sum(timing_logs["rollout_time"]) / np.sum(timing_logs["total_time"]) * 100),
        "scipy_total_time": float(scipy_total_time),
        "scipy_avg_time": float(scipy_avg_time),
        "scipy_num_trajectories": len(optimization_tasks),
        "gradient_total_time": float(gradient_total_time),
        "gradient_method": args.get("gradient_method", "scipy"),
        "version": "vectorized_with_early_stopping"
    }

    if not args.get("disable_plotting", False):
        plot()
    
    #########################
    # Model saving
    #########################
    save_model_path = args.get("save_model_path", None)
    save_best_model_path = args.get("save_best_model_path", None)
    
    print(f"[DEBUG] Model saving section reached!")
    print(f"[DEBUG] save_model_path: {save_model_path}")
    print(f"[DEBUG] save_best_model_path: {save_best_model_path}")
    print(f"[DEBUG] best_model_state is None: {best_model_state is None}")
    
    if save_model_path:
        try:
            # Save final model state
            checkpoint = {
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'stage': stage,
                'final_reward': results["best_optimized_reward"],
                'final_iteration': actual_iterations,
                'args': args
            }
            torch.save(checkpoint, save_model_path)
            print(f"Final model saved to: {save_model_path}")
        except Exception as e:
            print(f"Warning: Failed to save final model to {save_model_path}: {e}")
    
    if save_best_model_path:
        try:
            if best_model_state is not None:
                # Save the actual best model state found during training
                torch.save(best_model_state, save_best_model_path)
                print(f"Best model saved to: {save_best_model_path} (reward: {best_model_state['best_reward']:.6f}, iteration: {best_model_state['iteration']})")
            else:
                # Fallback: save current model if no best was tracked
                checkpoint = {
                    'policy_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'stage': stage,
                    'best_reward': results["best_optimized_reward"],
                    'final_iteration': actual_iterations,
                    'args': args
                }
                torch.save(checkpoint, save_best_model_path)
                print(f"Best model saved to: {save_best_model_path} (fallback to final model)")
        except Exception as e:
            print(f"Warning: Failed to save best model to {save_best_model_path}: {e}")
    
    print(f"========Stage {stage} Done========\n")
    
    return results

def run_stage_wrapper(stage_args_pair):
    """
    Wrapper function to unpack arguments for run_stage.
    This is needed because multiprocessing can't pickle lambda functions.
    """
    print(f"length of extreme rays: {len(EXTREME_RAYS)}")
    stage, args = stage_args_pair
    args["stage"] = stage
    
    # # n=3 example
    # args["x_value"] = 0.3 + 0.04*stage
    # args["name"] = f"target_08_07_1_06_1_1_{int(args['x_value']*100.5)}"
    # args["target"] = torch.tensor([0.8,0.7,1.0,0.6,1.0,1.0,args["x_value"]])
    
    # n=6 example - Using Ray 0 with coordinate 4 (AC subsystem)
    perturbations = [-0.006, -0.003, -0.125, 0.0125, 0.003, 0.006, -0.001, 0.001]
    args["x_value"] = 1.0 + perturbations[stage]  # Range 0.5 to 1.5 (10 stages: 0.5, 0.6, ..., 1.4)
    args["name"] = f"n6_N12_Ray0_coord0_x{int(args['x_value']*100.01)}"
    
    Svector_target = EXTREME_RAYS[0].copy()  # Use Ray 0 instead of Ray 20
    Svector_target[0] = args["x_value"]  # Use coordinate 4 (AC subsystem) instead of coordinate 2
    print(f"target: {Svector_target}")
    args["target"] = torch.tensor(Svector_target)
    
    return run_stage(args)

EXTREME_RAYS = []

# Load the converted extreme rays at module level
# Try multiple possible paths for the SA cone rays file
_ray_paths = [
    'data/n6/SA_cone_converted_rays.txt',  # New clean repo structure
    os.path.join(os.path.dirname(__file__), '..', 'data/n6/SA_cone_converted_rays.txt'),
    './extremal_rays/n6/converted_rays.txt',  # Legacy path
]
for _ray_path in _ray_paths:
    try:
        with open(_ray_path, 'r') as f:
            for line in f:
                ray = list(map(int, line.strip().split()))
                EXTREME_RAYS.append(ray)
        print(f"Loaded {len(EXTREME_RAYS)} extreme rays from {_ray_path}")
        break
    except FileNotFoundError:
        continue
else:
    print("Warning: SA_cone_converted_rays.txt not found, EXTREME_RAYS will be empty")

if __name__ == "__main__":
    args = {}
    # exp2 : ray 20 coordinate 4 -> seems like this ray is outside the cone
    # exp3 : ray 0 coordinate 4 -> inside the cone only at the x=1
    # exp4 : ray 0 coordinate 0 -> x <= 1.0 inside the cone  /  x > 1.0 outside the cone
    args["dir"] = "plots/n6/optimized_exp4_N12_timing"
    # args["dir"] = "plots/n3/gpu_batch_150_45000_n3_N5_1em5_noise04_normalized_scan"
    args["batch_size"] = 60  # Test with smaller batch first
    args["max_training_cycles"] = 400 # Maximum number of rollout-training cycles
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"  # Restore auto-detection
    args["dt"] = 1.1
    args["nnn"] = 6
    args["NN"] = 12
    args["lr"] = 4e-4
    args["epsilon"] = 0.1
    args["factor"] = 0.7
    args["keep_best_n"] = 10
    args["dS"] = 0.001
    args["plot_update_frequency"] = 5
    
    # Network architecture hyperparameters
    args["hidden_dim"] = 64     # Hidden layer dimension
    args["num_layers"] = 2      # Number of hidden layers
    args["rollout_length"] = 50 # Number of timesteps per rollout
    
    
    # Exploration hyperparameters
    args["exploration_noise"] = 0.15  # Gaussian noise std for action exploration
    
    # Weight normalization hyperparameter
    args["normalize_weights"] = True  # Normalize weights after each update for stability
    
    # Early stopping hyperparameters
    args["enable_early_stopping"] = True   # Enable early stopping
    args["early_stop_patience"] = 5       # Stop if no improvement for 15 training cycles
    
    max_stage = 8  # 10 stages for x_value range 0.5 to 1.4 (step 0.1)
    
    os.makedirs(args["dir"], exist_ok=True)
    results_file = os.path.join(args["dir"], "experiment_results.json")
    
    print(f"our target in n=6: {EXTREME_RAYS[20]}")

    # Load existing results or initialize an empty list
    try:
        with open(results_file, 'r') as f:
            save_data = json.load(f)
            if isinstance(save_data, dict):
                all_results = save_data.get("results", [])
                # You could also load and verify settings if needed
                # saved_args = save_data.get("settings", {})
            else:
                # Handle old format (list)
                if isinstance(save_data, list) and len(save_data) > 0:
                    # Check if first element is settings (doesn't have x_value)
                    if "x_value" not in save_data[0]:
                        all_results = save_data[1:]
                    else:
                        all_results = save_data
                else:
                    all_results = []
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = []

    # Write initial experiment settings before starting stages
    initial_save_data = {
        "settings": args,
        "results": all_results  # Empty or loaded from previous run
    }
    
    with open(results_file, 'w') as f:
        json.dump(initial_save_data, f, indent=4)
    
    print(f"Saved initial experiment settings to {results_file}")
    print(f"Starting experiment with {max_stage} stages...")

    # Run stages sequentially
    for stage in range(max_stage):
        print(f"\nStarting stage {stage}/{max_stage-1}...")
        result = run_stage_wrapper((stage, args))
        
        # Add the new result
        all_results.append(result)

        # Sort results by stage number (x_value)
        all_results.sort(key=lambda x: x["x_value"])
        
        # Save updated results with new structure
        save_data = {
            "settings": args,
            "results": all_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=4)
        
        print(f"Saved results for stage {result['stage']} (x_value: {result['x_value']})")
        print(f"Completed {len(all_results)} out of {max_stage} stages")
    
    print("All stages completed!")
    
    # Create final optimized rewards plot after all experiments are done
    if len(all_results) > 0:
        print("Creating final optimized rewards plot...")
        plt.figure(figsize=(12, 8))
        
        x_values = [r["x_value"] for r in all_results]
        # Use log(1-reward) with safe cutoff to avoid numerical issues
        # Clamp rewards to prevent log(0) when reward = 1.0
        safe_optimized = [np.log(1-min(r["best_optimized_reward"], 0.9999999999999999)) for r in all_results]
        safe_rl = [np.log(1-min(r["best_rewards_all"][0], 0.9999999999999999)) for r in all_results]
        
        plt.subplot(2, 1, 1)
        plt.plot(x_values, safe_optimized, marker='o', linestyle='-', label='Best Optimized Reward', linewidth=2, markersize=8)
        plt.plot(x_values, safe_rl, marker='s', linestyle='-', label='Best Reward from RL', linewidth=2, markersize=8)
        plt.xlabel("x_value")
        plt.ylabel("log(1 - Best Reward)")
        plt.title("Final Results: log(1 - Best Reward) vs x_value")
        plt.grid(True)
        plt.legend()
        
        # Add a second subplot with raw rewards for easier interpretation
        plt.subplot(2, 1, 2)
        raw_optimized = [r["best_optimized_reward"] for r in all_results]
        raw_rl = [r["best_rewards_all"][0] for r in all_results]
        
        plt.plot(x_values, raw_optimized, marker='o', linestyle='-', label='Best Optimized Reward', linewidth=2, markersize=8)
        plt.plot(x_values, raw_rl, marker='s', linestyle='-', label='Best Reward from RL', linewidth=2, markersize=8)
        plt.xlabel("x_value")
        plt.ylabel("Best Reward")
        plt.title("Final Results: Best Reward vs x_value")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # Save the final summary plot
        final_plot_path = os.path.join(args["dir"], f'optimized_rewards_plot.png')
        plt.savefig(final_plot_path)
        plt.close()
        
        print(f"Final optimized rewards plot saved to {final_plot_path}")
    else:
        print("No results to plot.")