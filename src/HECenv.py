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
from matplotlib import pyplot as plt

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


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
        else:
            # Non-batched case
            reward = torch.dot(Svector_target, Svector_current) / (torch.norm(Svector_target) * torch.norm(Svector_current) + eps)
            done = reward < 1e-3
        
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
    Convert the state vector from the w-space to the s-space.
    Handles batched inputs.
    """
    # print("="*50)
    # print("Sfromw called")
    # print(f"[DEBUG] w shape: {w.shape if torch.is_tensor(w) else 'not tensor'}")
    # print(f"[DEBUG] n: {n}")
    # print(f"[DEBUG] N: {N}")
    # print("="*50)
    
    # Handle batch dimensions
    batch_size = []
    if torch.is_tensor(w) and w.dim() > 1:
        batch_size = list(w.shape[:-1])
    
    # Convert n and N to Python integers if they are tensors
    if torch.is_tensor(n):
        if n.dim() == 0:  # 0-dim tensor (scalar)
            n_val = int(n.item())
        else:  # multi-dim tensor
            n_val = int(n[0].item())
    else:
        n_val = n
        
    if torch.is_tensor(N):
        if N.dim() == 0:  # 0-dim tensor (scalar)
            N_val = int(N.item())
        else:  # multi-dim tensor
            N_val = int(N[0].item())
    else:
        N_val = N
    
    # Check if w has batch dimensions and process each batch separately
    if batch_size:
        # Preallocate output tensor for all batches
        batched_Svector = torch.zeros(*batch_size, 2**n_val-1, device=w.device)
        batched_min_idx = torch.zeros(*batch_size, 2**n_val-1, dtype=torch.long, device=w.device)
        
        # Process each batch
        if len(batch_size) == 1:
            for b in range(batch_size[0]):
                batched_Svector[b], batched_min_idx[b] = Sfromw_single(w[b], n_val, N_val)
        else:
            # Flatten batch dimensions for processing
            flat_batch_size = torch.prod(torch.tensor(batch_size)).item()
            flat_w = w.view(flat_batch_size, -1)
            flat_result = torch.zeros(flat_batch_size, 2**n_val-1, device=w.device)
            flat_min_idx = torch.zeros(flat_batch_size, 2**n_val-1, dtype=torch.long, device=w.device)
            
            for b in range(flat_batch_size):
                flat_result[b], flat_min_idx[b] = Sfromw_single(flat_w[b], n_val, N_val)
            
            # Reshape back to original batch dimensions
            batched_Svector = flat_result.view(*batch_size, -1)
            batched_min_idx = flat_min_idx.view(*batch_size, -1)
        return batched_Svector, batched_min_idx
    else:
        # Non-batched case
        return Sfromw_single(w, n_val, N_val)

def Sfromw_single(w, n, N):
    """
    Non-batched version of Sfromw that processes a single example.
    """
    # Determine device based on w input
    device = w.device if torch.is_tensor(w) else torch.device('cpu') 
    Svector = torch.zeros(2**n-1, device=device)
    min_indices = torch.zeros(2**n-1, dtype=torch.long, device=device)
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
        tensordict = self.gen_params(self.dt, self.target, self.n, self.N, batch_size=self.batch_size, device=self.device)
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
    
    # Initialize random weights
    w = torch.rand(*tensordict.batch_size, (self.N*(self.N-1))//2, generator=self.rng, device=self.device)
    
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
def gen_params(dt=0.01, target=None, n=3, N=5, batch_size=None, device="cpu"):
        """
        Generate the parameters for the HEC problem.
        """
        # print("generating params...")
        # print(f"[DEBUG-GEN] Input target: {target}")
        # print(f"[DEBUG-GEN] Input target shape: {target.shape if torch.is_tensor(target) else 'not tensor'}")
        # print(f"[DEBUG-GEN] Input batch_size: {batch_size}")
        # print(f"[DEBUG-GEN] Input dt: {dt}, n: {n}, N: {N}")
        # print(f"[DEBUG-GEN] Input device: {device}")
        
        if batch_size is None:
            batch_size = []
            
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
            
            # Handle target
            if target is not None:
                # If target is not a tensor, convert it
                if not torch.is_tensor(target):
                    # Add device=device
                    batched_params["target"] = torch.tensor(target, device=device).expand(batch_size[0], -1)
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
                "N": torch.tensor(N, dtype=torch.int64, device=device) if not torch.is_tensor(N) else N.to(device)
            }
            if target is not None:
                # Add device=device
                batched_params["target"] = torch.tensor(target, device=device) if not torch.is_tensor(target) else target.to(device)
        
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
    def __init__(self, dt, target=None, n=3, N=5, seed=None, device="cpu"):
        self.n = n
        self.N = N
        self.dt = dt
        
        # Handle the target
        if target is None:
            # Default target with ones
            self.target = torch.ones(2**n-1, device=device)
        elif isinstance(target, (int, float)):
            # Scalar value
            self.target = torch.ones(2**n-1, device=device) * target
        elif torch.is_tensor(target) and target.dim() == 1:
            # Already a 1D tensor
            self.target = target.to(device)
        else:
            # Convert to tensor if not already
            self.target = torch.tensor(target, device=device)
            
        self.device = device
        # Pass device to gen_params
        td_params = self.gen_params(dt, self.target, n, N, device=device)

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

    # Initial guess (start with reference solution)
    initial_guess = w_reference.copy()

    # Constraints: each element must be within 0.01 of reference value
    explore = True
    
    while explore:
        print(f"explore within epsilon: {epsilon}")
        bounds = [(max(0, ref-epsilon), ref+epsilon) for ref in w_reference]

        # Solve the optimization problem
        result = minimize(
            objective, 
            initial_guess,
            method='L-BFGS-B',  # Works well with bounds
            bounds=bounds,
            options={'ftol': 1e-10}  # Tight tolerance for better convergence
        )

        # Get the optimized weights
        w_optimal = result.x    

        # Check if we found a solution that satisfies equations
        reward = np.dot(A @ w_optimal, target) / (np.linalg.norm(A @ w_optimal) * np.linalg.norm(target))
        is_solution = np.allclose(reward, 1, atol=1e-5)

        print("Optimization successful:", result.success)
        print("\nReference weights:", [f"{w:.4f}" for w in w_reference])
        print("\nOptimal weights:", [f"{w:.4f}" for w in w_optimal])
        
        real_value, real_idx = Sfromw_single(w_optimal, n, N)
        # print(f"\nreal values: {[f'{v:.4f}' for v in real_value.tolist()]}")
        real_idx = real_idx.cpu().tolist() if torch.is_tensor(real_idx) else [i.item() if torch.is_tensor(i) else i for i in real_idx]
        wanted_idx = min_idx
        print(f"\nreal idx: {real_idx}")
        print(f"\nwanted idx: {wanted_idx}")
        
        if real_idx == wanted_idx:
            print(f"explored within the same w configuration.")
            explore = False
        
            print(f"\nachieved values: {[f'{v:.4f}' for v in A @ w_optimal]}")
            print(f"\ntarget: {[f'{v:.4f}' for v in target]}")
            print(f"\nreward: {reward:.6f}")
            print("Found solution satisfies equations:", is_solution)
        else:
            print(f"explored out of the same w configuration.")
            epsilon *= factor
            
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


def gradient_search(S_target, w_reference, min_idx,n, N, dS=0.001, epsilon=0.1, factor=0.9):
    """
    S_target: the target S-vector
    w_reference: the reference weights
    n: the number of variables
    N: the number of elements in the W configuration
    """
    Cuts = get_Cuts(min_idx, n, N)
    S_grad = []
    base_results = scipy_search(w_reference, S_target, min_idx, n, N, epsilon, factor)
    base_reward = base_results["reward"]
    for i in range(len(S_target)):
        dS_vector = np.zeros(len(S_target))
        dS_vector[i] = dS
        S_target_perturbed = S_target + dS_vector
        results = scipy_search(w_reference, S_target_perturbed, min_idx, n, N, epsilon, factor)
        S_grad.append((results["reward"]-base_reward)/dS)
    return S_grad



#########################
# Test the environment
#########################
print("="*50)
print("Test the environment")
target = torch.tensor([1.0]*7)
print(f"[DEBUG-TEST] Original target: {target.shape}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG-TEST] Device: {device}")
env = HECenv(dt=0.01, target=target, n=3, N=5, device=device)
print(f"[DEBUG-TEST] After creating env, env.target shape: {env.target.shape}")


#########################
# Wrap the environment
#########################
# env = TransformedEnv(
#     env,
#     # Unsqueeze the observations that we will concatenate
#     UnsqueezeTransform(
#         dim=-1,
#         in_keys=["w", "Svector_current", "Svector_difference"],
#         in_keys_inv=["w", "Svector_current", "Svector_difference"],
#     ),
# )
env = TransformedEnv(env)

# Now add a CatTensors transform to combine your vectors into a single observation:
cat_transform = CatTensors(
    in_keys=["w", "Svector_current", "Svector_difference"],
    out_key="observation",
    del_keys=False
)
env.append_transform(cat_transform)


#########################
# Check the environment
#########################
print("="*50)
print("Check the environment specs")
check_env_specs(env)
print("="*50)
print("="*50)
print("="*50)


#########################
# Test the rollout
#########################
print("reset (batch size of 3)")
batch_size = 5 # number of environments to be executed in batch
print(f"[DEBUG-TEST] Creating params with batch_size={batch_size}")
# Pass device to gen_params
params = env.gen_params(dt=0.01, target=env.target, n=3, N=5, batch_size=[batch_size], device=device)
print(f"[DEBUG-TEST] After gen_params, params['params', 'target'] shape: {params['params', 'target'].shape}")
# Pass device to gen_params when resetting within rollout
td = env.reset(env.gen_params(dt=0.01, target=env.target, n=3, N=5, batch_size=[batch_size], device=device))
print(f"[DEBUG-TEST] After reset, td['params', 'target'] shape: {td['params', 'target'].shape}")
# print("reset (batch size of 3)", td)
td = env.rand_step(td)
# print("rand step (batch size of 3)", td)

rollout = env.rollout(
    3,
    auto_reset=False,  # we're executing the reset out of the ``rollout`` call
    # Pass device to gen_params
    tensordict=env.reset(env.gen_params(dt=0.01, target=target, n=3, N=5, batch_size=[batch_size], device=device)),
)
print("="*50)
print("Test the rollout finished")
print("="*50)
# print("rollout of len 3 (batch size of 5):", rollout)

dir = "plots/with_optimization/exp6"
os.makedirs(dir, exist_ok=True)
results_file = os.path.join(dir, "experiment_results.json")

# Load existing results or initialize an empty list
try:
    with open(results_file, 'r') as f:
        all_results = json.load(f)
        if not isinstance(all_results, list):
            print(f"Warning: Existing file {results_file} did not contain a list. Initializing with empty list.")
            all_results = []
except (FileNotFoundError, json.JSONDecodeError):
    all_results = []

for stage in range(1):
    #########################
    # Setting up the experiment
    #########################
    batch_size = 100
    n_iter = 20000  # set to 20_000 for a proper training
    def x_value(s):
        return 0.39
    target = torch.tensor([0.8,0.7,1.0,0.6,1.0,1.0,x_value(stage)])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = 1
    nnn=3
    NN=5
    lr = 1e-5
    epsilon = 0.3
    factor = 0.8
    keep_best_n = 10
    dS = 0.001
    name = f"target_08_07_1_06_1_1_39"
    
    
    #########################
    # Setting up the environment
    #########################
    env.close()
    env = None
    env = HECenv(dt=dt, target=target, n=nnn, N=NN, device=device)
    env = TransformedEnv(env)
    env.append_transform(cat_transform)
    
    #########################
    # Define policy
    #########################
    torch.manual_seed(0)
    env.set_seed(0)

    net = nn.Sequential(
        nn.LazyLinear(64),
        nn.Tanh(),
        nn.LazyLinear(64),
        nn.Tanh(),
        nn.LazyLinear(64),
        nn.Tanh(),
        nn.LazyLinear(1),
    )
    # Move network to the correct device
    net.to(device)

    net.train()
    policy = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["action"],
    )

    #########################
    # Train the policy and plot the results
    #########################
    pbar = tqdm.tqdm(range(n_iter // batch_size))
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_iter)
    logs = defaultdict(list)

    best_rewards_all = [0]*keep_best_n
    best_min_idxs_all = [None]*keep_best_n
    best_ws_all = [None]*keep_best_n
    for _ in pbar:
        # Pass device to gen_params
        init_td = env.reset(env.gen_params(dt=dt, target=target.to(device), n=nnn, N=NN, batch_size=[batch_size], device=device))
        rollout = env.rollout(100, policy, tensordict=init_td, auto_reset=False)
        traj_return = rollout["next", "reward"].mean()
        (-traj_return).backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        
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
            best_rewards_all.append(best_reward) # Convert tensor to float
            best_min_idxs_all.append(best_min_idx.cpu().tolist() if torch.is_tensor(best_min_idx) else best_min_idx) # Convert tensor to list
            best_ws_all.append(best_w.cpu().tolist() if torch.is_tensor(best_w) else best_w) # Convert tensor to list
            
            rewards_tensor = torch.tensor(best_rewards_all)
            sorted_indices = torch.argsort(rewards_tensor, descending=True)
            keep_indices = sorted_indices[:keep_best_n]
            
            best_rewards_all = [best_rewards_all[i] for i in keep_indices]
            best_min_idxs_all = [best_min_idxs_all[i] for i in keep_indices]
            best_ws_all = [best_ws_all[i] for i in keep_indices]
        
        # print(f"best_rewards_all: {best_rewards_all}")
        # print(f"debug type of best_rewards_all: {type(best_rewards_all[0])}")
        # Display the best reward and its index
        pbar.set_description(
            f"best among all: {best_rewards_all[0]:4.4f}, "
            f"best reward: {best_reward:4.4f}, "
            f"best min_idx: {best_min_idx}"
            # f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}, "
            # f"last current_Svector: {rollout[..., -1]['next','Svector_current']}"
            # f"last w: {rollout[..., -1]['next','w'][0]}"
        )
        
        logs["best_reward"].append(best_reward)
        logs["best_min_idx"].append(best_min_idx.cpu() if torch.is_tensor(best_min_idx) else best_min_idx)
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
        # logs["last_current_Svector"].append(rollout[..., -1]["next", "Svector_current"])
        # logs["last_w"].append(rollout[..., -1]["next", "w"])
        # logs["test"].append(rollout[..., -1])
        scheduler.step()
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

    def plot():

        is_ipython = "inline" in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        with plt.ion():
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
            file_path = os.path.join(dir, f'training_plot_n{nnn}_N{NN}_lr{lr}_batch{batch_size}_{name}.png')
            plt.savefig(file_path)
    
    print(f"target: {target}")
    Cuts = []
    for i in range(keep_best_n):
        print(f"=====Rank {i+1} among all======")
        print(f"reward: {best_rewards_all[i]}")
        print(f"min_idx: {best_min_idxs_all[i]}")
        print(f"W configuration: {Ws_from_index(best_min_idxs_all[i], nnn, NN)}")
        Cuts.append(get_Cuts(best_min_idxs_all[i], nnn, NN))
        print(f"Cut: {Cuts[-1]}")
    
    print(f"========Try to solve the problem with the cuts========\n")
        
    # Convert target to numpy for scipy operations
    target_numpy = target.cpu().numpy() if torch.is_tensor(target) else np.array(target.tolist())
    results = {}
    results["search_from_best_n"] = []
    for i in range(keep_best_n):
        # Best performing trajectory values (reference solution)
        # Convert to numpy if tensor, otherwise use as-is
        if torch.is_tensor(best_ws_all[i]):
            w_reference = best_ws_all[i].cpu().numpy()
        else:
            w_reference = np.array(best_ws_all[i])
        result = scipy_search(w_reference, target_numpy, best_min_idxs_all[i], nnn, NN, epsilon, factor)
        S_grad = gradient_search(target_numpy, w_reference, best_min_idxs_all[i], nnn, NN, dS, epsilon, factor)
        result["S_grad"] = S_grad
        results["search_from_best_n"].append(result)

    results["name"] = name
    results["x_value"] = x_value(stage)
    # Convert final best rewards list items if they might still be tensors (though unlikely now)
    results["best_rewards_all"] = [r if isinstance(r, (int, float)) else r.item() for r in best_rewards_all] 
    results["best_min_idxs_all"] = best_min_idxs_all # Already lists of ints
    results["best_ws_all"] = best_ws_all # Already lists of floats
    results["best_index"] = max(range(len(results["search_from_best_n"])), 
                               key=lambda i: results["search_from_best_n"][i]["reward"])
    results["best_optimized_reward"] = results["search_from_best_n"][results["best_index"]]["reward"]
    results["S_grad_from_best"] = results["search_from_best_n"][results["best_index"]]["S_grad"]
    
    # Save the results
    all_results.append(results)

    # Write the updated list back to the file after each iteration
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4) # indent makes it readable
        
    # Create a plot for optimized rewards
    plt.figure(figsize=(10, 6))
    x_values = [result["x_value"] for result in all_results]
    optimized_rewards = [np.log(1-result["best_optimized_reward"]) for result in all_results]
    rewards_from_RL = [np.log(1-result["best_rewards_all"][0]) for result in all_results]
    
    plt.plot(x_values, optimized_rewards, marker='o', linestyle='-', label='Best Optimized Reward')
    plt.plot(x_values, rewards_from_RL, marker='o', linestyle='-', label='Best Reward from RL')
    plt.xlabel("x_value")
    plt.ylabel("Best Optimized Reward")
    plt.title("Best Optimized Reward vs x_value")
    plt.grid(True)
    plt.legend()
    
    # Save the optimized rewards plot
    opt_rewards_path = os.path.join(dir, f'optimized_rewards_plot.png')
    plt.savefig(opt_rewards_path)
    plt.close()

    print(f"========Stage {stage} Done========\n")


    plot()
    
    
    
