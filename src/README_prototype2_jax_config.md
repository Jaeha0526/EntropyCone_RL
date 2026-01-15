# prototype2_jax_config.py

**Gradient-Following for HEC Boundary Discovery**

This script implements gradient-following to navigate from a starting entropy vector toward the HEC boundary. It's used for discovering unknown holographic entropy inequalities (Section 3.4: MMI rediscovery).

## Usage

```bash
python src/prototype2_jax_config.py --config configs/n3_mmi_finding/symmetric_N5.json
```

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to JSON configuration file |
| `--override-dir` | No | Override output directory from config |

## Configuration File Structure

```json
{
  "experiment": { ... },
  "training": { ... },
  "environment": { ... }
}
```

---

## Experiment Parameters

### Basic Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | - | Experiment name for logging |
| `description` | string | - | Human-readable description |
| `output_dir` | string | - | Directory for results output |
| `seed` | int | null | Random seed for reproducibility |

### Ray Source Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ray_source` | string | - | Source of starting ray (see Ray Sources) |
| `ray_index` | int | - | Index of ray (if using indexed source) |
| `custom_ray` | array | - | Custom starting point (if `ray_source: custom`) |
| `n_parties` | int | auto | Number of parties (required for custom rays) |

### Ray Sources

| Source | n | Description |
|--------|---|-------------|
| `n3_MMI_finding` | 3 | SA cone rays for N=3 MMI experiments |
| `custom` | varies | Custom entropy vector |
| `original` | 6 | Original N=6 extremal rays |
| `new` | 6 | Newly discovered N=6 rays |
| `spurious` | 5 | N=5 rays not in true HEC |
| `experiment4` | 5 | S6-unique rays from 3M enumeration |

### Gradient Following Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `velocity` | float | - | Step size multiplier for movement |
| `moves` | int | - | Number of gradient-following stages |
| `momentum` | float | - | Momentum coefficient (0-1) for gradient smoothing |
| `mode` | string | `reward_only` | Optimization mode |
| `gradient_clip_norm` | float | null | Maximum gradient norm (optional) |
| `use_log_weighting` | bool | true | Weight gradient by 1/(1-r) for log reward |

### Early Stopping (Trajectory)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_trajectory_early_stop` | bool | true | Enable trajectory early stopping |
| `gradient_convergence_threshold` | float | 1e-6 | Stop if gradient norm below this |
| `reward_plateau_threshold` | float | 1e-8 | Stop if reward variation below this |
| `min_stages_before_convergence_check` | int | 5 | Minimum stages before checking convergence |
| `min_stages_before_plateau_check` | int | 10 | Minimum stages before checking plateau |
| `plateau_window_size` | int | 5 | Window size for plateau detection |

---

## Movement Mode Configuration

### Standard Mode (Default)

Direct gradient application with safety scaling to prevent negative coordinates.

```json
{
  "experiment": {
    "movement_mode": "standard"
  }
}
```

### Safe Direction Mode

QP-constrained movement respecting known facet constraints.

```json
{
  "experiment": {
    "movement_mode": "safe_direction",
    "safe_direction": {
      "known_facets_file": "data/n3/ray8_known_facets.txt",
      "saturation_tolerance": 0.0001,
      "safety_factor": 0.9,
      "verbose": true
    }
  }
}
```

### Safe Direction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `known_facets_file` | string | `data/n3/ray8_known_facets.txt` | File with known facet constraints |
| `saturation_tolerance` | float | 1e-4 | Tolerance for facet saturation |
| `safety_factor` | float | 0.9 | Fraction of max safe distance to move |
| `verbose` | bool | false | Print QP solver details |

### Escape Mode (Extension of Safe Direction)

Prevents getting trapped near saturated facets by forcing minimum increase.

```json
{
  "experiment": {
    "movement_mode": "safe_direction",
    "safe_direction": {
      "use_escape_mode": true,
      "min_increase_rate": 0.05,
      "buffer_factor": 1.5,
      "escape_threshold_factor": 1.5
    }
  }
}
```

### Escape Mode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_escape_mode` | bool | false | Enable escape mode |
| `min_increase_rate` | float | 0.05 | Minimum rate for weak facets to increase |
| `buffer_factor` | float | 1.5 | Buffer multiplier for dS (distance) |
| `escape_threshold_factor` | float | 1.5 | Factor for escape threshold |

### Hybrid Mode

Combines escape and max-min directions with configurable blending.

```json
{
  "experiment": {
    "movement_mode": "safe_direction",
    "safe_direction": {
      "use_hybrid_mode": true,
      "hybrid_alpha": 0.5
    }
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_hybrid_mode` | bool | false | Enable hybrid mode |
| `hybrid_alpha` | float | 0.5 | Blending: 1.0=pure escape, 0.0=pure maxmin |

---

## Training Parameters

### Core RL Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NN` | int | **required** | Total graph vertices |
| `batch_size` | int | **required** | Number of parallel environments |
| `lr` | float | **required** | Learning rate |
| `max_training_cycles` | int | **required** | Maximum iterations per stage |

### Policy Network

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | int | 64 | Hidden layer dimension |
| `num_layers` | int | 2 | Number of hidden layers |
| `rollout_length` | int | 50 | Steps per rollout |

### Gradient Estimation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gradient_method` | string | `scipy` | `scipy`, `rl_v1`, or `rl_v2` |
| `dS` | float | 0.00001 | Perturbation size for gradient |
| `n_gradient_samples` | int | 100 | Samples for RL gradient estimation |
| `gradient_cold_start` | bool | false | Reset network for each gradient sample |
| `gradient_distance_mode` | string | `fixed` | `fixed` or `adaptive` |
| `gradient_n_repeats` | int | 1 | Repeat gradient computation |

### Other Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dt` | float | 1.1 | Action scaling factor |
| `epsilon` | float | 0.1 | Exploration parameter |
| `factor` | float | 0.7 | Epsilon decay factor |
| `keep_best_n` | int | 10 | Best weights to track |
| `exploration_noise` | float | 0.15 | Gaussian noise std |
| `normalize_weights` | bool | true | Normalize edge weights |
| `warm_start` | bool | true | Warm-start from previous stage |
| `precision` | string | `float64` | `float32` or `float64` |

### Early Stopping (Per-Stage)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_early_stopping` | bool | true | Enable RL early stopping |
| `early_stop_patience` | int | 5 | Patience for early stopping |

---

## Environment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | string | auto | `cuda` or `cpu` |
| `seed` | int | null | Random seed |

---

## Example Configuration (N=3 MMI Finding)

```json
{
  "experiment": {
    "name": "N=3 MMI Finding from Symmetric Point",
    "description": "Gradient following to rediscover MMI boundary",
    "ray_source": "custom",
    "custom_ray": [0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857],
    "n_parties": 3,
    "velocity": 0.7,
    "moves": 10,
    "momentum": 0.3,
    "output_dir": "results/n3_mmi_finding/symmetric",
    "seed": 42,
    "movement_mode": "safe_direction",
    "safe_direction": {
      "known_facets_file": "data/n3/ray8_known_facets.txt",
      "saturation_tolerance": 0.0001,
      "safety_factor": 0.9,
      "use_escape_mode": true,
      "min_increase_rate": 0.05,
      "buffer_factor": 1.5,
      "escape_threshold_factor": 1.5
    }
  },
  "training": {
    "NN": 5,
    "batch_size": 60,
    "lr": 0.0001,
    "max_training_cycles": 2000,
    "hidden_dim": 64,
    "num_layers": 2,
    "rollout_length": 50,
    "dt": 0.1,
    "epsilon": 0.1,
    "dS": 0.001,
    "gradient_method": "rl_v2",
    "n_gradient_samples": 50,
    "precision": "float64"
  },
  "environment": {
    "device": "cuda",
    "seed": 42
  }
}
```

---

## Output Files

| File | Description |
|------|-------------|
| `experiment_results.json` | Full trajectory data for all stages |
| `hyperparameters.json` | Complete configuration |
| `coordinate_evolution.png` | Trajectory visualization |
| `training_plot_stage_*.png` | Per-stage training curves |
| `gradient_samples_stage_*.json` | Raw gradient samples (if rl_v2) |
| `models/stage_*_best.pth` | Best model checkpoints |

---

## Gradient Methods

### `scipy` (Default)
Direct scipy.optimize for local gradient descent. Fast but may miss global structure.

### `rl_v1`
Standard policy gradient from RL training. Uses learned policy to estimate gradient.

### `rl_v2` (Recommended)
Random sampling + linear regression for gradient estimation:
1. Sample `n_gradient_samples` random perturbations around current point
2. Compute reward for each perturbed point
3. Fit linear regression to estimate gradient direction
4. More robust than rl_v1, especially near boundaries

---

## Algorithm Overview

**Per-Stage Process:**
1. Train RL agent to maximize reward at current position
2. Estimate gradient using selected method (scipy/rl_v1/rl_v2)
3. Apply momentum smoothing to gradient
4. Compute safe movement direction (if using safe_direction mode)
5. Move to new position: `new_pos = pos + velocity * safe_direction`
6. Check early stopping conditions
7. Repeat for `moves` stages

**Safe Direction QP:**
Solves: `min ||d - gradient||^2` subject to:
- All saturated facets: `a_i · d >= 0` (don't decrease)
- If escape mode: weak facets must increase by `min_increase_rate`

---

## Tips

- **Velocity**: Start with 0.01-0.1 for careful exploration, 0.5-1.0 for faster movement
- **Momentum**: 0.3-0.5 smooths trajectory, 0.0 for pure gradient descent
- **dS**: Smaller = more accurate gradient, larger = faster but noisier
- **n_gradient_samples**: 50-100 for reasonable accuracy, 500+ for high precision
- **escape_mode**: Enable when getting stuck near boundaries
