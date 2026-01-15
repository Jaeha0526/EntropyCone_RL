# HECenv_parallel_config.py

**SA Cone Ray Classification Runner**

This script runs RL-based classification experiments to determine whether entropy vectors lie inside the Holographic Entropy Cone (HEC). It uses JAX-accelerated training for ~334x speedup.

## Usage

```bash
python src/HECenv_parallel_config.py --config configs/n6_sa_cone_classification/template_N18.json
```

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to JSON configuration file |

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | - | Experiment name for logging |
| `description` | string | - | Human-readable description |
| `output_dir` | string | - | Directory for results output |
| `target_ray_index` | int | - | Index of ray to classify (1-indexed for most sources) |
| `target_ray_source` | string | - | Source of target ray (see Ray Sources below) |
| `run_name` | string | auto | Custom name for this run |

### Ray Sources

| Source | Description | File Path |
|--------|-------------|-----------|
| `sa_cone` | 208 SA cone extremal rays (N=6) | `data/n6/SA_cone_converted_rays.txt` |
| `custom` | Custom entropy vector | Specified in `w_optimal` field |
| `new` | Newly discovered rays | `extremal_rays/lp_ray_finder/converted_new_rays.txt` |
| `genuine_new` | Corrected S7-deduplicated rays | `extremal_rays/n6/new_extremal_rays/...` |
| `original` | Original extremal rays | `extremal_rays/n6/rays.txt` |
| `spurious` | Rays not in true HEC (n=5) | `extremal_rays/n5/from_SSA_MMI/...` |
| `experiment4` | S6-unique rays from 3M enumeration | `extremal_rays/n5/from_complete_SA_MMI/...` |

---

## Training Parameters

### Core RL Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | int | auto | Number of parties (auto-detected from ray dimension) |
| `N` | int | **required** | Total graph vertices = n + 1 + (N - n - 1) internal vertices |
| `batch_size` | int | **required** | Number of parallel environments |
| `lr` | float | **required** | Learning rate for policy network |
| `n_iter` | int | 5000 | Maximum training iterations |

### Policy Network Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | int | 128 | Hidden layer dimension |
| `num_layers` | int | 3 | Number of hidden layers |
| `rollout_length` | int | 100 | Steps per rollout before update |

### Exploration & Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dt` | float | 1.0 | Action scaling factor |
| `epsilon` | float | 0.1 | Initial exploration parameter |
| `factor` | float | 0.7 | Epsilon decay factor |
| `exploration_noise` | float | 0.1 | Gaussian noise std for exploration |
| `keep_best_n` | int | 5 | Number of best weights to track |
| `normalize_weights` | bool | true | Normalize edge weights |
| `target_magnitude_factor` | float | 0.5 | Target entropy vector scaling |

### Gradient Estimation (Optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gradient_method` | string | `scipy` | Method: `scipy`, `rl_v1`, `rl_v2` |
| `skip_gradient` | bool | false | Skip gradient computation (classification only) |
| `dS` | float | 0.001 | Perturbation size for gradient sampling |
| `n_gradient_samples` | int | 100 | Number of samples for gradient estimation |

### Early Stopping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_early_stopping` | bool | true | Enable early stopping |
| `early_stop_patience` | int | 5 | Iterations without improvement before stopping |

### Multi-Run Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `multi_run` | bool | false | Enable multiple independent runs |
| `num_runs` | int | 1 | Number of runs (requires `multi_run: true`) |

---

## Environment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | string | auto | `cuda` or `cpu` (auto-detects GPU) |
| `seed` | int | 42 | Random seed for reproducibility |
| `precision` | string | `float32` | Numeric precision: `float32` or `float64` |
| `num_workers` | int | 1 | Number of parallel workers |

---

## Example Configurations

### Basic Classification (SA Cone Ray)

```json
{
  "experiment": {
    "name": "SA Cone Ray 181 Classification",
    "output_dir": "results/sa_cone/ray181_N18",
    "target_ray_index": 181,
    "target_ray_source": "sa_cone"
  },
  "training": {
    "n": 6,
    "N": 18,
    "batch_size": 120,
    "lr": 0.0001,
    "n_iter": 3000,
    "hidden_dim": 128,
    "num_layers": 4,
    "rollout_length": 100,
    "enable_early_stopping": true,
    "early_stop_patience": 7,
    "skip_gradient": true
  },
  "environment": {
    "device": "cuda",
    "seed": 42,
    "precision": "float64"
  }
}
```

### Multi-Run Classification

```json
{
  "experiment": {
    "name": "SA Cone Ray 146 Multi-Run",
    "output_dir": "results/sa_cone/ray146_N18_multirun",
    "target_ray_index": 146,
    "target_ray_source": "sa_cone"
  },
  "training": {
    "N": 18,
    "batch_size": 120,
    "lr": 0.0001,
    "n_iter": 3000,
    "multi_run": true,
    "num_runs": 20,
    "skip_gradient": true
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
| `experiment_info.json` | Full configuration and metadata |
| `progressive_results.json` | Real-time tracking (multi-run mode) |
| `results_run_*.json` | Per-run results (multi-run mode) |
| `training_plot_*.png` | Training reward curves |

---

## Interpreting Results

**Reward Interpretation:**
- `reward = 1.0`: Target ray is **inside HEC** (exact graph realization found)
- `reward > 0.999`: Target ray is **likely inside HEC** (near-perfect realization)
- `reward < 0.999`: Target ray **may be outside HEC** (further investigation needed)

**Key Metrics:**
- `best_reward`: Maximum reward achieved across all iterations
- `best_optimized_reward`: Best reward after scipy refinement
- `final_weights`: Edge weight configuration achieving best reward

---

## Performance Notes

- **JAX Acceleration**: ~334x speedup over pure NumPy implementation
- **Scipy Optimization**: Additional 42x speedup for local refinement
- **Recommended N values**: N=13-20 for N=6 rays (more vertices = higher chance of finding realization)
- **Memory**: ~2GB GPU memory for N=18, batch_size=120
