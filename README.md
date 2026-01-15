# EntropyConeRL

Reinforcement Learning for Exploring the Holographic Entropy Cone

**Paper**: "Exploring the Holographic Entropy Cone via Machine Learning"
**Authors**: Temple He, Jaeha Lee, Hirosi Ooguri (Caltech)

## Overview

This repository implements a reinforcement learning algorithm to study the Holographic Entropy Cone (HEC). Given a target entropy vector, the algorithm searches for a graph realization whose min-cut entropies match the target, with a reward defined as the cosine similarity between achieved and target entropy vectors.

**Key Capabilities:**
- **Classification**: Reward = 1 indicates the target lies inside the HEC; reward < 1 indicates it's outside
- **Navigation**: Gradient of the reward points toward the HEC boundary, potentially revealing unknown facets

## Paper Results

### Section 3.3: N=3 Grid Validation
- Validates RL reward landscape against analytical predictions
- 324 grid points on the symmetric (s,t) slice
- Pearson correlation 0.996 between RL and analytical rewards

### Section 3.4: N=3 MMI Rediscovery
- Gradient-following from symmetric point toward MMI boundary
- Demonstrates the algorithm can discover unknown holographic entropy inequalities

### Section 4: N=6 SA Cone Classification
- Classifies 208 SA cone extremal rays
- **Found graph realizations** for mystery rays 146, 180, 181 (confirming they're in HEC)
- **Evidence against realizability** for rays 110, 145, 168 (suggesting unknown HEIs exist)

## Directory Structure

```
EntropyConeRL/
├── src/                          # Core source code
│   ├── HECenv_parallel.py        # Main parallel RL environment
│   ├── HECenv_parallel_config.py # Config-based classification runner
│   ├── prototype2_jax_config.py  # Gradient-following runner
│   ├── qp_safe_direction.py      # QP solver for safe movement
│   └── jax_optimization/         # JAX acceleration modules
├── data/                         # Ray data files
│   ├── n3/                       # N=3 SA cone rays & MMI facets
│   └── n6/                       # 208 SA cone rays, mystery rays
├── configs/                      # Experiment configurations
│   ├── n3_grid_validation/       # Section 3.3 configs
│   ├── n3_mmi_finding/           # Section 3.4 configs
│   └── n6_sa_cone_classification/# Section 4 configs
├── results/                      # Paper results
│   ├── n3_grid_validation/       # Grid validation results & plots
│   ├── n3_mmi_finding/           # MMI trajectory results
│   └── n6_sa_cone/               # Classification & realizations
└── analysis/                     # Analysis scripts
    ├── n3/                       # N=3 plotting scripts
    └── n6/                       # N=6 analysis scripts
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### N=6 SA Cone Ray Classification (Section 4)

```bash
# Classify a single ray
python src/HECenv_parallel_config.py --config configs/n6_sa_cone_classification/template_N18.json

# Run analysis across all N values
python analysis/n6/analyze_across_N.py
```

📖 **Detailed documentation**: [src/README_HECenv_parallel_config.md](src/README_HECenv_parallel_config.md)

### N=3 MMI Finding (Section 3.4)

```bash
# Run gradient-following from symmetric point
python src/prototype2_jax_config.py --config configs/n3_mmi_finding/symmetric_N5.json
```

📖 **Detailed documentation**: [src/README_prototype2_jax_config.md](src/README_prototype2_jax_config.md)

### N=3 Grid Validation (Section 3.3)

```bash
# Analyze grid validation results
python analysis/n3/analyze_grid_validation.py

# Generate reward landscape plots
python analysis/n3/plot_rl_reward_landscape.py
```

## Key Algorithm

The RL algorithm uses:
- **Policy Gradient**: Neural network learns to update graph edge weights
- **Reward Function**: Cosine similarity between target and achieved entropy vectors
- **JAX Acceleration**: 334x speedup for S-vector computation

The reward function serves dual purposes:
1. **Classification**: r = 1 means inside HEC, r < 1 means outside
2. **Navigation**: ∇r points toward nearest HEC boundary

## Mystery Rays Results

| Ray | Status | N | Evidence |
|-----|--------|---|----------|
| 146 | In HEC | 18 | Graph realization found (reward=0.9997) |
| 180 | In HEC | 14 | **Perfect integer realization** (12:7 weights) |
| 181 | In HEC | 18 | **Perfect integer realization** (9,6,5,4,0 weights) |
| 110 | Likely outside | - | max_reward = 0.9998, surrounded by non-HEC rays |
| 145 | Likely outside | - | max_reward = 0.9996, surrounded by non-HEC rays |
| 168 | Likely outside | - | max_reward = 0.9995, surrounded by non-HEC rays |

📖 **Detailed realization analysis**: [results/n6_sa_cone/realizations/](results/n6_sa_cone/realizations/)

## Citation

```bibtex
@article{He:2025entropy,
  title={Exploring the Holographic Entropy Cone via Machine Learning},
  author={He, Temple and Lee, Jaeha and Ooguri, Hirosi},
  year={2025}
}
```

## License

This project is for academic research purposes.
