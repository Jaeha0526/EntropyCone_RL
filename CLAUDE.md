# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project Overview

Reinforcement learning for exploring the **Holographic Entropy Cone (HEC)**.
Reproduction code for He, Lee & Ooguri, *"Exploring the Holographic Entropy
Cone via Machine Learning"* (Caltech).

Given a target entropy vector, a policy searches for a **graph realization**
whose min-cut entropies match it. Reward = cosine similarity between achieved
and target entropy vectors. That single reward does double duty:

- **Classification** — reward = 1 ⟹ the target lies inside the HEC; reward < 1 ⟹ outside.
- **Navigation** — ∇reward points toward the HEC boundary, which is how unknown
  facets (holographic entropy inequalities) get discovered.

**This repo backs a published paper.** Results under `results/` are paper
artifacts. Do not regenerate or overwrite them casually — reproduce into a new
`output_dir` and compare.

## Setup

```bash
pip install -r requirements.txt
```

## Commands

Everything is **config-driven**: one JSON file per experiment under `configs/`,
passed with `--config`. Prefer adding a config over adding CLI flags.

```bash
# Section 4 — N=6 SA cone ray classification
python src/HECenv_parallel_config.py --config configs/n6_sa_cone_classification/template_N18.json
python analysis/n6/analyze_across_N.py

# Section 3.4 — N=3 MMI rediscovery (gradient following)
python src/prototype2_jax_config.py --config configs/n3_mmi_finding/symmetric_N5.json

# Section 3.3 — N=3 grid validation
python analysis/n3/analyze_grid_validation.py
python analysis/n3/plot_rl_reward_landscape.py

# Appendix B — gradient estimation quality
python src/prototype2_jax_config.py --config configs/n3_mmi_finding/gradient_quality/dS_analyze/dS0010_max10.json
python analysis/n3/plot_dS_analysis_combined.py
python analysis/n3/plot_samples_vs_alignment.py
```

Per-runner detail lives in `src/README_HECenv_parallel_config.md` and
`src/README_prototype2_jax_config.md`.

## Config anatomy

Three blocks: `experiment` (name, `output_dir`, `target_ray_index`,
`target_ray_source`), `training` (`n`, `N`, `batch_size`, `lr`, `n_iter`,
`rollout_length`, `multi_run`/`num_runs`, early stopping), `environment`
(`device`, `seed`, `num_workers`, `precision`).

- `n` = number of parties (6 for the SA cone work); `N` = graph size. They are
  different knobs and both appear in `training` — don't conflate them.
- `precision: "float64"` is deliberate. Classification turns on distinguishing
  reward 0.9998 from 1.0; float32 destroys that margin.
- `template_N18.json` uses literal `PLACEHOLDER` for the ray index — it is a
  template to be substituted per ray, not a runnable config.

## S-vector indexing — the recurring bug

The S-vector uses **binary subset indexing**: index `i` is the subset in which
party `j` is present iff bit `j` is set in `i+1`.

For n=3 the ordering is `[S_A, S_B, S_AB, S_C, S_AC, S_BC, S_ABC]`.

⚠️ **`S_AB` (index 2) comes BEFORE `S_C` (index 3)**, because binary `011 < 100`.
This is the single most common mistake when reading or constructing entropy
vectors here. Sizes: n=2→3, n=3→7, n=4→15, n=5→31, n=6→63 components.
Reference implementation: `HECenv_parallel.py:Sfromw_single()`.

## Layout

```
src/HECenv_parallel.py          Main parallel RL environment
src/HECenv_parallel_config.py   Config-driven classification runner (Section 4)
src/prototype2_jax_config.py    Config-driven gradient-following runner (Section 3.4)
src/qp_safe_direction.py        QP solver for safe movement along the boundary
src/jax_optimization/           JAX acceleration (334x speedup on S-vector computation)
data/n3/, data/n6/              Ray data: SA cone rays, MMI facets, mystery rays
configs/                        One JSON per paper experiment
results/                        PAPER ARTIFACTS — treat as read-only
analysis/                       Plotting + cross-N analysis scripts
```

## Interpreting results

A ray is only claimed "in HEC" on a **near-exact** realization (reward
≥ 0.99999999, ideally an exact integer weight assignment). Rays 146/180/181
cleared that bar; 110/145/168 plateau at ~0.9995–0.9998 across N and are
reported as *evidence against* realizability — deliberately weaker language,
since a failed search is not a proof. Preserve that distinction in any
write-up.
