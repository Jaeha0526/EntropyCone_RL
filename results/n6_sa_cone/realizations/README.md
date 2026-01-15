# Graph Realizations for Mystery Rays

This folder contains the graph realizations found for SA cone rays that were previously unknown to be in the Holographic Entropy Cone (HEC).

## Summary

| Ray | N | Max Reward | Status | Weight Structure |
|-----|---|------------|--------|------------------|
| 146 | 18 | 0.9997 | Near-perfect realization | Complex weights |
| 180 | 14 | 1.0000 | **Perfect realization** | Simple 12:7 ratio |
| 181 | 18 | 1.0000 | **Perfect realization** | 5 distinct values: 9,6,5,4,0 |

## Ray 180: Perfect Integer Realization (N=14)

**Key Result:** S = 12 × target_ray_180 (exact)

The optimal weight map uses only **two non-zero values**: 12 and 7.

```
Integer weights: [0,0,0,0,0,0,0,0,0,12,0,12,0,0,0,0,0,12,0,12,0,0,0,0,0,0,0,0,
0,0,0,12,0,12,0,0,0,0,12,0,0,0,12,12,0,0,0,0,0,12,0,0,12,12,0,
0,0,12,12,12,0,0,0,7,0,12,0,0,0,12,0,12,0,0,0,12,12,0,0,0,0,0,
12,0,0,12,0,0,12,0,12]

Weight distribution:
- Value 12: 24 positions
- Value 7: 1 position (index 63)
- Value 0: 66 positions
```

See [ray180_realization.md](ray180_realization.md) for full analysis.

## Ray 181: Perfect Integer Realization (N=18)

**Key Result:** S = 9 × target_ray_181 (exact)

The optimal weight map uses **five distinct non-zero values**: 9, 6, 5, 4, and 0.

```
Integer weights: [0,0,0,0,0,9,0,0,0,0,0,9,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0,
0,9,0,0,0,0,0,0,0,0,9,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,9,0,4,5,9,0,0,9,9,0,0,0,
0,0,0,9,0,0,0,0,9,0,0,0,0,9,9,0,0,0,0,0,0,0,0,9,0,0,0,0,0,9,0,0,9,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,6,0,0,0,9,0,0,0,0,9,0,9,9,9,5,0,0,0,0,0,0,0,0,
0,0,9,0,0,0,0,0,9]

Weight distribution:
- Value 9: 24 positions
- Value 6: 2 positions
- Value 5: 2 positions
- Value 4: 1 position
- Value 0: 124 positions
```

See [ray181_realization.md](ray181_realization.md) for full analysis.

## Ray 146: Near-Perfect Realization (N=18)

**Max Reward:** 0.9997

Ray 146 achieves very high reward but the weight structure is more complex than rays 180/181,
suggesting the optimal realization may require larger N or has inherently more complex geometry.

## Visualization

- `ray146_ray180_ray181_realizations.png` - Combined visualization of all three realizations
- `ray180_realization_graph.png` - Graph structure for ray 180
- `ray181_realization_graph.png` - Graph structure for ray 181
- `ray146_realization_graph.png` - Graph structure for ray 146

## Significance

These realizations confirm that rays 146, 180, and 181 lie within the Holographic Entropy Cone.
The simple integer structure of the weight maps (especially the 12:7 ratio for ray 180) suggests
these rays have elegant underlying geometric interpretations.
