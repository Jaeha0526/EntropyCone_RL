# Ray 180 Realization Analysis (N=14)

## Summary

SA cone ray 180 can be realized with a simple integer weight vector using only two values: **12** and **7**.

The resulting entropy vector is exactly **12 × ray_180**, achieving perfect cosine similarity of 1.0.

## Optimal Weights (from results_run_4.json)

**Best reward:** 0.9999999999999186 (essentially 1.0)

### Original w_optimal (continuous values)

```
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9667543241000572, 0.0, 0.9667543240562605,
0.0, 0.0, 0.0, 0.0, 0.0, 0.9667542017244684, 0.0, 0.9667543190354092, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9667495462491312, 0.0, 0.966754324098566, 0.0, 0.0,
0.0, 0.0, 0.9667542597834704, 0.0, 0.0, 0.0, 0.9667543241000572, 0.9667543241000572, 0.0,
0.0, 0.0, 0.0, 0.0, 0.9667543241000571, 0.0, 0.0, 0.9667543241000572, 0.9667543241000572,
0.0, 0.0, 0.0, 0.9667542920422939, 0.9667543241000572, 0.9667542096260893, 0.0, 0.0, 0.0,
0.565050947372967, 0.0, 0.9667543114402783, 0.0, 0.0, 0.0, 0.9667543033469345, 0.0,
0.9667543240993801, 0.0, 0.0, 0.0, 0.9667508249984973, 0.9667543241000277, 0.0, 0.0, 0.0,
0.0, 0.0, 0.9667543241000572, 0.0, 0.0, 0.9667543241000572, 0.0, 0.0, 0.96675432057596,
0.0, 0.9667543241000572]
```

**Key values:**
- High weight: ~0.9667543241 (24 occurrences)
- Low weight: ~0.5650509474 (1 occurrence at position 63)
- Zero weights: 66 occurrences

**Ratio:** 0.9667543241 / 0.5650509474 = 1.7109153229

## Integer Approximation

**Simple fraction:** 12/7 = 1.7142857143 (error: 0.0034)

### Integerized w_optimal

Scaling the weights to integer ratio 12:7:

```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 12, 0, 0, 0, 0, 0, 12, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 12, 0, 12, 0, 0, 0, 0, 12, 0, 0, 0, 12, 12, 0, 0, 0, 0, 0, 12, 0, 0, 12, 12, 0,
0, 0, 12, 12, 12, 0, 0, 0, 7, 0, 12, 0, 0, 0, 12, 0, 12, 0, 0, 0, 12, 12, 0, 0, 0, 0, 0,
12, 0, 0, 12, 0, 0, 12, 0, 12]
```

**Weight distribution:**
- Value 12: 24 positions
- Value 7: 1 position (index 63)
- Value 0: 66 positions

## Entropy Vector from Integerized Weights

Computing S = Sfromw_single(w_integerized, n=6, N=14):

```
[24, 24, 48, 24, 48, 48, 72, 36, 60, 60, 84, 60, 84, 84, 108, 36, 60, 60, 84, 60, 84, 84,
108, 72, 72, 96, 96, 96, 72, 96, 72, 36, 60, 60, 84, 60, 84, 84, 108, 72, 72, 96, 96, 96,
96, 72, 72, 72, 96, 72, 96, 96, 72, 96, 72, 108, 84, 84, 60, 84, 60, 60, 36]
```

## Target Ray 180

```
[2, 2, 4, 2, 4, 4, 6, 3, 5, 5, 7, 5, 7, 7, 9, 3, 5, 5, 7, 5, 7, 7, 9, 6, 6, 8, 8, 8, 6,
8, 6, 3, 5, 5, 7, 5, 7, 7, 9, 6, 6, 8, 8, 8, 8, 6, 6, 6, 8, 6, 8, 8, 6, 8, 6, 9, 7, 7, 5,
7, 5, 5, 3]
```

## Key Result

**Perfect scaling relationship:**

```
S_vector = 12 × target_ray_180
```

**All 63 components satisfy:**
- S[i] / target[i] = 12.0 (exactly)

**Cosine similarity:** 1.000000000000000 (perfect alignment)

## Interpretation

1. **The integerized weights (12, 7, 0) perfectly represent ray 180**
   - The 12:7 ratio captures the optimal weight structure
   - Only two non-zero weight values are needed

2. **Special constraint at position 63**
   - This is the only constraint with weight 7
   - All other active constraints have weight 12
   - This creates the specific pattern needed for ray 180

3. **Perfect realization**
   - The S vector is exactly 12 times the target ray
   - This confirms ray 180 is perfectly achievable in the HEC at N=14
   - The direction is exact; only the magnitude differs by the factor of 12

## Comparison: Original vs Integerized Weights

**Original weights (continuous):**
- S ≈ 0.967 × target_ray_180
- Cosine similarity: 0.999999999999919
- Point is slightly inside the cone

**Integerized weights (12:7 ratio):**
- S = 12.0 × target_ray_180 (exact)
- Cosine similarity: 1.000000000000000
- Perfect direction alignment

The integerization simplifies the weights while maintaining perfect direction alignment with the target ray.
