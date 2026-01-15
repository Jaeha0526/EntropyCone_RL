"""
QP-based Safe Direction Module for Prototype2
Ensures movement respects known facet constraints while following gradient as closely as possible.
"""

import numpy as np
from scipy.optimize import minimize
import json
from typing import Tuple, Dict, Optional


class QPSafeDirection:
    """
    Computes safe movement direction using Quadratic Programming.
    Only considers known (satisfied) facets, not violated ones.
    """
    
    def __init__(self, known_facets_file: str, saturation_tolerance: float = 1e-4,
                 safety_factor: float = 0.9, verbose: bool = False,
                 dS: float = 0.005, use_escape_mode: bool = False,
                 min_increase_rate: float = 0.05, buffer_factor: float = 1.5,
                 escape_threshold_factor: float = 1.5,
                 use_hybrid_mode: bool = False, hybrid_alpha: float = 0.5):
        """
        Initialize the QP safe direction calculator.

        Args:
            known_facets_file: Path to file containing only known (satisfied) facets
            saturation_tolerance: Tolerance for considering a facet saturated (default 1e-4)
            safety_factor: Fraction of max safe distance to actually move (default 0.9)
            verbose: Whether to print debug information
            dS: Gradient sampling perturbation size (default 0.005)
            use_escape_mode: Enable gradient-aligned escape mode (default False)
            min_increase_rate: Minimum rate for saturated facets to increase (default 0.05)
            buffer_factor: Buffer multiplier for dS (default 1.5) - for distance constraint
            escape_threshold_factor: Escape threshold multiplier for buffer (default 1.5) - for direction constraint
            use_hybrid_mode: Enable hybrid mode mixing escape and max-min directions (default False)
            hybrid_alpha: Mixing ratio for hybrid mode (1.0 = pure escape, 0.0 = pure maxmin, default 0.5)
        """
        self.saturation_tol = saturation_tolerance
        self.safety_factor = safety_factor
        self.verbose = verbose
        self.dS = dS
        self.use_escape_mode = use_escape_mode
        self.min_increase_rate = min_increase_rate
        self.buffer_factor = buffer_factor
        self.buffer = buffer_factor * dS
        self.escape_threshold_factor = escape_threshold_factor
        self.escape_threshold = escape_threshold_factor * self.buffer
        self.use_hybrid_mode = use_hybrid_mode
        self.hybrid_alpha = hybrid_alpha

        # Load known facets
        self.facets = self._load_facets(known_facets_file)
        self.n_facets = len(self.facets)

        if self.verbose:
            print(f"QPSafeDirection initialized with {self.n_facets} known facets")
            print(f"Saturation tolerance: {self.saturation_tol}")
            print(f"Safety factor: {self.safety_factor}")
            if self.use_hybrid_mode:
                print(f"Hybrid mode: ENABLED")
                print(f"  Alpha: {self.hybrid_alpha:.2f} (1.0=pure escape, 0.0=pure maxmin)")
                print(f"  dS: {self.dS}")
                print(f"  Buffer: {self.buffer:.6f} ({self.buffer_factor} × dS)")
                print(f"  Escape threshold: {self.escape_threshold:.6f} ({self.escape_threshold_factor} × buffer)")
                print(f"  Min increase rate: {self.min_increase_rate}")
            elif self.use_escape_mode:
                print(f"Escape mode: ENABLED")
                print(f"  dS: {self.dS}")
                print(f"  Buffer: {self.buffer:.6f} ({self.buffer_factor} × dS)")
                print(f"  Escape threshold: {self.escape_threshold:.6f} ({self.escape_threshold_factor} × buffer)")
                print(f"  Min increase rate: {self.min_increase_rate}")

        # State tracking
        self.current_saturated_indices = []
        self.current_facet_values = None
        self.iteration = 0
        
    def _load_facets(self, facets_file: str) -> np.ndarray:
        """Load facet constraints from file."""
        facets = []
        with open(facets_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    coeffs = [float(x) for x in line.split()]
                    facets.append(coeffs)
        return np.array(facets)
    
    def compute_safe_direction(self, current_point: np.ndarray,
                              gradient: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """
        Main entry point: routes to hybrid, escape, or standard mode.

        Args:
            current_point: Current position (31D)
            gradient: Raw gradient direction (31D)

        Returns:
            direction: QP-adjusted unit direction
            safe_distance: Safe movement distance (already includes safety factor)
            metadata: Dictionary with adjustment details
        """
        if self.use_hybrid_mode:
            return self._try_hybrid_mode(current_point, gradient)
        elif self.use_escape_mode:
            return self._try_escape_mode(current_point, gradient)
        else:
            return self._compute_standard_mode(current_point, gradient)

    def _compute_standard_mode(self, current_point: np.ndarray,
                               gradient: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """
        Original safe direction mode: saturated facets can stay at 0.

        Args:
            current_point: Current position (31D)
            gradient: Raw gradient direction (31D)

        Returns:
            direction: QP-adjusted unit direction
            safe_distance: Safe movement distance (already includes safety factor)
            metadata: Dictionary with adjustment details
        """
        self.iteration += 1
        gradient_norm = gradient / np.linalg.norm(gradient)
        
        # Step 1: Evaluate current position against known facets
        self.current_facet_values = self.facets @ current_point
        
        # Step 2: Find saturated facets (|a·S| < tolerance)
        saturated_mask = np.abs(self.current_facet_values) < self.saturation_tol
        self.current_saturated_indices = np.where(saturated_mask)[0]
        n_saturated = len(self.current_saturated_indices)
        
        # Step 3: Among saturated, find which would be violated by gradient (for logging)
        violating_indices = []
        if n_saturated > 0:
            A_saturated = self.facets[self.current_saturated_indices]
            gradient_dots = A_saturated @ gradient_norm
            violating_mask = gradient_dots < 0
            violating_indices = self.current_saturated_indices[violating_mask]

        n_violating = len(violating_indices)

        if self.verbose:
            print(f"\n[Iteration {self.iteration}] Safe direction computation:")
            print(f"  Saturated facets (|v| < {self.saturation_tol}): {n_saturated}")
            print(f"  Would be violated by gradient: {n_violating}")

        # Step 4: Solve QP if saturated constraints exist
        # IMPORTANT: Constrain ALL saturated facets, not just violating ones
        if n_saturated > 0:
            A_saturated = self.facets[self.current_saturated_indices]
            direction = self._solve_qp(gradient_norm, A_saturated)
            gradient_similarity = np.dot(gradient_norm, direction)
        else:
            # No constraints, use original gradient
            direction = gradient_norm
            gradient_similarity = 1.0
        
        # Step 5: Compute maximum safe distance
        max_safe_distance = self._compute_max_safe_distance(current_point, direction)
        
        # Step 6: Apply safety factor
        safe_distance = self.safety_factor * max_safe_distance
        
        # Prepare metadata
        metadata = {
            'n_known_facets': self.n_facets,
            'n_saturated': n_saturated,
            'n_violating': n_violating,
            'saturated_indices': self.current_saturated_indices.tolist(),
            'gradient_similarity': gradient_similarity,
            'angle_degrees': np.arccos(np.clip(gradient_similarity, -1, 1)) * 180 / np.pi,
            'max_safe_distance_raw': max_safe_distance,
            'safe_distance': safe_distance,
            'safety_factor': self.safety_factor
        }
        
        if self.verbose:
            print(f"  Gradient similarity: {gradient_similarity:.4f}")
            print(f"  Angle from gradient: {metadata['angle_degrees']:.2f}°")
            print(f"  Max safe distance: {max_safe_distance:.6e}")
            print(f"  Actual movement: {safe_distance:.6e}")
        
        return direction, safe_distance, metadata
    
    def _solve_qp(self, gradient: np.ndarray, A_constraints: np.ndarray) -> np.ndarray:
        """
        Solve QP: min ||d - gradient||^2 subject to A @ d >= 0.
        
        Args:
            gradient: Target gradient direction (normalized)
            A_constraints: Matrix of constraint normals
            
        Returns:
            Normalized QP solution
        """
        def objective(d):
            diff = d - gradient
            return np.dot(diff, diff)
        
        def objective_grad(d):
            return 2 * (d - gradient)
        
        # Constraints: A @ d >= 0 for each row of A
        constraints = []
        for facet_normal in A_constraints:
            constraints.append({
                'type': 'ineq',
                'fun': lambda d, a=facet_normal: np.dot(a, d),
                'jac': lambda d, a=facet_normal: a
            })
        
        # Norm constraint: ||d||^2 <= 1
        constraints.append({
            'type': 'ineq',
            'fun': lambda d: 1.0 - np.dot(d, d),
            'jac': lambda d: -2 * d
        })
        
        # Initial guess is the gradient
        x0 = gradient.copy()
        
        # Solve
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=objective_grad,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success and self.verbose:
            print(f"  Warning: QP optimization did not fully converge: {result.message}")
        
        # Normalize the result
        direction = result.x / np.linalg.norm(result.x)
        return direction
    
    def _compute_max_safe_distance(self, current_point: np.ndarray, 
                                  direction: np.ndarray) -> float:
        """
        Compute maximum distance before any known facet would be violated.
        
        Args:
            current_point: Current position
            direction: Movement direction (normalized)
            
        Returns:
            Maximum safe distance
        """
        # How each facet changes along direction
        facet_rates = self.facets @ direction
        
        max_distance = float('inf')
        critical_facet = None
        
        for i, (value, rate) in enumerate(zip(self.current_facet_values, facet_rates)):
            if rate < -1e-10:  # Facet would decrease
                if value < 1e-15:  # Already at boundary
                    max_distance = 0.0
                    critical_facet = i
                    break
                else:
                    # Distance until this facet hits zero
                    dist = -value / rate
                    if dist < max_distance:
                        max_distance = dist
                        critical_facet = i
        
        if self.verbose and critical_facet is not None:
            print(f"  Critical facet: {critical_facet} (value={self.current_facet_values[critical_facet]:.6e})")

        return max_distance

    def _try_escape_mode(self, current_point: np.ndarray,
                        gradient: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """
        Try escape mode with fallback to standard mode if it fails.

        Args:
            current_point: Current position
            gradient: Raw gradient direction

        Returns:
            direction, safe_distance, metadata (with fallback tracking)
        """
        try:
            direction, safe_distance, metadata = self._compute_gradient_aligned_escape(
                current_point, gradient
            )

            # Check if optimization was successful
            if metadata.get('optimization_success', False):
                # Check if we achieved sufficient increase
                min_achieved = metadata.get('min_facet_increase', 0)
                if min_achieved >= self.min_increase_rate * 0.8:  # Allow 20% tolerance
                    # Success!
                    metadata['escape_mode_successful'] = True
                    metadata['fallback_occurred'] = False
                    return direction, safe_distance, metadata
                else:
                    reason = f"insufficient_increase (achieved {min_achieved:.4f}, required {self.min_increase_rate:.4f})"
                    if self.verbose:
                        print(f"  ⚠ FALLBACK: {reason}")
                    return self._fallback_to_standard(current_point, gradient, reason)
            else:
                reason = "optimization_failed"
                if self.verbose:
                    print(f"  ⚠ FALLBACK: {reason}")
                return self._fallback_to_standard(current_point, gradient, reason)

        except Exception as e:
            reason = f"exception: {str(e)}"
            if self.verbose:
                print(f"  ⚠ FALLBACK: {reason}")
            return self._fallback_to_standard(current_point, gradient, reason)

    def _fallback_to_standard(self, current_point: np.ndarray,
                             gradient: np.ndarray, reason: str) -> Tuple[np.ndarray, float, Dict]:
        """
        Fallback to standard mode with metadata tracking.

        Args:
            current_point: Current position
            gradient: Raw gradient
            reason: Reason for fallback

        Returns:
            direction, safe_distance, metadata (standard mode with fallback info)
        """
        direction, safe_distance, metadata = self._compute_standard_mode(current_point, gradient)
        metadata['escape_mode_attempted'] = True
        metadata['escape_mode_successful'] = False
        metadata['fallback_occurred'] = True
        metadata['fallback_reason'] = reason
        return direction, safe_distance, metadata

    def _try_hybrid_mode(self, current_point: np.ndarray,
                        gradient: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """
        Try hybrid mode: mix escape QP and max-min QP directions.

        NEW ORDER (adaptive):
        1. Compute max-min QP first (if fails → standard mode)
        2. Use maxmin_value to calculate adaptive escape constraint
        3. Compute escape QP with adaptive constraint (if fails → standard mode)
        4. Mix directions

        Args:
            current_point: Current position
            gradient: Raw gradient direction

        Returns:
            direction, safe_distance, metadata (with hybrid tracking)
        """
        gradient_norm = gradient / np.linalg.norm(gradient)

        # Evaluate current position (needed for both escape and max-min)
        self.current_facet_values = self.facets @ current_point

        # LEVEL 1: Compute max-min direction FIRST
        try:
            d_maxmin, maxmin_value, maxmin_success = self._compute_maxmin_direction(current_point)

            if not maxmin_success:
                # Max-min QP failed, fall back to standard mode
                reason = "maxmin_qp_failed"
                if self.verbose:
                    print(f"  ⚠ HYBRID FALLBACK: {reason}")
                return self._fallback_to_standard(current_point, gradient, reason)

        except Exception as e:
            reason = f"maxmin_exception: {str(e)}"
            if self.verbose:
                print(f"  ⚠ HYBRID FALLBACK: {reason}")
            return self._fallback_to_standard(current_point, gradient, reason)

        # LEVEL 2: Calculate adaptive escape constraint based on maxmin_value
        if maxmin_value > self.min_increase_rate:
            # Geometry is good - maxmin achieves MORE than config asks
            # Use the LARGER of: (half of maxmin, config setting)
            escape_min_increase = max(0.5 * maxmin_value, self.min_increase_rate)
        else:
            # Geometry is constrained - maxmin achieves LESS than config asks
            # Config setting would be impossible, so adapt down
            escape_min_increase = 0.5 * maxmin_value

        if self.verbose:
            print(f"  Adaptive constraint: maxmin_value={maxmin_value:.6f}, config={self.min_increase_rate:.6f} → escape_min_increase={escape_min_increase:.6f}")

        # LEVEL 2: Compute escape direction with adaptive constraint
        try:
            d_escape, _, escape_metadata = self._compute_gradient_aligned_escape(
                current_point, gradient, min_increase_rate=escape_min_increase
            )

            if not escape_metadata.get('optimization_success', False):
                # Escape QP failed, fall back to standard mode
                reason = "escape_qp_failed"
                if self.verbose:
                    print(f"  ⚠ HYBRID FALLBACK: {reason}")
                return self._fallback_to_standard(current_point, gradient, reason)

        except Exception as e:
            reason = f"escape_exception: {str(e)}"
            if self.verbose:
                print(f"  ⚠ HYBRID FALLBACK: {reason}")
            return self._fallback_to_standard(current_point, gradient, reason)

        # LEVEL 3: Mix directions
        d_hybrid = self._compute_hybrid_direction(d_escape, d_maxmin, self.hybrid_alpha)

        # Step 4: Compute safe distance for hybrid direction
        max_safe_distance = self._compute_escape_distance(current_point, d_hybrid)
        safe_distance = self.safety_factor * max_safe_distance

        # Step 5: Analyze hybrid direction
        below_threshold_indices = np.where(self.current_facet_values < self.escape_threshold)[0]
        A_below = self.facets[below_threshold_indices]

        # Compute minimum increases for all directions
        escape_increases = A_below @ d_escape
        escape_min_increase_achieved = np.min(escape_increases) if len(escape_increases) > 0 else np.inf

        maxmin_increases = A_below @ d_maxmin
        maxmin_measured_increase = np.min(maxmin_increases) if len(maxmin_increases) > 0 else np.inf

        hybrid_increases = A_below @ d_hybrid
        hybrid_min_increase = np.min(hybrid_increases) if len(hybrid_increases) > 0 else np.inf

        # Gradient similarities
        escape_grad_sim = np.dot(d_escape, gradient_norm)
        maxmin_grad_sim = np.dot(d_maxmin, gradient_norm)
        hybrid_grad_sim = np.dot(d_hybrid, gradient_norm)

        if self.verbose:
            print(f"  Hybrid composition:")
            print(f"    MaxMin: grad_sim={maxmin_grad_sim:.4f}, min_inc={maxmin_measured_increase:.6f} (max achievable)")
            print(f"    Escape: grad_sim={escape_grad_sim:.4f}, min_inc={escape_min_increase_achieved:.6f} (asked for {escape_min_increase:.6f})")
            print(f"    Hybrid: grad_sim={hybrid_grad_sim:.4f}, min_inc={hybrid_min_increase:.6f}, alpha={self.hybrid_alpha:.2f}")

        # Build comprehensive metadata
        metadata = {
            'hybrid_mode_successful': True,
            'hybrid_mode_attempted': True,
            'hybrid_alpha': self.hybrid_alpha,
            'n_below_threshold': len(below_threshold_indices),
            'below_threshold_indices': below_threshold_indices.tolist(),

            # Max-min direction stats (computed first)
            'maxmin_value': float(maxmin_value),  # What max-min QP reported (maximum achievable)
            'maxmin_gradient_similarity': float(maxmin_grad_sim),
            'maxmin_min_increase': float(maxmin_measured_increase),

            # Escape direction stats (with adaptive constraint)
            'escape_min_increase_used': float(escape_min_increase),  # What we asked escape for
            'escape_gradient_similarity': float(escape_grad_sim),
            'escape_min_increase': float(escape_min_increase_achieved),  # What escape achieved
            'maxmin_qp_optimal': float(maxmin_value),  # What max-min QP reported

            # Hybrid direction stats
            'gradient_similarity': float(hybrid_grad_sim),  # For compatibility
            'hybrid_gradient_similarity': float(hybrid_grad_sim),
            'hybrid_min_increase': float(hybrid_min_increase),

            # Distance info
            'max_safe_distance_raw': max_safe_distance,
            'safe_distance': safe_distance,
            'optimization_success': True
        }

        return d_hybrid, safe_distance, metadata

    def _compute_gradient_aligned_escape(self, current_point: np.ndarray,
                                        gradient: np.ndarray,
                                        min_increase_rate: float = None) -> Tuple[np.ndarray, float, Dict]:
        """
        Escape mode: maximize gradient alignment while forcing facets below threshold to increase.

        Args:
            current_point: Current position
            gradient: Raw gradient direction
            min_increase_rate: Minimum increase rate (if None, uses self.min_increase_rate)

        Returns:
            direction, safe_distance, metadata
        """
        # Use provided min_increase_rate or default to self.min_increase_rate
        if min_increase_rate is None:
            min_increase_rate = self.min_increase_rate
        self.iteration += 1
        gradient_norm = gradient / np.linalg.norm(gradient)

        # Evaluate current position
        self.current_facet_values = self.facets @ current_point

        # Find facets below escape threshold (for direction constraints)
        below_threshold_mask = self.current_facet_values < self.escape_threshold
        below_threshold_indices = np.where(below_threshold_mask)[0]
        n_below_threshold = len(below_threshold_indices)

        if self.verbose:
            print(f"\n[Iteration {self.iteration}] Escape mode computation:")
            print(f"  Facets below escape threshold ({self.escape_threshold:.6f}): {n_below_threshold}")

        if n_below_threshold == 0:
            # No facets below threshold - use gradient directly
            if self.verbose:
                print(f"  No facets below threshold - using gradient directly")
            max_safe_distance = self._compute_escape_distance(current_point, gradient_norm)
            safe_distance = self.safety_factor * max_safe_distance

            metadata = {
                'n_below_threshold': 0,
                'gradient_similarity': 1.0,
                'optimization_success': True,
                'min_facet_increase': np.inf,
                'max_safe_distance_raw': max_safe_distance,
                'safe_distance': safe_distance,
                'escape_mode_successful': True
            }
            return gradient_norm, safe_distance, metadata

        # Solve QP: maximize gradient alignment subject to min_increase constraints
        A_below_threshold = self.facets[below_threshold_indices]

        def objective(d):
            # Minimize negative alignment
            return -np.dot(d, gradient_norm)

        def objective_grad(d):
            return -gradient_norm

        # Constraints
        constraints = []

        # Each facet below threshold must increase by at least min_increase_rate
        for facet_normal in A_below_threshold:
            constraints.append({
                'type': 'ineq',
                'fun': lambda d, a=facet_normal, mir=min_increase_rate: np.dot(a, d) - mir,
                'jac': lambda d, a=facet_normal: a
            })

        # Unit norm constraint
        constraints.append({
            'type': 'eq',
            'fun': lambda d: np.linalg.norm(d) - 1.0,
            'jac': lambda d: d / np.linalg.norm(d)
        })

        # Solve
        x0 = gradient_norm.copy()
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=objective_grad,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
        )

        if result.success:
            direction = result.x / np.linalg.norm(result.x)
            gradient_similarity = np.dot(direction, gradient_norm)
            facet_dots = A_below_threshold @ direction
            min_facet_increase = np.min(facet_dots)

            # Compute safe distance with buffer
            max_safe_distance = self._compute_escape_distance(current_point, direction)
            safe_distance = self.safety_factor * max_safe_distance

            if self.verbose:
                print(f"  Gradient alignment: {gradient_similarity:.4f}")
                print(f"  Min facet increase: {min_facet_increase:.4f}")
                print(f"  Max safe distance: {max_safe_distance:.6e}")

            metadata = {
                'n_below_threshold': n_below_threshold,
                'below_threshold_indices': below_threshold_indices.tolist(),
                'gradient_similarity': gradient_similarity,
                'angle_degrees': np.arccos(np.clip(gradient_similarity, -1, 1)) * 180 / np.pi,
                'min_facet_increase': min_facet_increase,
                'max_safe_distance_raw': max_safe_distance,
                'safe_distance': safe_distance,
                'optimization_success': True,
                'escape_mode_successful': True
            }

            return direction, safe_distance, metadata
        else:
            # Optimization failed
            metadata = {
                'n_below_threshold': n_below_threshold,
                'optimization_success': False,
                'optimization_message': result.message
            }
            return gradient_norm, 0.0, metadata

    def _compute_maxmin_direction(self, current_point: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Compute direction that maximizes minimum facet increase (ignoring gradient).

        Solves:
            MAXIMIZE  t
            s.t.      facet[i] · d ≥ t  (for all saturated facets)
                      ||d|| = 1

        Args:
            current_point: Current position (assumes current_facet_values already computed)

        Returns:
            direction: Optimal direction (unit vector)
            min_increase: Maximum achievable minimum increase (t*)
            success: Whether optimization succeeded
        """
        # Find facets below escape threshold (same set as escape mode)
        below_threshold_mask = self.current_facet_values < self.escape_threshold
        below_threshold_indices = np.where(below_threshold_mask)[0]

        if len(below_threshold_indices) == 0:
            # No constraints - should not happen but handle gracefully
            return np.ones(len(current_point)) / np.sqrt(len(current_point)), np.inf, False

        saturated_facets = self.facets[below_threshold_indices]
        n_dims = len(current_point)

        # Optimize over (d, t) where d is direction, t is minimum increase
        # Variables: x = [d (n_dims), t (1)]
        def objective(x):
            t = x[-1]
            return -t  # Maximize t = minimize -t

        def objective_grad(x):
            grad = np.zeros(n_dims + 1)
            grad[-1] = -1  # derivative w.r.t. t
            return grad

        # Constraints
        constraints = []

        # For each saturated facet: facet · d ≥ t
        for facet in saturated_facets:
            def make_constraint(a):
                return {
                    'type': 'ineq',
                    'fun': lambda x, a=a: np.dot(a, x[:n_dims]) - x[-1],  # a·d - t ≥ 0
                    'jac': lambda x, a=a: np.concatenate([a, [-1]])
                }
            constraints.append(make_constraint(facet))

        # Unit norm: ||d|| = 1
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.linalg.norm(x[:n_dims]) - 1.0,
            'jac': lambda x: np.concatenate([x[:n_dims] / np.linalg.norm(x[:n_dims]), [0]])
        })

        # Initial guess: random direction + small t
        x0 = np.random.randn(n_dims)
        x0 = x0 / np.linalg.norm(x0)
        x0 = np.concatenate([x0, [0.01]])

        # Solve
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=objective_grad,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
        )

        if result.success:
            direction = result.x[:n_dims]
            direction = direction / np.linalg.norm(direction)  # Ensure unit norm
            min_increase = result.x[-1]

            if self.verbose:
                print(f"  Max-Min QP: min_increase = {min_increase:.6f}")

            return direction, min_increase, True
        else:
            if self.verbose:
                print(f"  Max-Min QP FAILED: {result.message}")
            return None, None, False

    def _compute_hybrid_direction(self, d_escape: np.ndarray, d_maxmin: np.ndarray,
                                   alpha: float) -> np.ndarray:
        """
        Compute hybrid direction mixing escape and max-min directions.

        d_hybrid = α × d_escape + (1-α) × d_maxmin

        Args:
            d_escape: Escape QP direction (gradient-aligned)
            d_maxmin: Max-Min QP direction (maximum escape)
            alpha: Mixing ratio (1.0 = pure escape, 0.0 = pure maxmin)

        Returns:
            Normalized hybrid direction
        """
        d_hybrid = alpha * d_escape + (1 - alpha) * d_maxmin
        d_hybrid = d_hybrid / np.linalg.norm(d_hybrid)  # Renormalize
        return d_hybrid

    def _compute_escape_distance(self, current_point: np.ndarray,
                                 direction: np.ndarray) -> float:
        """
        Compute maximum safe distance with buffer constraint.
        Loose facets must stay above buffer (1.5*dS).

        Args:
            current_point: Current position
            direction: Movement direction (normalized)

        Returns:
            Maximum safe distance
        """
        facet_rates = self.facets @ direction
        max_distance = float('inf')
        critical_facet = None

        for i, (value, rate) in enumerate(zip(self.current_facet_values, facet_rates)):
            if rate < -1e-10:  # Facet would decrease
                if value <= self.buffer + 1e-15:
                    # Already at or below buffer - can't move
                    max_distance = 0.0
                    critical_facet = i
                    break
                else:
                    # Distance until hitting buffer
                    dist = (value - self.buffer) / (-rate)
                    if dist < max_distance:
                        max_distance = dist
                        critical_facet = i

        if self.verbose and critical_facet is not None:
            print(f"  Critical facet: {critical_facet} (value={self.current_facet_values[critical_facet]:.6e}, buffer={self.buffer:.6e})")

        return max_distance

    def get_facet_status(self, point: np.ndarray) -> Dict:
        """
        Get detailed status of facets at given point.
        
        Args:
            point: Position to evaluate
            
        Returns:
            Dictionary with facet statistics
        """
        values = self.facets @ point
        
        return {
            'n_total': self.n_facets,
            'n_saturated': np.sum(np.abs(values) < self.saturation_tol),
            'n_nearly_saturated': np.sum(np.abs(values) < 1e-3),
            'n_positive': np.sum(values > self.saturation_tol),
            'min_value': np.min(values),
            'max_value': np.max(values),
            'saturated_indices': np.where(np.abs(values) < self.saturation_tol)[0].tolist()
        }
    
    def reset(self):
        """Reset internal state tracking."""
        self.current_saturated_indices = []
        self.current_facet_values = None
        self.iteration = 0


def test_qp_safe_direction():
    """Test the QPSafeDirection class with ray8 data."""
    print("Testing QPSafeDirection with ray8...")
    
    # Initialize
    known_facets_file = "extremal_rays/n5/experiment4_ray8/ray8_known_facets.txt"
    qp_director = QPSafeDirection(known_facets_file, verbose=True)
    
    # Load a test point and gradient
    import os
    test_file = "results/prototype2_experiment4_ray8_v01_rl2_dS001_samples400/ray8_7/experiment_results.json"
    
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        test_point = np.array(data[0]['start_points'])
        
        # Load gradient
        grad_file = test_file.replace('experiment_results.json', 'gradient_samples_stage_0.json')
        if os.path.exists(grad_file):
            with open(grad_file, 'r') as f:
                grad_data = json.load(f)
            test_gradient = np.array(grad_data['gradient_computed'])
            
            # Test computation
            direction, safe_distance, metadata = qp_director.compute_safe_direction(
                test_point, test_gradient
            )
            
            print(f"\nTest results:")
            print(f"  Direction norm: {np.linalg.norm(direction):.6f}")
            print(f"  Safe distance: {safe_distance:.6e}")
            print(f"  Metadata: {json.dumps(metadata, indent=2)}")
            
            # Test movement
            new_point = test_point + safe_distance * direction
            new_status = qp_director.get_facet_status(new_point)
            print(f"\nAfter movement:")
            print(f"  Min facet value: {new_status['min_value']:.6e}")
            print(f"  Saturated facets: {new_status['n_saturated']}")
        else:
            print(f"Gradient file not found: {grad_file}")
    else:
        print(f"Test file not found: {test_file}")
    
    print("\nQPSafeDirection module created successfully!")


if __name__ == "__main__":
    test_qp_safe_direction()
