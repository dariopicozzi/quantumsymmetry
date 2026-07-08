"""Fubini-Study metric and polyspherical/Cartesian coordinate maps.

Provides the pruned metric, its inverse, and the coordinate transforms
between tree parameters and statevector amplitudes used by imaginary- and
real-time evolution.
"""

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from functools import partial
from dataclasses import dataclass
from typing import List, Optional

from .tree import *
from .affine import *


__all__ = [
    'get_term',
    'polyspherical_metric_term_pruned',
    'polyspherical_metric_pruned',
    'polyspherical_inv_metric_pruned',
    'polyspherical_to_cartesian_term_pruned',
    'polyspherical_to_cartesian_pruned',
    '_extract_leaf_phases',
    'cartesian_to_polyspherical_pruned',
    'make_inverse_metric',
    'make_cartesian_to_polyspherical',
    'make_polyspherical_to_cartesian',
]


def get_term(i, active_params, fixed_params, active_vals, fixed_vals):
    """Return the angle for tree node *i* (active value or fixed constant)."""
    if i in fixed_params:
        if fixed_vals[fixed_params.index(i)] == 0:
            return 0
        elif fixed_vals[fixed_params.index(i)] == 1:
            return np.pi/2
    elif i in active_params:
        return active_vals[active_params.index(i)]


def polyspherical_metric_term_pruned(i, active_params, fixed_params, active_vals, fixed_vals):
    """Compute the metric factor for tree node *i* (recursive)."""
    if i == 0:
        return 1
    else:
        if (i - 1) % 2 == 0:
            return polyspherical_metric_term_pruned((i - 1) // 2, active_params, fixed_params, active_vals, fixed_vals) * np.cos(get_term((i - 1) // 2, active_params, fixed_params, active_vals, fixed_vals))
        else:
            return polyspherical_metric_term_pruned((i - 1) // 2, active_params, fixed_params, active_vals, fixed_vals) * np.sin(get_term((i - 1) // 2, active_params, fixed_params, active_vals, fixed_vals))

def polyspherical_metric_pruned(active_params, fixed_params, active_vals, fixed_vals):
    """Return the diagonal Fubini-Study metric in polyspherical coordinates."""
    metric = []
    for i in active_params:
        metric.append(polyspherical_metric_term_pruned(i, active_params, fixed_params, active_vals, fixed_vals)**2)
    return np.diag(metric)

def polyspherical_inv_metric_pruned(active_params, fixed_params, active_vals, fixed_vals, epsilon=1e-6):
	"""Return the inverse diagonal Fubini-Study metric in polyspherical coordinates.
	
	At a corner where a metric element is zero (parameter lies on a degenerate subspace),
	the inverse would diverge. However, the physical gradient on a degenerate manifold
	is zero in the degenerate directions, so the product (inv_metric * grad) is still well-defined
	as zero. To avoid 0*∞=NaN in floating-point, we set the inverse-metric diagonal to 0
	wherever the metric itself is negligible, effectively zeroing out the nat_grad component
	in those directions (which is the correct limit).
	"""
	inv_metric = []
	for i in active_params:
		metric_element = polyspherical_metric_term_pruned(i, active_params, fixed_params, active_vals, fixed_vals) ** 2
		# If metric element is negligibly small (at a corner singularity),
		# set inv_metric to 0 (no update in that direction; the manifold
		# is degenerate and the physical gradient is zero there).
		# Otherwise compute 1/metric as usual.
		if metric_element < epsilon:
			inv_metric.append(0.0)
		else:
			inv_metric.append(1 / metric_element)
	return np.diag(inv_metric)

def polyspherical_to_cartesian_term_pruned(i, n, active_params, fixed_params, active_vals, fixed_vals):
    """Recursively compute one Cartesian amplitude from polyspherical angles."""
    if n == 0:
        return 1
    else:
        if i % 2 == 0:
            return np.cos(get_term(i // 2 + 2**(n - 1) - 1, active_params, fixed_params, active_vals, fixed_vals)) * polyspherical_to_cartesian_term_pruned(i // 2, n - 1, active_params, fixed_params, active_vals, fixed_vals)
        else:
            return np.sin(get_term(i // 2 + 2**(n - 1) - 1, active_params, fixed_params, active_vals, fixed_vals)) * polyspherical_to_cartesian_term_pruned(i // 2, n - 1, active_params, fixed_params, active_vals, fixed_vals)


def polyspherical_to_cartesian_pruned(n, active_params, fixed_params, active_vals, fixed_vals,
                                      active_leaves, reordered_active_leaves, inactive_qubit_values,
                                      phase_active_params=None, phase_active_vals=None):
    """Convert polyspherical coordinates to a Cartesian state vector.

    When ``phase_active_params`` is ``None`` (default) the returned vector is
    real and matches the amplitude-only ansatz.  When ``phase_active_params``
    is a sentinel (any non-``None`` value -- the per-leaf omega indexing is
    implicit in ``reordered_active_leaves``) and ``phase_active_vals`` is a
    length-``|S|`` array of per-leaf phases, the returned vector is complex
    and equal to the input amplitudes dressed leaf-wise by ``exp(i * omega_k)``.
    """
    if phase_active_params is None:
        state = np.zeros(2**n)
    else:
        state = np.zeros(2**n, dtype=complex)
    n_eff = n - len(inactive_qubit_values)
    for k, i in enumerate(reordered_active_leaves):
        amp = polyspherical_to_cartesian_term_pruned(
            i, n_eff, active_params, fixed_params, active_vals, fixed_vals,
        )
        if phase_active_params is not None:
            amp = amp * np.exp(1j * phase_active_vals[k])
        state[active_leaves[k]] = amp
    return state


def _extract_leaf_phases(X_complex, reordered_active_leaves):
    """Extract per-leaf phases ``omega_k = arg(X[reordered_active_leaves[k]])``
    in the canonical gauge ``omega[0] = 0``.

    The canonical gauge is enforced by subtracting the phase of the first
    active leaf, so the returned vector has ``omega[0] = 0`` always.  Leaves
    with vanishing amplitude have an undefined phase; we fall back to
    ``np.angle(0) = 0`` for those entries (any other choice is equivalent up
    to the ``omega[0]`` gauge).
    """
    omega = np.array([np.angle(X_complex[i]) for i in reordered_active_leaves])
    omega = omega - omega[0]
    return omega


def cartesian_to_polyspherical_pruned(X, n, active_params, active_leaves, reordered_active_leaves,
                                       phase_active_params=None):
    """Convert a Cartesian state vector to polyspherical coordinates.

    When ``phase_active_params`` is ``None`` (default) ``X`` is treated as a
    real vector and the function returns the active ``R_y`` tree parameters
    ``theta``.  When ``phase_active_params`` is provided (any non-``None``
    sentinel; per-leaf indexing is implicit in ``reordered_active_leaves``)
    ``X`` is treated as complex; the magnitudes drive the ``theta``
    extraction (via ``|X|``) and the leaf phases are extracted via
    :func:`_extract_leaf_phases` to recover the per-leaf phases ``omega``.
    In that case the function returns the tuple ``(theta, omega)``.  The
    canonical gauge ``omega[0] = 0`` is applied so that the reconstructed
    state matches ``X`` up to a global phase.

    Chart range conventions (real branch).  At each internal tree node, the
    canonical range of its angle ``theta`` depends on the *active-leaf count*
    under each child subtree, since the signs of children are exactly the
    signs of the polyspherical factors ``cos theta`` / ``sin theta``:

    * ``(>=2 left, >=2 right)`` ("NN"): both children are subtree norms
      (non-negative), so ``theta in [0, pi/2]``.  Two corners ``{0, pi/2}``.
    * ``(>=2 left, 1 right)`` ("NS"): right is a signed single leaf, left is
      a non-negative norm, so ``cos theta >= 0`` and ``theta in [-pi/2, pi/2]``.
      Three corners ``{-pi/2, 0, pi/2}``.
    * ``(1 left, >=2 right)`` ("SN"): left is a signed single leaf, right is
      a non-negative norm, so ``sin theta >= 0`` and ``theta in [0, pi]``.
      Three corners ``{0, pi/2, pi}``.
    * ``(1 left, 1 right)`` ("SS"): both children are signed single leaves,
      so ``theta in [-pi, pi]`` (a full circle).  Four corners
      ``{-pi/2, 0, pi/2, pi}``.

    The chart map below uses the signed leaf amplitude when a child
    subtree contains exactly one active leaf, and the Euclidean norm
    otherwise; this makes ``c2p`` surjective onto the circuit's image
    (in particular it captures negative amplitudes on FIXED-funneled
    single-leaf subtrees, which the previous level-based switch lost).
    """
    if phase_active_params is None:
        X_new = np.zeros(2**n)
        for k,i in enumerate(reordered_active_leaves):
            X_new[i] = np.real(X[active_leaves[k]])
        active_leaf_set = set(reordered_active_leaves)
        t = np.zeros(2**n - 1)
        for k in range(n, 0, -1):
            half = 2**(k - 1)
            for i in range(2**(n - k)):
                lo_x, hi_x = 2*i*half, (2*i + 1)*half
                lo_y, hi_y = (2*i + 1)*half, (2*i + 2)*half
                x_slice = X_new[lo_x:hi_x]
                y_slice = X_new[lo_y:hi_y]
                # Single-active-leaf subtree -> use signed amplitude (sum of
                # the slice picks out the one nonzero entry with its sign).
                # Multi-active-leaf subtree -> use Euclidean norm (the parent
                # angle's range is restricted so the factor is non-negative).
                n_act_L = sum(1 for lf in active_leaf_set if lo_x <= lf < hi_x)
                n_act_R = sum(1 for lf in active_leaf_set if lo_y <= lf < hi_y)
                x_val = float(x_slice.sum() if n_act_L == 1 else np.linalg.norm(x_slice))
                y_val = float(y_slice.sum() if n_act_R == 1 else np.linalg.norm(y_slice))
                t[2**(n - k) + i - 1] = np.arctan2(y_val, x_val)
        return t[active_params]

    # Complex path: extract theta from |X| and omega from arg(X).  We keep
    # both a complex copy (for phase extraction) and an absolute-value copy
    # (for the magnitude-driven theta extraction, which uses arctan2 and
    # therefore needs real, non-negative inputs at the leaf level).
    X_complex = np.zeros(2**n, dtype=complex)
    for k, i in enumerate(reordered_active_leaves):
        X_complex[i] = X[active_leaves[k]]
    X_abs = np.abs(X_complex)
    t = np.zeros(2**n - 1)
    for k in range(n, 0, -1):
        for i in range(2**(n - k)):
            x = X_abs[2*i*2**(k - 1): (2*i + 1)*2**(k - 1)]
            y = X_abs[(2*i + 1)*2**(k - 1): (2*i+2)*2**(k - 1)]
            if k != 1:
                x = np.linalg.norm(x)
                y = np.linalg.norm(y)
            else:
                x = x[0]
                y = y[0]
            t[2**(n - k) + i - 1] = np.arctan2(y, x)
    theta = t[active_params]
    omega = _extract_leaf_phases(X_complex, reordered_active_leaves)
    return theta, omega


def _select_reducer(reorder):
    """Pick the qubit-ordering reducer that matches the circuit builder.

    The pruned circuit (:func:`quantumsymmetry.treecircuit.affine.get_affine_blocks`)
    selects ``best_bit_reordering_bnb`` when ``reorder=True`` and
    ``discard_constant_bits`` otherwise.  The chart / metric / optimiser code
    paths must use the *same* reducer, since a different qubit permutation
    relabels the tree nodes and so reassigns each ``theta_i`` to a different
    elementary rotation -- producing a chart that no longer matches the
    circuit for the same parameter vector.
    """
    return best_bit_reordering_bnb if reorder else discard_constant_bits


def make_inverse_metric(n, active_leaves, reorder=False):
    """Return a callable that computes the inverse metric for the given active leaves."""
    qubits_order, inactive_qubit_indices, inactive_qubit_values, reordered_active_leaves = _select_reducer(reorder)(active_leaves, n)
    n = n - len(inactive_qubit_indices)
    active_params, inactive_params, fixed_params, fixed_params_vals = classify_params(n, reordered_active_leaves)
    return partial(polyspherical_inv_metric_pruned, active_params, fixed_params, fixed_vals=fixed_params_vals)

def make_cartesian_to_polyspherical(n, active_leaves, complex=False, reorder=False):
    """Return a callable that converts Cartesian vectors to polyspherical coordinates.

    With ``complex=False`` (default) the returned callable maps a real state
    vector to the active ``theta`` array.  With ``complex=True`` it maps a
    complex state vector to a ``(theta, omega)`` pair.
    """
    qubits_order, inactive_qubit_indices, inactive_qubit_values, reordered_active_leaves = _select_reducer(reorder)(active_leaves, n)
    n = n - len(inactive_qubit_indices)
    active_params, inactive_params, fixed_params, fixed_params_vals = classify_params(n, reordered_active_leaves)
    kwargs = dict(n=n, active_params=active_params,
                  active_leaves=active_leaves, reordered_active_leaves=reordered_active_leaves)
    if complex:
        # Per-leaf omega; the contents of phase_active_params are unused, the
        # sentinel just selects the complex code path.
        kwargs['phase_active_params'] = True
    return partial(cartesian_to_polyspherical_pruned, **kwargs)

def make_polyspherical_to_cartesian(n, active_leaves, complex=False, reorder=False):
    """Return a function that maps polyspherical coordinates to a Cartesian state vector.

    With ``complex=False`` (default) the returned callable takes ``active_vals``
    and produces a real state vector.  With ``complex=True`` it additionally
    takes ``phase_active_vals`` and produces a complex state vector that
    matches the action of the ``R_z`` phase block (up to a global phase).
    """
    qubits_order, inactive_qubit_indices, inactive_qubit_values, reordered_active_leaves = _select_reducer(reorder)(active_leaves, n)
    n_eff = n - len(inactive_qubit_indices)
    active_params, inactive_params, fixed_params, fixed_params_vals = classify_params(n_eff, reordered_active_leaves)
    kwargs = dict(n=n, active_params=active_params,
                  fixed_params=fixed_params, fixed_vals=fixed_params_vals,
                  active_leaves=active_leaves, reordered_active_leaves=reordered_active_leaves,
                  inactive_qubit_values=inactive_qubit_values)
    if complex:
        kwargs['phase_active_params'] = True
    return partial(polyspherical_to_cartesian_pruned, **kwargs)


