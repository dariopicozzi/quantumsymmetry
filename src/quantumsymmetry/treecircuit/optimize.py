"""Gradients and singular-point initialisation.

Parameter-shift and finite-difference gradients plus the singular-point
initialisation routines used to escape rank-deficient regions of the
parameter manifold.
"""

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from functools import partial
from dataclasses import dataclass
from typing import List, Optional

from .tree import *
from .affine import *
from .circuit import *
from .metric import *


__all__ = [
    'shift_rule_gradient',
    'finite_difference_gradient',
    'singular_initialise',
    'singular_initialise_realtime',
    '_two_frame_shift_grad',
    '_singular_initialise_shift_rule_sweep',
    'singular_initialise_complex',
    'singular_initialise_realtime_shift_rule',
]


# ---------------------------------------------------------------------------
# Exact analytic gradient (four-term shift rule)
# ---------------------------------------------------------------------------

# Each tree parameter theta_i drives a single uniformly-controlled R_y(2*theta_i)
# whose generator G_i = -|b_i><b_i|_ctrl x Y has spectrum {-1, 0, +1}; hence
# E(theta_i) is a degree-2 trigonometric polynomial and its derivative admits
# the exact four-term shift rule with the weights below. See the appendix
# "Derivation of the four-term shift rule" of the accompanying paper.
_SHIFT_ALPHA = (2.0 + np.sqrt(2.0)) / 4.0   # weight of the +/- pi/4 pair
_SHIFT_BETA  = (2.0 - np.sqrt(2.0)) / 4.0   # weight of the +/- 3 pi/4 pair
_SHIFT_S1    = np.pi / 4.0
_SHIFT_S2    = 3.0 * np.pi / 4.0


def shift_rule_gradient(energy_func, params):
    """Exact four-term parameter-shift gradient of an energy function whose
    parameters are tree parameters of a circuit produced by :func:`circuit`.

    Costs four evaluations of ``energy_func`` per parameter (vs two for central
    finite differences), in exchange for being bias-free to floating-point
    precision and free of any ``epsilon`` to tune.

    Parameters
    ----------
    energy_func : callable
        Maps a 1-D parameter array to a real scalar energy
        ``E(params) = <psi(params)|H|psi(params)>``.
    params : array-like
        Current parameter vector.

    Returns
    -------
    numpy.ndarray
        Gradient vector with the same shape as ``params``.
    """
    params = np.asarray(params, dtype=float)
    grad = np.zeros_like(params)
    for i in range(params.size):
        pp1 = params.copy(); pp1[i] += _SHIFT_S1
        pm1 = params.copy(); pm1[i] -= _SHIFT_S1
        pp2 = params.copy(); pp2[i] += _SHIFT_S2
        pm2 = params.copy(); pm2[i] -= _SHIFT_S2
        grad[i] = (
            _SHIFT_ALPHA * (energy_func(pp1) - energy_func(pm1))
            - _SHIFT_BETA * (energy_func(pp2) - energy_func(pm2))
        )
    return grad


def finite_difference_gradient(energy_func, params, eps=1e-3):
    """Central finite-difference gradient (two evaluations per parameter).

    Provided as a convenience companion to :func:`shift_rule_gradient` so that
    callers can switch between the two estimators behind a single interface.
    """
    params = np.asarray(params, dtype=float)
    grad = np.zeros_like(params)
    for i in range(params.size):
        pp = params.copy(); pp[i] += eps
        pm = params.copy(); pm[i] -= eps
        grad[i] = (energy_func(pp) - energy_func(pm)) / (2.0 * eps)
    return grad


# ---------------------------------------------------------------------------
# Analytic singular-initialisation rule for natural-gradient descent
# ---------------------------------------------------------------------------

def singular_initialise(initial_params, energy_func, n, active_leaves, lr, *, tau=1e-10, reorder=False):
    """Replace a corner-valued parameter vector by an interior point obtained
    from the analytic singular-update rule for diagonal-Fubini-Study natural
    gradient descent.

    At a corner of the polyspherical chart (``theta_a in {0, pi/2}`` for some
    ancestor ``a``) the diagonal metric ``g_ii`` has zero entries on every
    parameter living in the closed subtree, so the natural-gradient step
    ``-lr * g^{-1} * grad E`` is undefined.  Hartree-Fock is precisely such
    a corner: every active angle starts at ``0`` or ``pi/2``.  This routine
    walks the tree top-down and, at each active internal node ``a`` whose
    angle is at a corner, applies the analytic rule

        D_j      = r_a * dE/dX_j           (j ranges over active leaves of R)
        Y*       = -D / ||D||              (downhill direction in R)
        alpha'   = lr * ||D|| / r_a^2      (linear NG step in opening coord)

    where ``R`` is the closed subtree rooted at the child of ``a`` whose
    metric prefactor vanishes, ``r_a`` is the product of cos/sin factors over
    the open ancestors of ``a`` (i.e. ``polyspherical_metric_term_pruned`` at
    ``a``), and ``D_j`` is computed via 4-term parameter-shift on
    ``theta_a`` while the internal angles of ``R`` encode the corner state
    ``e_j`` (i.e. all amplitude on leaf ``j``).  After application, ``a``
    is opened (``theta_a' = alpha'`` if it was at 0, ``pi/2 - alpha'`` if at
    pi/2) and the internal angles of ``R`` are set to encode ``Y*``.  The
    sweep then recurses into the *open* side of ``a`` (where the residual
    HF amplitude lives), which itself is at a corner one level deeper.

    Cost: ``4 * |R|`` energy evaluations per opener processed.

    Parameters
    ----------
    initial_params : array-like
        Starting active-parameter vector, typically ``c2p(hf_state)``.
        Entries must be approximately at ``0`` or ``pi/2`` for the rule to
        fire; otherwise the corresponding subtree is left unchanged.
    energy_func : callable
        Maps an active-parameter array to a real scalar energy.
    n : int
        Number of qubits (full circuit, before pruning).
    active_leaves : list[int]
        Computational-basis indices that span the active subspace, in the
        same convention used elsewhere (matches ``classify_params`` /
        ``cartesian_to_polyspherical_pruned``).
    lr : float
        Learning rate ``eta`` controlling the magnitude of the opening step
        ``alpha' = lr * ||D|| / r_a^2``.  Use the same value as the natural
        gradient descent that follows.
    tau : float, optional
        Absolute tolerance for detecting a corner ``theta_a in {0, pi/2}``.
        Defaults to ``1e-10`` (essentially exact).

    Returns
    -------
    numpy.ndarray
        Interior parameter vector, suitable as the initial point for an
        unregularised diagonal-NG descent (no clipping, no inverse-metric
        nan_to_num cap, no nat-grad clip required).
    """
    qubits_order, inactive_qubit_indices, _inactive_qubit_values, reordered_active_leaves = \
        (best_bit_reordering_bnb if reorder else discard_constant_bits)(active_leaves, n)
    n_eff = n - len(inactive_qubit_indices)
    active_params, inactive_params, fixed_params, fixed_params_vals = \
        classify_params(n_eff, reordered_active_leaves)

    params = np.asarray(initial_params, dtype=float).copy()
    if params.size != len(active_params):
        raise ValueError(
            f"initial_params has length {params.size} but classify_params expects "
            f"{len(active_params)} active parameters"
        )

    active_idx = {p: k for k, p in enumerate(active_params)}
    fixed_val_at = dict(zip(fixed_params, fixed_params_vals))  # 0 -> theta=0, 1 -> theta=pi/2
    n_internals = 2**n_eff - 1

    def is_leaf(node):
        return node >= n_internals

    # Cache leaf membership per node.
    _leaves_cache = {}
    def leaves_under(node):
        if node not in _leaves_cache:
            _leaves_cache[node] = _leaves_under(node, n_eff)
        return _leaves_cache[node]

    def active_leaf_positions_under(node):
        """List of indices k into ``reordered_active_leaves`` whose leaf is under ``node``."""
        leaf_set = leaves_under(node)
        return [k for k, lf in enumerate(reordered_active_leaves) if lf in leaf_set]

    def metric_prefactor(node):
        """r_node = product of cos/sin over proper ancestors of ``node``, using current params."""
        return polyspherical_metric_term_pruned(
            node, active_params, fixed_params, list(params), fixed_params_vals,
        )

    def encode_ej(closed_root, target_leaf_pos):
        """Walk path from ``closed_root`` to active leaf at ``target_leaf_pos`` (index
        into ``reordered_active_leaves``) and set every active ``theta_v`` along
        the path to encode amplitude 1 on that leaf within the subtree.
        Other active params under ``closed_root`` may be left at any value.
        Returns the list of (param, prev_value) pairs touched, for restoration.
        """
        target_leaf = reordered_active_leaves[target_leaf_pos]
        touched = []
        node = closed_root
        while not is_leaf(node):
            left_set = leaves_under(2 * node + 1)
            go_left = target_leaf in left_set
            if node in active_idx:
                k = active_idx[node]
                touched.append((k, params[k]))
                params[k] = 0.0 if go_left else np.pi / 2
            # FIXED node: forced to the active side; consistent with go_left automatically.
            node = (2 * node + 1) if go_left else (2 * node + 2)
        return touched

    def encode_subtree(node, target):
        """Set active angles under ``node`` to encode signed amplitude vector ``target``.

        ``target`` is indexed by ``active_leaf_positions_under(node)`` order.
        For non-leaf children with multiple active leaves, the magnitude
        ``||target_child||`` is encoded at the parent's angle and signs are
        delegated to descendant atan2 calls.  For children that resolve to
        a single active leaf (whether leaf node or sub-subtree pruned to one
        leaf via FIXED nodes), the signed scalar is used directly so that
        negative amplitudes are absorbed into the parent angle.
        """
        if is_leaf(node):
            return
        L, R = 2 * node + 1, 2 * node + 2
        L_pos = active_leaf_positions_under(L)
        R_pos = active_leaf_positions_under(R)
        all_pos = active_leaf_positions_under(node)
        # Build index maps from absolute leaf-pos -> position in `target`
        target_idx = {p: i for i, p in enumerate(all_pos)}
        Y_L = np.array([target[target_idx[p]] for p in L_pos]) if L_pos else np.array([])
        Y_R = np.array([target[target_idx[p]] for p in R_pos]) if R_pos else np.array([])

        if len(L_pos) <= 1:
            cos_val = float(Y_L[0]) if len(L_pos) == 1 else 0.0
        else:
            cos_val = float(np.linalg.norm(Y_L))
        if len(R_pos) <= 1:
            sin_val = float(Y_R[0]) if len(R_pos) == 1 else 0.0
        else:
            sin_val = float(np.linalg.norm(Y_R))

        if node in active_idx:
            params[active_idx[node]] = float(np.arctan2(sin_val, cos_val))
        # FIXED / INACTIVE: do not write; the existing forced value is consistent.

        if len(L_pos) > 1:
            encode_subtree(L, Y_L)
        if len(R_pos) > 1:
            encode_subtree(R, Y_R)

    def compute_D(opener_node, closed_root, leaves_in_R):
        """Return raw 4-term-shift gradient w.r.t. ``theta_{opener_node}`` evaluated
        once per leaf j in R with the internal angles of R set to encode e_j.

        Returns array of length ``len(leaves_in_R)``.
        """
        ai = active_idx[opener_node]
        theta0 = params[ai]
        D_raw = np.zeros(len(leaves_in_R))
        for jj, leaf_pos in enumerate(leaves_in_R):
            touched = encode_ej(closed_root, leaf_pos)
            params[ai] = theta0 + _SHIFT_S1; e_pp1 = energy_func(params)
            params[ai] = theta0 - _SHIFT_S1; e_pm1 = energy_func(params)
            params[ai] = theta0 + _SHIFT_S2; e_pp2 = energy_func(params)
            params[ai] = theta0 - _SHIFT_S2; e_pm2 = energy_func(params)
            params[ai] = theta0
            D_raw[jj] = (
                _SHIFT_ALPHA * (e_pp1 - e_pm1)
                - _SHIFT_BETA * (e_pp2 - e_pm2)
            )
            # Restore touched params (so the next e_j encoding starts from the
            # same baseline; not strictly necessary because we overwrite, but
            # keeps params self-consistent if energy_func is called externally).
            for k, prev in touched:
                params[k] = prev
        return D_raw

    def sweep(node):
        if is_leaf(node):
            return
        if node in active_idx:
            theta = params[active_idx[node]]
            # A corner is any theta with sin(theta) == 0 or cos(theta) == 0,
            # i.e. theta is a multiple of pi/2.  The sin/cos detector covers
            # all four canonical corners {0, +-pi/2, pi} (and their 2*pi
            # translates) uniformly, regardless of which of the cases NN /
            # NS / SN / SS the node falls into (see
            # :func:`cartesian_to_polyspherical_pruned` for the case taxonomy).
            #
            # The opener formula ``theta_new = theta + sign * Y_sign * alpha'``
            # is the *same* for every corner: ``sign = f'_a(theta)`` is the
            # derivative of the closed-side factor at the current theta
            # (= cos(theta) for sin-closed corners and = -sin(theta) for
            # cos-closed corners), and ``Y_sign`` absorbs the steepest-descent
            # sign in the single-leaf-R case.  This works because at any
            # corner the open child's amplitude is locally invariant
            # (its derivative w.r.t. theta vanishes), so the 4-term shift on
            # theta sees only contributions from the closed-side leaves and
            # captures whatever sign the open side already has in the state.
            s_th = np.sin(theta)
            c_th = np.cos(theta)
            if abs(s_th) < tau:
                # sin theta = 0 -> right child is closed (sin factor zero).
                # Snap theta to the nearest exact corner in {0, pi} and
                # remember the derivative sign of sin at that corner.
                closed_root = 2 * node + 2
                open_child = 2 * node + 1
                if c_th >= 0:
                    theta_snap = 0.0; sign = 1.0
                else:
                    theta_snap = np.pi; sign = -1.0
            elif abs(c_th) < tau:
                # cos theta = 0 -> left child is closed (cos factor zero).
                # Snap theta to the nearest exact corner in {+-pi/2} and
                # remember the derivative sign of cos at that corner.
                closed_root = 2 * node + 1
                open_child = 2 * node + 2
                if s_th >= 0:
                    theta_snap = np.pi / 2; sign = -1.0
                else:
                    theta_snap = -np.pi / 2; sign = 1.0
            else:
                # Active but interior: descend into both sides.
                sweep(2 * node + 1)
                sweep(2 * node + 2)
                return
            leaves_in_R = active_leaf_positions_under(closed_root)
            if not leaves_in_R:
                sweep(open_child); return
            r_a = metric_prefactor(node)
            if abs(r_a) < 1e-14:
                # Ancestor still degenerate; cannot fix this level yet.
                sweep(open_child); return
            D_raw = compute_D(node, closed_root, leaves_in_R)
            D = sign * D_raw          # gradient w.r.t. opening coord alpha
            D_norm = float(np.linalg.norm(D))
            if D_norm < 1e-14:
                # Energy independent of this closed subtree; leave at corner.
                sweep(open_child); return
            Y_star = -D / D_norm
            alpha_prime = lr * D_norm / (r_a * r_a)
            if len(leaves_in_R) == 1:
                # Single-leaf R: there is no inner angle in R to absorb the
                # sign of Y*[0], so it must be folded into the opener via
                # the opening direction.  Snap-then-add: write the canonical
                # corner value then take the linear NG step from there.
                Y_sign = 1.0 if Y_star[0] >= 0 else -1.0
                params[active_idx[node]] = theta_snap + sign * Y_sign * alpha_prime
            else:
                # Multi-leaf R: snap theta to the canonical corner, then take
                # the linear NG step in the opening direction.  Signs in R
                # are absorbed by encode_subtree (Y_star).
                params[active_idx[node]] = theta_snap + sign * alpha_prime
                encode_subtree(closed_root, Y_star)
            # Closed subtree fully initialised; descend into BOTH sides.
            # The open side still holds residual HF amplitude (possibly at a
            # corner one level deeper).  The closed side may also contain
            # newly-revealed sub-corners, e.g. when ``Y_star`` has exactly-zero
            # entries (a sub-branch of R received no amplitude), in which case
            # ``encode_subtree`` lands the corresponding sub-angle at 0 or
            # pi/2.  Recursing here gives the rule a chance to open those
            # sub-corners too.
            sweep(closed_root)
            sweep(open_child)
            return
        if node in fixed_val_at:
            # FIXED: only one side has active leaves; recurse into that side.
            if fixed_val_at[node] == 0:
                sweep(2 * node + 1)
            else:
                sweep(2 * node + 2)
            return
        # INACTIVE: no active leaves below; nothing to do.

    sweep(0)
    return params


# ---------------------------------------------------------------------------
# Real-time analog of the singular-initialisation rule
# ---------------------------------------------------------------------------

def singular_initialise_realtime(initial_theta, initial_omega, H_matvec, n,
                                 active_leaves, dt, *, tau=1e-10, reorder=False):
    """Open chart corners by matching the Cartesian Schroedinger step.

    .. warning::
        Reference implementation only.  This variant reads per-leaf
        amplitudes ``F_j = <j|H|psi_o>`` via an explicit ``H @ psi``
        matvec, which requires statevector access and is therefore not
        usable on quantum hardware.  Production code (and all hubbard
        dynamics runners) must call
        :func:`singular_initialise_realtime_shift_rule` instead, which
        produces the same ``(theta, omega)`` up to floating-point
        precision using only parameter-shift queries of
        ``E(theta, omega)``.  This matvec version is kept solely as a
        cross-check for the shift-rule implementation (see
        ``test_singular_shift_rule_regression.py``).

    Real-time analog of :func:`singular_initialise`.  At a chart corner
    (opener ``theta_a in {0, +-pi/2, pi}`` closing a subtree ``R`` with
    ``m`` active leaves), the diagonal Fubini--Study metric vanishes on
    every parameter strictly under ``R``, so the TDVP equation cannot
    determine the rates of those parameters.  We resolve this by matching
    the chart parameterisation to first order in ``dt`` against the exact
    Cartesian step ``|psi(dt)> = |psi_o> - i dt H|psi_o> + O(dt^2)`` where
    ``|psi_o>`` is the current state.  The induced amplitude on each
    active leaf ``j in R`` is the complex number ``F_j = <j_R|H|psi_o>``;
    matching against the chart amplitude
    ``alpha * r_a * Y_j * exp(i*phi_j)`` (to first order in
    ``alpha = theta_a - theta_corner``) gives

        Y_j   = |F_j| / ||F||,           (magnitude profile in R)
        alpha = ||F|| * dt / r_a,        (opener step)
        phi_j = arg(F_j) - pi/2,         (per-leaf phase in R)

    where ``r_a = polyspherical_metric_term_pruned(a)`` is the product of
    cos/sin factors over the *open* ancestors of ``a``.  The magnitudes
    are written into the inner ``theta``'s of ``R`` via the same atan2
    recursion used by :func:`singular_initialise`; the per-leaf phases
    are written directly into ``omega[k]`` for each leaf ``k`` of ``R``
    (the chart's phase block is per-leaf, no inner recursion needed).

    Compared with :func:`singular_initialise`:

    * The VQE rule matches the imaginary-time step ``|psi> - tau H|psi>``,
      so the matching is real; signs of ``Y_j`` are absorbed into
      ancestors via atan2.  Here ``Y_j >= 0`` always and the signs become
      per-leaf phases.
    * No learning-rate knob: the opener size is fixed by ``dt``.
    * Cost per opener: one ``H @ psi_o`` matvec plus the same tree walk.

    Parameters
    ----------
    initial_theta : array-like
        Active ``theta`` parameters, length ``len(active_params)``.
    initial_omega : array-like
        Per-leaf ``omega`` parameters, length ``len(active_leaves)``.
    H_matvec : callable or ndarray
        Either a dense/sparse ``(2**n, 2**n)`` matrix, or a callable
        mapping a length-``2**n`` complex state vector to ``H @ psi``.
    n : int
        Number of qubits (full circuit, before pruning).
    active_leaves : list[int]
        Computational-basis indices spanning the active subspace.
    dt : float
        Real-time step size.  Must be small enough that the first-order
        match against ``e^{-iH dt}|psi_o>`` is accurate.
    tau : float, optional
        Absolute tolerance for detecting ``sin(theta) = 0`` or
        ``cos(theta) = 0``.  Defaults to ``1e-10``.

    Returns
    -------
    theta, omega : tuple of numpy.ndarray
        Updated active ``theta`` and per-leaf ``omega`` arrays.  The closed
        subtree at every corner found by the sweep is opened in-place;
        non-corner branches are left untouched.
    """
    qubits_order, inactive_qubit_indices, inactive_qubit_values, reordered_active_leaves = \
        (best_bit_reordering_bnb if reorder else discard_constant_bits)(active_leaves, n)
    n_eff = n - len(inactive_qubit_indices)
    active_params, inactive_params, fixed_params, fixed_params_vals = \
        classify_params(n_eff, reordered_active_leaves)

    theta = np.asarray(initial_theta, dtype=float).copy()
    omega = np.asarray(initial_omega, dtype=float).copy()
    if theta.size != len(active_params):
        raise ValueError(
            f"initial_theta has length {theta.size} but expected {len(active_params)}"
        )
    if omega.size != len(active_leaves):
        raise ValueError(
            f"initial_omega has length {omega.size} but expected {len(active_leaves)}"
        )

    p2c = make_polyspherical_to_cartesian(n, active_leaves, complex=True, reorder=reorder)
    if callable(H_matvec):
        H_apply = H_matvec
    else:
        _H_mat = H_matvec
        def H_apply(psi, _M=_H_mat):
            return _M @ psi

    active_idx = {p: k for k, p in enumerate(active_params)}
    fixed_val_at = dict(zip(fixed_params, fixed_params_vals))
    n_internals = 2 ** n_eff - 1

    def is_leaf(node):
        return node >= n_internals

    _leaves_cache = {}
    def leaves_under(node):
        if node not in _leaves_cache:
            _leaves_cache[node] = _leaves_under(node, n_eff)
        return _leaves_cache[node]

    def active_leaf_positions_under(node):
        leaf_set = leaves_under(node)
        return [k for k, lf in enumerate(reordered_active_leaves) if lf in leaf_set]

    def metric_prefactor(node):
        return polyspherical_metric_term_pruned(
            node, active_params, fixed_params, list(theta), fixed_params_vals,
        )

    def encode_subtree_magnitudes(node, target):
        """Write non-negative magnitude profile ``target`` (indexed by
        ``active_leaf_positions_under(node)``) into inner theta's under
        ``node``.  Each internal node receives ``theta = atan2(||R||, ||L||)``;
        all resulting theta's lie in [0, pi/2].
        """
        if is_leaf(node):
            return
        L, R = 2 * node + 1, 2 * node + 2
        L_pos = active_leaf_positions_under(L)
        R_pos = active_leaf_positions_under(R)
        all_pos = active_leaf_positions_under(node)
        idx_in_target = {p: i for i, p in enumerate(all_pos)}
        Y_L = np.array([target[idx_in_target[p]] for p in L_pos]) if L_pos else np.array([])
        Y_R = np.array([target[idx_in_target[p]] for p in R_pos]) if R_pos else np.array([])
        cos_val = float(np.linalg.norm(Y_L)) if L_pos else 0.0
        sin_val = float(np.linalg.norm(Y_R)) if R_pos else 0.0
        if node in active_idx:
            theta[active_idx[node]] = float(np.arctan2(sin_val, cos_val))
        if len(L_pos) > 1:
            encode_subtree_magnitudes(L, Y_L)
        if len(R_pos) > 1:
            encode_subtree_magnitudes(R, Y_R)

    def sweep(node):
        if is_leaf(node):
            return
        if node in active_idx:
            th = theta[active_idx[node]]
            s_th, c_th = np.sin(th), np.cos(th)
            if abs(s_th) < tau:
                closed_root = 2 * node + 2
                open_child = 2 * node + 1
                if c_th >= 0:
                    theta_snap = 0.0; sign = 1.0
                else:
                    theta_snap = np.pi; sign = -1.0
            elif abs(c_th) < tau:
                closed_root = 2 * node + 1
                open_child = 2 * node + 2
                if s_th >= 0:
                    theta_snap = np.pi / 2; sign = -1.0
                else:
                    theta_snap = -np.pi / 2; sign = 1.0
            else:
                sweep(2 * node + 1)
                sweep(2 * node + 2)
                return
            leaves_in_R = active_leaf_positions_under(closed_root)
            if not leaves_in_R:
                sweep(open_child); return
            r_a = metric_prefactor(node)
            if abs(r_a) < 1e-14:
                sweep(open_child); return
            psi_o = p2c(active_vals=theta, phase_active_vals=omega)
            Hpsi = H_apply(psi_o)
            F = np.array([Hpsi[active_leaves[k]] for k in leaves_in_R], dtype=complex)
            F_norm = float(np.linalg.norm(F))
            if F_norm < 1e-14:
                sweep(open_child); return
            Y = np.abs(F) / F_norm
            phi = np.angle(F) - np.pi / 2
            alpha = F_norm * dt / r_a
            theta[active_idx[node]] = theta_snap + sign * alpha
            if len(leaves_in_R) > 1:
                encode_subtree_magnitudes(closed_root, Y)
            for k, phi_k in zip(leaves_in_R, phi):
                omega[k] = float(phi_k)
            sweep(closed_root)
            sweep(open_child)
            return
        if node in fixed_val_at:
            if fixed_val_at[node] == 0:
                sweep(2 * node + 1)
            else:
                sweep(2 * node + 2)
            return

    sweep(0)
    return theta, omega


# ---------------------------------------------------------------------------
# Shift-rule (pure (theta, omega)) variants of the singular-coordinate
# initialisation rules.  These use only parameter-shift queries on the
# energy and never construct the Cartesian state vector or apply H as a
# matvec.  At each fired corner with closed subtree containing m active
# leaves they cost 8 m shift-rule energy evaluations (vs 4 m for the
# real-H opener `singular_initialise`), and at most 16 |A| in total over a
# full top-down sweep (geometric bound), where |A| is the number of
# active leaves.  This is the same asymptotic cost as one NG/VITE update
# step (6 |A| - 4 queries), so the chart-singularity treatment is
# asymptotically free on pruned circuits.
#
# The primitive is the existing 4-term theta_a shift evaluated in TWO
# omega_j frames (current and current + pi/2):
#
#     D_j^Re = 4-term theta_a shift at omega_j     = 2 r_a Re[ e^{-i omega_j} F_j ]
#     D_j^Im = 4-term theta_a shift at omega_j+pi/2 = 2 r_a Im[ e^{-i omega_j} F_j ]
#
# from which the complex amplitude F_j = <j_R|H|psi_o> is reconstructed
# as F_j = e^{+i omega_j} (D_j^Re + i D_j^Im) / (2 r_a).  Both openers
# below specialise this to their respective flow:
#
#     VITE  :  omega_j <- omega_j + arg(D_j^Re + i D_j^Im) + pi
#              alpha'  = lr * ||D|| / r_a^2
#     TDVP  :  omega_j <- omega_j + arg(D_j^Re + i D_j^Im) - pi/2
#              alpha   = ||D|| * dt / (2 r_a^2)
#
# with ||D||^2 = sum_j (D_j^Re)^2 + (D_j^Im)^2 = 4 r_a^2 ||F||^2 and
# Y_j = sqrt((D_j^Re)^2 + (D_j^Im)^2) / ||D||.

def _two_frame_shift_grad(energy_func, theta, omega, idx_theta, idx_omega):
    """Return the complex amplitude  e^{-i omega_j} F_j / r_a-prefactor  via
    two 4-term theta shifts at omega and at omega + pi/2 on coordinate j.

    Returns ``(D_re, D_im)``: the two real shift-rule outputs.  The caller
    rescales by ``r_a`` and the e^{+i omega} factor to recover F_j.

    Eight energy evaluations total.
    """
    t = theta.copy()
    om = omega.copy()
    # Frame 0: current omega.
    t[idx_theta] = theta[idx_theta] + _SHIFT_S1; e_pp1 = energy_func(t, om)
    t[idx_theta] = theta[idx_theta] - _SHIFT_S1; e_pm1 = energy_func(t, om)
    t[idx_theta] = theta[idx_theta] + _SHIFT_S2; e_pp2 = energy_func(t, om)
    t[idx_theta] = theta[idx_theta] - _SHIFT_S2; e_pm2 = energy_func(t, om)
    t[idx_theta] = theta[idx_theta]
    D_re = (
        _SHIFT_ALPHA * (e_pp1 - e_pm1)
        - _SHIFT_BETA * (e_pp2 - e_pm2)
    )
    # Frame 1: omega_j shifted by +pi/2.
    om[idx_omega] = omega[idx_omega] + np.pi / 2
    t[idx_theta] = theta[idx_theta] + _SHIFT_S1; e_pp1 = energy_func(t, om)
    t[idx_theta] = theta[idx_theta] - _SHIFT_S1; e_pm1 = energy_func(t, om)
    t[idx_theta] = theta[idx_theta] + _SHIFT_S2; e_pp2 = energy_func(t, om)
    t[idx_theta] = theta[idx_theta] - _SHIFT_S2; e_pm2 = energy_func(t, om)
    D_im = (
        _SHIFT_ALPHA * (e_pp1 - e_pm1)
        - _SHIFT_BETA * (e_pp2 - e_pm2)
    )
    return D_re, D_im


def _singular_initialise_shift_rule_sweep(initial_theta, initial_omega,
                                          energy_func, n, active_leaves, *,
                                          phase_offset, step_scale, tau, reorder=False):
    """Generic shift-rule singular-coordinate initialisation sweep, common
    to the VITE and TDVP openers.  Parameters

        phase_offset : float       added to arg(D_re + i D_im) when writing omega_j
        step_scale   : float       step-size prefactor: alpha = step_scale * ||D|| / r_a^2

    Returns updated (theta, omega) arrays.

    VITE: phase_offset = +pi,   step_scale = lr
    TDVP: phase_offset = -pi/2, step_scale = dt / 2
    """
    qubits_order, inactive_qubit_indices, _inactive_qubit_values, reordered_active_leaves = \
        (best_bit_reordering_bnb if reorder else discard_constant_bits)(active_leaves, n)
    n_eff = n - len(inactive_qubit_indices)
    active_params, _inactive_params, fixed_params, fixed_params_vals = \
        classify_params(n_eff, reordered_active_leaves)

    theta = np.asarray(initial_theta, dtype=float).copy()
    omega = np.asarray(initial_omega, dtype=float).copy()
    if theta.size != len(active_params):
        raise ValueError(
            f"initial_theta has length {theta.size} but expected {len(active_params)}"
        )
    if omega.size != len(active_leaves):
        raise ValueError(
            f"initial_omega has length {omega.size} but expected {len(active_leaves)}"
        )

    active_idx = {p: k for k, p in enumerate(active_params)}
    fixed_val_at = dict(zip(fixed_params, fixed_params_vals))
    n_internals = 2 ** n_eff - 1

    def is_leaf(node):
        return node >= n_internals

    _leaves_cache = {}
    def leaves_under(node):
        if node not in _leaves_cache:
            _leaves_cache[node] = _leaves_under(node, n_eff)
        return _leaves_cache[node]

    def active_leaf_positions_under(node):
        leaf_set = leaves_under(node)
        return [k for k, lf in enumerate(reordered_active_leaves) if lf in leaf_set]

    def metric_prefactor(node):
        return polyspherical_metric_term_pruned(
            node, active_params, fixed_params, list(theta), fixed_params_vals,
        )

    def encode_subtree_magnitudes(node, target):
        """Write non-negative magnitude profile ``target`` into inner thetas."""
        if is_leaf(node):
            return
        L, R = 2 * node + 1, 2 * node + 2
        L_pos = active_leaf_positions_under(L)
        R_pos = active_leaf_positions_under(R)
        all_pos = active_leaf_positions_under(node)
        idx_in_target = {p: i for i, p in enumerate(all_pos)}
        Y_L = np.array([target[idx_in_target[p]] for p in L_pos]) if L_pos else np.array([])
        Y_R = np.array([target[idx_in_target[p]] for p in R_pos]) if R_pos else np.array([])
        cos_val = float(np.linalg.norm(Y_L)) if L_pos else 0.0
        sin_val = float(np.linalg.norm(Y_R)) if R_pos else 0.0
        if node in active_idx:
            theta[active_idx[node]] = float(np.arctan2(sin_val, cos_val))
        if len(L_pos) > 1:
            encode_subtree_magnitudes(L, Y_L)
        if len(R_pos) > 1:
            encode_subtree_magnitudes(R, Y_R)

    def encode_ej(closed_root, target_leaf_pos):
        """Walk path from ``closed_root`` to active leaf at ``target_leaf_pos``,
        set every active theta along the path to encode amplitude 1 on that
        leaf within the subtree.  Returns list of (k, prev) pairs touched.
        """
        target_leaf = reordered_active_leaves[target_leaf_pos]
        touched = []
        node = closed_root
        while not is_leaf(node):
            left_set = leaves_under(2 * node + 1)
            go_left = target_leaf in left_set
            if node in active_idx:
                k = active_idx[node]
                touched.append((k, theta[k]))
                theta[k] = 0.0 if go_left else np.pi / 2
            node = (2 * node + 1) if go_left else (2 * node + 2)
        return touched

    def compute_D_complex(opener_node, closed_root, leaves_in_R):
        """Return arrays D_re, D_im of length len(leaves_in_R) via the
        two-frame 4-term shift on theta_{opener_node}.  At each leaf j in R,
        rotate R to encode e_j, then call the two-frame primitive.
        """
        ai = active_idx[opener_node]
        m = len(leaves_in_R)
        D_re = np.zeros(m)
        D_im = np.zeros(m)
        for jj, leaf_pos in enumerate(leaves_in_R):
            touched = encode_ej(closed_root, leaf_pos)
            D_re[jj], D_im[jj] = _two_frame_shift_grad(
                energy_func, theta, omega, ai, leaf_pos
            )
            for k, prev in touched:
                theta[k] = prev
        return D_re, D_im

    def sweep(node):
        if is_leaf(node):
            return
        if node in active_idx:
            th = theta[active_idx[node]]
            s_th, c_th = np.sin(th), np.cos(th)
            if abs(s_th) < tau:
                closed_root = 2 * node + 2
                open_child = 2 * node + 1
                if c_th >= 0:
                    theta_snap = 0.0; sign = 1.0
                else:
                    theta_snap = np.pi; sign = -1.0
            elif abs(c_th) < tau:
                closed_root = 2 * node + 1
                open_child = 2 * node + 2
                if s_th >= 0:
                    theta_snap = np.pi / 2; sign = -1.0
                else:
                    theta_snap = -np.pi / 2; sign = 1.0
            else:
                sweep(2 * node + 1)
                sweep(2 * node + 2)
                return
            leaves_in_R = active_leaf_positions_under(closed_root)
            if not leaves_in_R:
                sweep(open_child); return
            r_a = metric_prefactor(node)
            if abs(r_a) < 1e-14:
                sweep(open_child); return
            D_re, D_im = compute_D_complex(node, closed_root, leaves_in_R)
            # Magnitude (always >= 0): goes into Y and into alpha.  The
            # snap-derivative `sign` is irrelevant for magnitudes (|sign|=1).
            D_norm_sq = D_re ** 2 + D_im ** 2
            D_norm = float(np.sqrt(np.sum(D_norm_sq)))
            if D_norm < 1e-14:
                sweep(open_child); return
            Y = np.sqrt(D_norm_sq) / D_norm
            # Phase of the per-leaf amplitude F_j = <j_R|H|psi_o>:
            # At a corner, dE/dtheta_a at the e_j-encoded state is
            #   D = sign * 2 r_a (Re + i Im)[e^{-i omega_j^old} F_j]
            # where the sign factor flips at the snaps {pi/2, -pi/2}
            # because d/dtheta (cos^2 - sin^2)|_{pi/2} = -2 (vs +2 at 0
            # and pi).  Therefore
            #   F_j = sign * e^{+i omega_j^old} (D_re + i D_im) / (2 r_a)
            # and arg(F_j) = omega_j^old + atan2(sign*D_im, sign*D_re).
            # The opener overwrites omega_j with arg(F_j) + phase_offset
            # where phase_offset = +pi (VITE) or -pi/2 (TDVP).
            old_omega = np.array([omega[k] for k in leaves_in_R])
            phi = old_omega + np.arctan2(sign * D_im, sign * D_re) + phase_offset
            alpha = step_scale * D_norm / (r_a * r_a)
            theta[active_idx[node]] = theta_snap + sign * alpha
            if len(leaves_in_R) > 1:
                encode_subtree_magnitudes(closed_root, Y)
            for k, phi_k in zip(leaves_in_R, phi):
                omega[k] = float(phi_k)
            sweep(closed_root)
            sweep(open_child)
            return
        if node in fixed_val_at:
            if fixed_val_at[node] == 0:
                sweep(2 * node + 1)
            else:
                sweep(2 * node + 2)
            return

    sweep(0)
    return theta, omega


def singular_initialise_complex(initial_theta, initial_omega, energy_func, n,
                                active_leaves, lr, *, tau=1e-10, reorder=False):
    """Shift-rule (pure (theta, omega)) singular-coordinate initialisation
    for VITE / natural-gradient descent with a possibly complex Hamiltonian.

    Generalises :func:`singular_initialise` to arbitrary complex H by
    reconstructing the complex per-leaf amplitude F_j = <j_R|H|psi_o> from
    two 4-term theta_a shifts (one at omega_j, one at omega_j + pi/2)
    rather than relying on the signed-Y* trick that only works for real H.

    Update at each fired corner:

        Y_j     = sqrt((D_j^Re)^2 + (D_j^Im)^2) / ||D||
        omega_j <- omega_j + arctan2(D_j^Im, D_j^Re) + pi
        alpha'  = lr * ||D|| / r_a^2

    Reduces bit-for-bit to :func:`singular_initialise` when H is real and
    omega starts real (then D_j^Im = 0 and the +pi phase carries the sign).

    Parameters
    ----------
    initial_theta : array-like
    initial_omega : array-like
    energy_func : callable
        ``energy_func(theta, omega) -> float``.  Standard chart energy.
    n : int
    active_leaves : list[int]
    lr : float
    tau : float, optional

    Returns
    -------
    theta, omega : tuple of numpy.ndarray
    """
    return _singular_initialise_shift_rule_sweep(
        initial_theta, initial_omega, energy_func, n, active_leaves,
        phase_offset=np.pi, step_scale=lr, tau=tau, reorder=reorder,
    )


def singular_initialise_realtime_shift_rule(initial_theta, initial_omega,
                                            energy_func, n, active_leaves,
                                            dt, *, tau=1e-10, reorder=False):
    """Shift-rule (pure (theta, omega)) singular-coordinate initialisation
    for variational real-time evolution (TDVP).

    Same primitive as :func:`singular_initialise_complex`, with TDVP
    specialisations:

        Y_j     = sqrt((D_j^Re)^2 + (D_j^Im)^2) / ||D||
        omega_j <- omega_j + arctan2(D_j^Im, D_j^Re) - pi/2
        alpha   = ||D|| * dt / (2 r_a^2)

    Produces the same (theta, omega) as :func:`singular_initialise_realtime`
    up to floating-point precision (verified by the regression test in
    test_singular_shift_rule.py).  Costs 8 |R| shift-rule energy evaluations
    per opener fired and at most ~16 |A| over a full sweep, where |A| is
    the number of active leaves.

    The matvec-based :func:`singular_initialise_realtime` remains available
    as a classical shortcut: on a classical simulator a single H @ psi_o
    matvec is cheaper than 8 |R| separate circuit evaluations, but on
    quantum hardware (where the Cartesian state vector is not accessible)
    this shift-rule variant is the canonical implementation.
    """
    return _singular_initialise_shift_rule_sweep(
        initial_theta, initial_omega, energy_func, n, active_leaves,
        phase_offset=-np.pi / 2, step_scale=dt / 2, tau=tau, reorder=reorder,
    )


