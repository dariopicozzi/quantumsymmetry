"""Pure binary-tree combinatorics: parameter classification, tree
topology, Gray/Walsh helpers, qubit reordering and constant-bit removal.

This is the foundation layer of the pruned tree-circuit machinery; it has
no dependencies on the other modules in this subpackage.
"""

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from functools import partial
from dataclasses import dataclass
from typing import List, Optional


__all__ = [
    'classify_params',
    'classify_phase_params',
    '_leaves_under',
    '_path_sign_matrix',
    '_leaf_to_internal_phase_matrix',
    'gray_code',
    'hadamard_walsh',
    '_level_offset',
    '_block_scale',
    'best_bit_reordering_bnb',
    'chart_topology',
    'node_angles',
    'node_amplitudes',
    'discard_constant_bits',
]


def classify_params(n, active_leaves):
    """
        Classify binary-tree parameter slots into active, inactive, and fixed sets.

    Parameters:
    - n (int): Number of qubits.
    - active_leaves (List[int]): Indices of active leaf nodes (0-based).

    Returns:
        - active_params (List[int]): Tree parameters that remain free variables.
        - inactive_params (List[int]): Tree parameters that are gauge freedom and
            may be chosen arbitrarily without changing the represented state family.
        - fixed_params (List[int]): Tree parameters forced by the support pattern.
        - fixed_params_vals (List[float]): Values of the fixed parameters in the
            normalized tree parameterization, corresponding to circuit angles 0 or
            ``pi/2``.
    """
    num_leaves = 2**n
    total_slots = (2**(n + 1)) - 1
    has_active = [False] * total_slots

    # Mark active leaves
    for lf in active_leaves:
        has_active[num_leaves - 1 + lf] = True

    active, inactive, fixed, fixed_vals = [], [], [], []

    # Traverse internal nodes in reverse (bottom-up)
    for i in reversed(range(num_leaves - 1)):
        left, right = has_active[2 * i + 1], has_active[2 * i + 2]
        if left and right:
            active.append(i)
        elif not left and not right:
            inactive.append(i)
        else:
            fixed.append(i)
            fixed_vals.append(0 if left else 1)
        has_active[i] = left or right

    return active[::-1], inactive[::-1], fixed[::-1], fixed_vals[::-1]

def classify_phase_params(active_params, inactive_params, fixed_params):
    """Reclassify tree-parameter indices for the phase (``R_z``) block.

    The user-facing phase parameters are the *per-leaf* phases ``omega_a``
    (see :func:`circuit`), but the internal Walsh-transform machinery
    operates on per-internal-node ``alpha`` angles obtained from ``omega`` by
    a fixed linear reduction (see :func:`_leaf_to_internal_phase_matrix`).
    For that reduction we need to know which internal-node ``alpha``s carry
    information.  The rule is::

        phase-active   = R_y-active   (both subtrees contain active leaves)
        phase-fixed    = R_y-fixed    (one subtree empty -> alpha gauge -> 0)
        phase-inactive = R_y-inactive (no active leaves below -> alpha = 0)

    R_y-fixed nodes must be folded in as ``fixed = 0`` (rather than as a
    free "inactive" column subject to Gaussian elimination) so that zeroing
    the column leaves untouched the surrounding subtrees' angle pattern
    (Gaussian elimination would redistribute non-zero contributions onto
    siblings whose active leaves must not pick up spurious phases).  Returns
    a 4-tuple ``(active, inactive, fixed, fixed_vals)`` matching the shape
    of :func:`classify_params`.
    """
    return (
        list(active_params),
        list(inactive_params),
        list(fixed_params),
        [0] * len(fixed_params),
    )


def _leaves_under(node, n):
    """Return the set of leaf indices in ``range(2**n)`` under BFS-internal
    node ``node`` (children at ``2*node+1`` and ``2*node+2``).

    BFS indexing convention: internal nodes occupy ``range(2**n - 1)`` and
    leaves occupy ``range(2**n - 1, 2**(n+1) - 1)`` with leaf index
    ``node - (2**n - 1)``.
    """
    internals = 2**n - 1
    if node >= internals:
        return {node - internals}
    return _leaves_under(2 * node + 1, n) | _leaves_under(2 * node + 2, n)


def _path_sign_matrix(n, active_internals, reordered_active_leaves):
    """Return the (|S| x |active_internals|) path-sign matrix ``S_path`` such
    that the Möttönen UC-Rz cascade dressed with per-internal-node phases
    ``alpha`` realises leaf phases ``S_path @ alpha``.

    Entry ``S_path[k, j]`` is ``+1`` if leaf ``reordered_active_leaves[k]``
    lies in the right subtree of ``active_internals[j]`` (and that node is
    an ancestor of the leaf), ``-1`` if in the left subtree, and ``0``
    otherwise.  Inactive / fixed internal nodes have ``alpha = 0`` and are
    not represented as columns.
    """
    S_path = np.zeros((len(reordered_active_leaves), len(active_internals)))
    for j, node in enumerate(active_internals):
        left_leaves = _leaves_under(2 * node + 1, n)
        right_leaves = _leaves_under(2 * node + 2, n)
        for k, leaf in enumerate(reordered_active_leaves):
            if leaf in left_leaves:
                S_path[k, j] = -1.0
            elif leaf in right_leaves:
                S_path[k, j] = +1.0
    return S_path


def _leaf_to_internal_phase_matrix(n, active_internals, reordered_active_leaves):
    """Return the (|active_internals| x |S|) reduction matrix ``M`` such that
    the cascade dressed with ``alpha = M @ omega`` realises leaf phases
    ``omega`` modulo a global phase.

    The cascade realises leaf phases ``S_path @ alpha`` (see
    :func:`_path_sign_matrix`).  For a pruned tree the column space of
    ``S_path`` is *not* in general the orthogonal complement of the all-ones
    vector, so the simple pseudoinverse of ``S_path`` does not absorb the
    global-phase mismatch into a global shift.  Instead we solve the
    augmented linear system

        [S_path | 1] [alpha; c]^T  =  omega,

    which is square (``|S| x |S|``) and invertible whenever the active
    subtree is connected (so that ``S_path`` has full column rank
    ``|S|-1`` and the all-ones vector is not in its column space, since at
    every active internal node both subtrees contain at least one active
    leaf).  ``M`` is the first ``|active_internals|`` rows of the inverse;
    the discarded row recovers the global-phase ``c`` and is irrelevant.
    Edge case ``|S| = 1``: no active internals, return an empty
    ``(0 x 1)`` matrix; the cascade has no RZ gates and the canonical-gauge
    ``omega[0] = 0`` is realised trivially.
    """
    n_leaves = len(reordered_active_leaves)
    if not active_internals:
        return np.zeros((0, n_leaves))
    S_path = _path_sign_matrix(n, active_internals, reordered_active_leaves)
    A = np.hstack([S_path, np.ones((n_leaves, 1))])
    A_inv = np.linalg.inv(A)
    return A_inv[:-1, :]

def gray_code(n):
    """Generate a list of the first 2^n Gray code numbers."""
    result = []
    for i in range(2**n):
        gray = i ^ (i >> 1)
        result.append(gray)
    return result

def hadamard_walsh(n):
    """Return the 2^n x 2^n Hadamard-Walsh matrix (recursive construction)."""
    if n == 0:
        return np.array([[1]])
    else:
        H = hadamard_walsh(n - 1)
        return np.block([[H, H], [H, -H]])

def _level_offset(level):
    """Return the global BFS offset of the block at the given level."""
    return 2**level - 1

def _block_scale(n, q):
    """Return the clean block prefactor for final effective qubit q."""
    return 2**(q - n + 2)

def best_bit_reordering_bnb(active_indices, n):
    """
    Finds a permutation of the (variable) bits that maximizes the number of
    inactive internal nodes, given a fixed tree depth n >= ceil(log2(max_leaf+1)).

    Returns:
      - used_bits_order: the list of variable bit positions (from the original indexing)
                         in the new order (LSB -> MSB)
      - discarded_bit_indices: which bits were constant across all leaves
      - discarded_bit_values: the 0/1 values for those discarded bits
      - new_leaf_values: the permuted indices of each input leaf (in the same order).
    """

    # 1. Identify constant vs. variable bits
    if not active_indices:
        # No leaves => trivial
        return [], [], [], []

    max_idx = max(active_indices)
    N = n

    discarded_bits = []
    discarded_values = []
    varying_bits = []

    # Determine which bits are constant
    for b in range(N):
        first_val = (active_indices[0] >> b) & 1
        if all(((x >> b) & 1) == first_val for x in active_indices):
            discarded_bits.append(b)
            discarded_values.append(first_val)
        else:
            varying_bits.append(b)

    # Edge case: if no variable bits remain, all indices are the same
    if not varying_bits:
        # Everything is constant => all leaves map to 0 in the "new" system
        new_leaf_values = [0]*len(active_indices)
        return ([],
                discarded_bits,
                discarded_values,
                new_leaf_values)

    M = len(varying_bits)

    # Precompute, for each leaf, the array of variable bits (0/1)
    leaf_varbit_tuples = []
    for x in active_indices:
        # leaf_varbit_tuples[i] = tuple of bit values for each bit in 'varying_bits', in that order
        leaf_varbit_tuples.append(tuple((x >> b) & 1 for b in varying_bits))

    # We only care about maximizing the number of inactive nodes; 'n' is fixed.
    global_best_inactive_count = -1
    global_best_perm = None
    global_best_permuted_vals = None

    # We'll do a full backtracking over permutations of [0..M-1].
    # (Branch-and-bound on "inactive count" is possible but nontrivial,
    #  so we'll keep it straightforward unless the user specifically needs advanced pruning.)

    def backtrack(current_perm, used):
        nonlocal global_best_inactive_count, global_best_perm, global_best_permuted_vals

        # If we've used all M bits, evaluate
        if len(current_perm) == M:
            permuted_values = []
            for bits_tuple in leaf_varbit_tuples:
                val = 0
                # current_perm[i] is which index in [0..M-1] we put at position i (LSB->MSB)
                # But we need to read bits_tuple[...] accordingly:
                for new_pos, var_bit_idx in enumerate(current_perm):
                    bit_val = bits_tuple[var_bit_idx]  # 0 or 1
                    val |= (bit_val << new_pos)
                permuted_values.append(val)

            # Now run classify_params using the user-supplied n
            _, inactives, _, _ = classify_params(n, permuted_values)
            inactives_count = len(inactives)

            if inactives_count > global_best_inactive_count:
                global_best_inactive_count = inactives_count
                global_best_perm = current_perm[:]
                global_best_permuted_vals = permuted_values
            return

        # Otherwise, pick the next unused bit
        for b_idx in range(M):
            if not used[b_idx]:
                used[b_idx] = True
                current_perm.append(b_idx)

                backtrack(current_perm, used)

                current_perm.pop()
                used[b_idx] = False

    used = [False]*M
    backtrack([], used)

    if global_best_perm is None:
        # Something degenerate happened (e.g. empty leaves).
        return [], discarded_bits, discarded_values, []

    # Convert the permutation from [0..M-1] (indexes of 'varying_bits') to actual bit indices:
    final_bit_order = [varying_bits[i] for i in global_best_perm]

    return final_bit_order, discarded_bits, discarded_values, global_best_permuted_vals


# ---------------------------------------------------------------------------
# Chart / binary-tree topology helpers
#
# These three functions package the recurring three-line boilerplate
#
#     _, iqi, _, ral = best_bit_reordering_bnb(active_leaves, n)
#     n_eff = n - len(iqi)
#     ap, ip, fp, fv = classify_params(n_eff, ral)
#
# together with BFS index bookkeeping (internal nodes occupy
# ``[0, 2**n_eff - 1)``; leaves occupy ``[2**n_eff - 1, 2**(n_eff+1) - 1)``).
# They are used by the real-time TDVP integrator and any analysis tool that
# walks the pruned binary tree.
# ---------------------------------------------------------------------------

def chart_topology(n, active_leaves, reorder=False):
    """Resolve the pruned binary-tree topology used by the polyspherical chart.

    ``reorder`` selects the qubit-ordering reducer and must match the value
    passed to the circuit builder: ``best_bit_reordering_bnb`` when ``True``,
    ``discard_constant_bits`` when ``False`` (default).  Using a different
    reducer here relabels the tree nodes relative to the compiled circuit.

    Returns a ``dict`` with all node-indexing information needed by any
    per-node tree algorithm:

    * ``n_eff``                effective qubit count after constant-bit removal
    * ``n_internals``          ``2**n_eff - 1``
    * ``n_nodes``              ``2**(n_eff + 1) - 1`` (internals + leaves)
    * ``n_leaves``             ``len(reordered_active_leaves)``
    * ``active_params``        BFS indices of free internal nodes
    * ``inactive_params``      BFS indices of gauge-redundant internal nodes
    * ``fixed_params``         BFS indices of internal nodes forced by support
    * ``fixed_params_vals``    matching values (``0`` -> angle 0, ``1`` -> pi/2)
    * ``ral``                  reordered active-leaf indices in the n_eff tree
    * ``active_idx``           map ``BFS node a -> position in active_params``
    * ``fixed_val_map``        map ``BFS node a -> {0,1}`` for fixed nodes
    * ``leaf_tree_node``       per active-leaf-k tree-node index
                               (``n_internals + ral[k]``)
    """
    reducer = best_bit_reordering_bnb if reorder else discard_constant_bits
    _, iqi, _, ral = reducer(active_leaves, n)
    n_eff = n - len(iqi)
    active_params, inactive_params, fixed_params, fixed_params_vals = \
        classify_params(n_eff, ral)
    n_internals = (2 ** n_eff) - 1 if n_eff > 0 else 0
    n_nodes = (2 ** (n_eff + 1)) - 1 if n_eff > 0 else 1
    n_leaves = len(ral)
    return {
        "n_eff": n_eff,
        "n_internals": n_internals,
        "n_nodes": n_nodes,
        "n_leaves": n_leaves,
        "active_params": active_params,
        "inactive_params": inactive_params,
        "fixed_params": fixed_params,
        "fixed_params_vals": fixed_params_vals,
        "ral": ral,
        "active_idx": {a: i for i, a in enumerate(active_params)},
        "fixed_val_map": dict(zip(fixed_params, fixed_params_vals)),
        "leaf_tree_node": [n_internals + ral[k] for k in range(n_leaves)],
    }


def node_angles(theta, n=None, active_leaves=None, *, topo=None):
    """Return the BFS-indexed angle array ``angle[a]`` for every internal node.

    Active nodes take their value from ``theta`` (in ``topo['active_params']``
    order); fixed nodes are pinned to ``0`` or ``pi/2`` according to
    ``fixed_val_map``; inactive nodes default to ``0``.

    Either pass ``(n, active_leaves)`` or a pre-computed ``topo`` dict from
    :func:`chart_topology`.
    """
    if topo is None:
        if n is None or active_leaves is None:
            raise TypeError("node_angles requires either topo or (n, active_leaves)")
        topo = chart_topology(n, active_leaves)
    theta = np.asarray(theta, float)
    angle = np.zeros(topo["n_internals"])
    aidx = topo["active_idx"]
    fmap = topo["fixed_val_map"]
    for a in range(topo["n_internals"]):
        if a in aidx:
            angle[a] = float(theta[aidx[a]])
        elif a in fmap:
            angle[a] = 0.0 if fmap[a] == 0 else np.pi / 2.0
        else:
            angle[a] = 0.0
    return angle


def node_amplitudes(theta=None, n=None, active_leaves=None, *,
                    angle=None, topo=None):
    """Return per-tree-node amplitudes ``r[a]`` (root = 1; leaves = chart ``|c_k|``).

    Top-down product: ``r[2a+1] = r[a] * cos(angle[a])`` and
    ``r[2a+2] = r[a] * sin(angle[a])``.  Either pass ``theta`` (and
    ``n, active_leaves`` or ``topo``) -- in which case the angle array is
    derived via :func:`node_angles` -- or supply the precomputed ``angle``
    array directly together with ``topo``.
    """
    if topo is None:
        if n is None or active_leaves is None:
            raise TypeError("node_amplitudes requires either topo or (n, active_leaves)")
        topo = chart_topology(n, active_leaves)
    if angle is None:
        if theta is None:
            raise TypeError("node_amplitudes requires either theta or angle")
        angle = node_angles(theta, topo=topo)
    r = np.zeros(topo["n_nodes"])
    r[0] = 1.0
    for a in range(topo["n_internals"]):
        ca = np.cos(angle[a])
        sa = np.sin(angle[a])
        r[2 * a + 1] = r[a] * ca
        r[2 * a + 2] = r[a] * sa
    return r


def discard_constant_bits(active_indices, n):
    """Remove bits that are constant across all active leaves.

    Parameters
    ----------
    active_indices : list of int
        Active leaf indices in the original *n*-bit encoding.
    n : int
        Total number of qubits.

    Returns
    -------
    varying_bits : list of int
        Original bit positions that vary (LSB to MSB order, ascending).
    discarded_bits : list of int
        Original bit positions that are constant.
    discarded_values : list of int
        The 0/1 value of each discarded bit.
    new_leaves : list of int
        Active leaf indices re-encoded using only the varying bits.
    """

    varying_bits = []
    discarded_bits = []
    discarded_values = []

    for b in range(n):
        first_val = (active_indices[0] >> b) & 1
        if all(((x >> b) & 1) == first_val for x in active_indices):
            discarded_bits.append(b)
            discarded_values.append(first_val)
        else:
            varying_bits.append(b)

    if not varying_bits:
        return [], discarded_bits, discarded_values, [0] * len(active_indices)

    new_leaves = []
    for x in active_indices:
        val = 0
        for new_pos, old_bit in enumerate(varying_bits):
            val |= (((x >> old_bit) & 1) << new_pos)
        new_leaves.append(val)

    return varying_bits, discarded_bits, discarded_values, new_leaves

