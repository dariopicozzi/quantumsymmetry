"""Affine-block construction and local pruning.

Builds the affine map from tree parameters to circuit angles, applies the
local pruning that exposes redundant CNOTs, and extracts the numeric affine
map used by the fast compiled path.
"""

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from functools import partial
from dataclasses import dataclass
from typing import List, Optional

from .tree import *


__all__ = [
    '_apply_local_pruning',
    '_build_affine_blocks',
    '_block_argument_expressions_qiskit',
    '_fixed_qubit_preparation',
    '_map_effective_qubits',
    'get_affine_blocks',
    '_extract_affine_map',
]


def _apply_local_pruning(matrix, vector_constants, fixed_cols, fixed_vals, inactive_cols):
    """Apply one block-local gauge-fixing step to the affine circuit-angle map.

    Fixed tree parameters are first folded into the constant term.  The
    inactive tree parameters are then solved away by Gaussian elimination on the
    associated inactive-variable columns, choosing pivot rows so that selected
    circuit ``RY`` angles vanish.  The remaining nonzero rows determine the
    surviving affine circuit angles for the block.
    """
    matrix = np.array(matrix, dtype=float, copy=True)
    vector_constants = np.array(vector_constants, dtype=float, copy=True)

    for col, val in zip(fixed_cols, fixed_vals):
        vector_constants += val * matrix[:, col]
        matrix[:, col] = 0.0

    for col in inactive_cols:
        for row in range(matrix.shape[0]):
            pivot = matrix[row, col]
            if abs(pivot) <= 1e-14:
                continue
            pivot_row = matrix[row] / pivot
            pivot_const = vector_constants[row] / pivot
            for other_row in range(matrix.shape[0]):
                if other_row == row:
                    continue
                factor = matrix[other_row, col]
                if abs(factor) > 1e-14:
                    matrix[other_row] -= factor * pivot_row
                    vector_constants[other_row] -= factor * pivot_const
            matrix[row] = 0.0
            vector_constants[row] = 0.0
            break

    return matrix, vector_constants

def _build_affine_blocks(n, active_params, inactive_params, fixed_params, fixed_params_vals):
    """Return unscaled active-only affine blocks in final effective-qubit order."""
    fixed_param_vals = dict(zip(fixed_params, fixed_params_vals))
    blocks = []

    for q in range(n):
        level = n - 1 - q
        size = 2**level
        offset = _level_offset(level)
        local_active_params = [param for param in active_params if offset <= param < offset + size]
        local_active_cols = [param - offset for param in local_active_params]
        local_fixed_params = [param for param in fixed_params if offset <= param < offset + size]
        local_fixed_cols = [param - offset for param in local_fixed_params]
        local_fixed_vals = [fixed_param_vals[param] for param in local_fixed_params]
        local_inactive_cols = [param - offset for param in inactive_params if offset <= param < offset + size]

        matrix = hadamard_walsh(level)[gray_code(level)].astype(float)
        vector_constants = np.zeros(size, dtype=float)
        matrix, vector_constants = _apply_local_pruning(
            matrix,
            vector_constants,
            local_fixed_cols,
            local_fixed_vals,
            local_inactive_cols,
        )

        surviving_rows = [row for row in range(size) if np.any(matrix[row] != 0) or vector_constants[row] != 0]
        blocks.append({
            'active_params': local_active_params,
            'matrix': matrix[np.ix_(surviving_rows, local_active_cols)],
            'constants': vector_constants[surviving_rows],
            'surviving_rows': surviving_rows,
        })

    return blocks

def _block_argument_expressions_qiskit(blocks, name='theta'):
    """Return active-only Qiskit expressions block by block in final effective-qubit order.

    Uses a single shared ``ParameterVector(name, M)`` whose entry ``name[j]``
    corresponds to the ``j``-th global active tree parameter (i.e. the order produced
    by :func:`classify_params`, matching the convention used by ``c2p`` /
    ``cartesian_to_polyspherical_pruned``).  Without this, Qiskit's alphabetical
    sort of multiple per-block ParameterVectors would scramble the parameter
    ordering relative to the coordinate map.  ``name`` is ``'theta'`` for the
    ``R_y`` amplitude block and ``'omega'`` for the ``R_z`` phase block.
    """
    block_args = []
    n = len(blocks)

    # Recover the global active_params ordering used by ``classify_params`` /
    # ``cartesian_to_polyspherical_pruned``: BFS ascending (root first).  Note that
    # the per-block lists are in BFS-descending order across blocks (block 0 holds
    # the deepest level), so we sort the union to recover the c2p convention.
    global_active_params = sorted({p for block in blocks for p in block['active_params']})
    params = ParameterVector(name, len(global_active_params))
    global_index = {p: idx for idx, p in enumerate(global_active_params)}

    for q, block in enumerate(blocks):
        args = []
        local_params = [params[global_index[p]] for p in block['active_params']]
        for row in range(len(block['surviving_rows'])):
            eq = 0
            for col, param in enumerate(local_params):
                coeff = _block_scale(n, q) * block['matrix'][row, col]
                if coeff != 0:
                    eq += coeff * param
            const = _block_scale(n, q) * block['constants'][row]
            if const != 0:
                eq += np.pi/2 * const
            args.append(eq)
        block_args.append(args)

    return block_args

def _fixed_qubit_preparation(inactive_qubit_indices, inactive_qubit_values):
    """Return basis-state preparation gates/angles for discarded qubits."""
    circuit_prefix = []
    args_prefix = []
    for bit, value in zip(inactive_qubit_indices, inactive_qubit_values):
        if value == 1:
            circuit_prefix.append(['RY', bit])
            args_prefix.append(np.pi)
    return circuit_prefix, args_prefix

def _map_effective_qubits(circuit_list, qubits_order, inactive_qubit_indices, inactive_qubit_values):
    """Map final effective-qubit labels directly to physical reordered qubits."""
    for gate in circuit_list:
        if gate[0] in ('RY', 'RZ'):
            gate[1] = qubits_order[gate[1]]
        elif gate[0] == 'CNOT':
            gate[1] = qubits_order[gate[1]]
            gate[2] = qubits_order[gate[2]]
    circuit_prefix, _ = _fixed_qubit_preparation(inactive_qubit_indices, inactive_qubit_values)
    return circuit_prefix + circuit_list

def get_affine_blocks(n, active_leaves, reorder=False, complex=False):
    """Return unscaled active-only affine blocks in reordered effective-qubit order.

    When ``complex=True`` the returned dict additionally contains a
    ``'phase_blocks'`` entry holding the analogous affine blocks for the
    ``R_z`` phase cascade, built from the phase classification of the same
    binary tree (see :func:`classify_phase_params`).
    """
    reducer = best_bit_reordering_bnb if reorder else discard_constant_bits
    qubits_order, inactive_qubit_indices, inactive_qubit_values, reordered_active_leaves = reducer(active_leaves, n)
    n_eff = n - len(inactive_qubit_indices) if reorder else len(qubits_order)
    active_params, inactive_params, fixed_params, fixed_params_vals = classify_params(n_eff, reordered_active_leaves)

    result = {
        'effective_num_qubits': n_eff,
        'qubits_order': qubits_order,
        'inactive_qubit_indices': inactive_qubit_indices,
        'inactive_qubit_values': inactive_qubit_values,
        'reordered_active_leaves': reordered_active_leaves,
        'blocks': _build_affine_blocks(
            n_eff,
            active_params,
            inactive_params,
            fixed_params,
            fixed_params_vals,
        ),
    }
    if complex:
        phase_active, phase_inactive, phase_fixed, phase_fixed_vals = classify_phase_params(
            active_params, inactive_params, fixed_params,
        )
        # Build the internal-node-indexed phase blocks (one alpha per active
        # internal node) and then convert them to per-leaf-omega indexing by
        # right-multiplying each block matrix by the leaf-to-internal
        # reduction map (alpha = M @ omega).  The user-facing phase
        # parameters become the |S| per-leaf omega's, indexed by
        # ``reordered_active_leaves`` order.
        #
        # The fixed (one-sided) nodes are folded in as ``alpha = 0`` exactly as
        # the classify_phase_params docstring requires: this zeroes their Walsh
        # columns so rows whose active-restricted expression vanishes are
        # dropped from ``surviving_rows`` instead of surviving as RZ(0)
        # identity gates (which also blocked CNOT cancellation in the cascade).
        internal_phase_blocks = _build_affine_blocks(
            n_eff, phase_active, phase_inactive, phase_fixed, phase_fixed_vals,
        )
        n_leaves = len(reordered_active_leaves)
        M_full = _leaf_to_internal_phase_matrix(
            n_eff, phase_active, reordered_active_leaves,
        )
        # Map each active internal node (global index) to its row in M_full.
        internal_row = {p: i for i, p in enumerate(phase_active)}
        leaf_indices = list(range(n_leaves))
        phase_blocks = []
        for block in internal_phase_blocks:
            local_rows = [internal_row[p] for p in block['active_params']]
            if local_rows:
                # block['matrix'] has shape (surviving_rows, |local active internals|);
                # M_full[local_rows] has shape (|local active internals|, |S|).
                new_matrix = block['matrix'] @ M_full[local_rows]
            else:
                new_matrix = np.zeros((len(block['surviving_rows']), n_leaves))
            phase_blocks.append({
                'active_params': leaf_indices,
                'matrix': new_matrix,
                'constants': block['constants'],
                'surviving_rows': block['surviving_rows'],
            })
        result['phase_blocks'] = phase_blocks
    return result

def _extract_affine_map(blocks):
    """Numeric counterpart of :func:`_block_argument_expressions_qiskit`.

    Returns ``(A, b)`` such that, in the order in which surviving rotations
    are emitted by :func:`_build_pruned_block` (blocks traversed as
    ``q = n-1, n-2, ..., 0`` and rows in ascending order within each block),
    the ``g``-th surviving angle equals ``A[g, :] @ theta + b[g]``.

    ``theta`` indexes the global active tree parameters in ascending order,
    matching the convention of :func:`_block_argument_expressions_qiskit` and
    :func:`classify_params`.
    """
    n = len(blocks)
    global_active_params = sorted({p for block in blocks for p in block['active_params']})
    global_index = {p: idx for idx, p in enumerate(global_active_params)}
    M = len(global_active_params)

    A_rows = []
    b_vals = []
    for q in range(n - 1, -1, -1):
        block = blocks[q]
        scale = _block_scale(n, q)
        local_cols = [global_index[p] for p in block['active_params']]
        mat = block['matrix']
        consts = block['constants']
        for row in range(len(block['surviving_rows'])):
            A_row = np.zeros(M, dtype=float)
            for col, gj in enumerate(local_cols):
                A_row[gj] = scale * float(mat[row, col])
            A_rows.append(A_row)
            b_vals.append(np.pi / 2 * scale * float(consts[row]))

    if not A_rows:
        return np.zeros((0, M), dtype=float), np.zeros(0, dtype=float)
    return np.asarray(A_rows, dtype=float), np.asarray(b_vals, dtype=float)


