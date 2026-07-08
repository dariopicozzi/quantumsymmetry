"""Pruned Qiskit circuit construction and fast compiled binding.

Assembles the triangular scaffold, removes redundant CNOTs, builds the
pruned ``QuantumCircuit`` and provides the precompiled affine binder.
"""

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from functools import partial
from dataclasses import dataclass
from typing import List, Optional

from .tree import *
from .affine import *


__all__ = [
    'triangular_gate_list',
    'triangular_circuit_list',
    'get_cnot_indices_to_remove',
    'get_pruned_circuit_list',
    'get_qiskit_circuit',
    '_build_pruned_block',
    'circuit',
    'AffineCompiledCircuit',
    'compile_affine',
    'bind_fast',
]


def triangular_gate_list(depth, target, gate='RY'):
    """Build one triangular block directly in final effective-qubit labels.

    ``gate`` is either ``'RY'`` (default, amplitude block) or ``'RZ'`` (phase
    block); the surrounding CNOT scaffold is identical in both cases.
    """
    if depth == 0:
        return [[gate, target]]
    return (
        triangular_gate_list(depth - 1, target, gate)
        + [['CNOT', target, target + depth]]
        + triangular_gate_list(depth - 1, target, gate)
    )

def triangular_circuit_list(n, gate='RY'):
    """Build the full scaffold directly in final effective-qubit order.

    ``gate`` selects the rotation kind (``'RY'`` or ``'RZ'``) emitted at the
    leaves of each triangular block; CNOT positions are unaffected.
    """
    circuit_list = []
    anchor = n - 1
    for q in range(anchor, -1, -1):
        circuit_list += triangular_gate_list(anchor - q, q, gate)
        if q != anchor:
            circuit_list += [['CNOT', q, anchor]]
    return circuit_list

def get_cnot_indices_to_remove(n, circuit_list, rot_indices_to_remove):
    """Identify CNOT gates that cancel after removing the given rotation gates.

    ``rot_indices_to_remove`` collects every rotation index (``RY`` or ``RZ``)
    being pruned across all blocks; the search stops at any *surviving* (i.e.
    not pruned) single-qubit rotation, since such a rotation would block the
    cancellation of two surrounding identical CNOTs.
    """
    cnot_indices_to_remove = []
    for i in range(len(circuit_list)):
        if circuit_list[i][0] != 'CNOT':
            pass
        elif i in cnot_indices_to_remove:
            pass
        else:
            for j in range(1, len(circuit_list) - i):
                if circuit_list[i + j] == circuit_list[i]:
                    cnot_indices_to_remove.append(i)
                    cnot_indices_to_remove.append(j + i)
                    break
                elif circuit_list[i + j][0] in ('RY', 'RZ') and i + j not in rot_indices_to_remove:
                    break
    return cnot_indices_to_remove

def get_pruned_circuit_list(n, circuit_list, rot_indices_to_remove, cnot_indices_to_remove):
    """Return *circuit_list* with the specified rotation and CNOT gates removed."""
    pruned_circuit_list = []
    for i in range(len(circuit_list)):
        if i in rot_indices_to_remove or i in cnot_indices_to_remove:
            pass
        else:
            pruned_circuit_list.append(circuit_list[i])
    return pruned_circuit_list


def get_qiskit_circuit(n, circuit_list, args):
    """Translate a gate list into a Qiskit ``QuantumCircuit``."""
    qc = QuantumCircuit(n)
    c = 0
    for i in range(len(circuit_list)):
        if circuit_list[i][0] == 'RY':
            qc.ry(args[c], circuit_list[i][1])
            c += 1
        elif circuit_list[i][0] == 'RZ':
            qc.rz(args[c], circuit_list[i][1])
            c += 1
        elif circuit_list[i][0] == 'CNOT':
            qc.cx(circuit_list[i][2], circuit_list[i][1])
    return qc

def _build_pruned_block(n, blocks, gate, name):
    """Build a pruned cascade for a single block kind (``'RY'`` or ``'RZ'``).

    Returns the pruned gate list (with ``CNOT`` cancellations applied) and the
    corresponding list of Qiskit angle expressions, in the order in which the
    surviving rotations appear in the gate list.  Used twice from
    :func:`circuit` when ``complex=True`` -- once for the amplitude block and
    once for the phase block -- with the appropriate ``blocks`` and parameter
    name.
    """
    circuit_list = triangular_circuit_list(n, gate=gate)

    rot_indices = [i for i, g in enumerate(circuit_list) if g[0] == gate]

    args = []
    block_args = _block_argument_expressions_qiskit(blocks, name=name)
    rot_indices_to_remove = []
    rot_offset = 0
    for q in range(n - 1, -1, -1):
        block = blocks[q]
        block_size = 2**(n - 1 - q)
        surviving_rows = set(block['surviving_rows'])
        for row in range(block_size):
            if row not in surviving_rows:
                rot_indices_to_remove.append(rot_indices[rot_offset + row])
        args.extend(block_args[q])
        rot_offset += block_size

    cnot_indices_to_remove = get_cnot_indices_to_remove(n, circuit_list, rot_indices_to_remove)
    pruned = get_pruned_circuit_list(n, circuit_list, rot_indices_to_remove, cnot_indices_to_remove)
    return pruned, args


def circuit(n, active_leaves, reorder=False, complex=False):
    """Build a pruned variational quantum circuit.

    Parameters
    ----------
    n : int
        Number of qubits.
    active_leaves : list of int
        Indices of the computational-basis states that require non-zero
        amplitudes.
    reorder : bool, optional
        If ``True``, run the exhaustive branch-and-bound search for the qubit
        permutation that maximises the number of inactive tree nodes. Disabled
        by default (``False``): on structured active sets it gives no CNOT
        reduction over the constant-bit pass and can be exponentially slow.
    complex : bool, optional
        If ``True`` (default ``False``), append a pruned ``R_z`` phase block
        after the ``R_y`` amplitude block.  The returned circuit then carries
        a ``ParameterVector('theta', M)`` followed by a
        ``ParameterVector('omega', K)``, with ``M`` (resp. ``K``) the number
        of active tree parameters under the amplitude (resp. phase)
        classification (see :func:`classify_phase_params`).  Note that
        Qiskit sorts the resulting ``qc.parameters`` alphabetically, so to
        bind values reliably use ``qc.assign_parameters({param: value, ...})``
        keyed by the parameter object rather than by position.

    Returns
    -------
    qiskit.circuit.QuantumCircuit
        A parameterised Qiskit circuit.
    """
    full_n = n
    affine_data = get_affine_blocks(n, active_leaves, reorder=reorder, complex=complex)
    n = affine_data['effective_num_qubits']
    qubits_order = affine_data['qubits_order']
    inactive_qubit_indices = affine_data['inactive_qubit_indices']
    inactive_qubit_values = affine_data['inactive_qubit_values']
    blocks = affine_data['blocks']
    fixed_qubit_circuit_prefix, fixed_qubit_args_prefix = _fixed_qubit_preparation(
        inactive_qubit_indices,
        inactive_qubit_values,
    )

    if n == 0:
        return get_qiskit_circuit(full_n, fixed_qubit_circuit_prefix, fixed_qubit_args_prefix)

    pruned_ry, args_ry = _build_pruned_block(n, blocks, gate='RY', name='theta')

    if complex:
        pruned_rz, args_rz = _build_pruned_block(
            n, affine_data['phase_blocks'], gate='RZ', name='omega',
        )
        combined = pruned_ry + pruned_rz
        all_args = list(fixed_qubit_args_prefix) + args_ry + args_rz
    else:
        combined = pruned_ry
        all_args = list(fixed_qubit_args_prefix) + args_ry

    reordered_circuit_list = _map_effective_qubits(
        combined,
        qubits_order,
        inactive_qubit_indices,
        inactive_qubit_values,
    )
    qc = get_qiskit_circuit(n + len(inactive_qubit_indices), reordered_circuit_list, all_args)
    return qc


# ---------------------------------------------------------------------------
# Affine fast-bind: precompile the tree-parameter -> circuit-angle map.
#
# Every parametric ``RY``/``RZ`` angle produced by ``_block_argument_expressions_qiskit``
# is, by construction, an affine form in the global active tree parameters::
#
#     angle_g  =  A[g, :] @ theta  +  b[g]
#
# with ``A`` and ``b`` depending only on the active-leaf set (topology), not on
# the variational state.  The default code path stores these as Qiskit
# ``ParameterExpression``s and re-evaluates them symbolically through symengine
# on every ``qc.assign_parameters(...)`` call -- which dominates per-iteration
# cost in the noisy / hardware sampler paths because each post-pruning RZ angle
# in particular is a dense linear combination of all |S| ``omega``s.
#
# ``compile_affine`` extracts ``(A, b)`` once and returns a fully *numeric*
# template circuit plus per-parametric-gate pointers; ``bind_fast`` then does
# one BLAS matvec and writes the resulting angles into the gates' ``.params``
# in place, bypassing Qiskit's symbolic substitution.  This is purely a
# performance optimisation -- the produced circuit is bit-identical (to
# floating-point precision) to the one returned by ``circuit(...).assign_parameters(...)``.
# ---------------------------------------------------------------------------


@dataclass
class AffineCompiledCircuit:
    """Precompiled affine tree-parameter -> circuit-angle map.

    Attributes
    ----------
    template_qc : QuantumCircuit
        A fully-numeric circuit (no ``ParameterExpression``s).  ``bind_fast``
        mutates this object in place.  The same object is returned on every
        bind; callers needing an independent snapshot must call ``.copy()``.
    A_theta, b_theta : np.ndarray
        Affine map for the ``R_y`` amplitude block.  Shape ``(G_ry, n_theta)``
        and ``(G_ry,)``.  The ``g``-th row corresponds to the ``g``-th
        parametric ``RY`` gate in ``template_qc.data`` (excluding the fixed
        ``RY(pi)`` qubit-preparation prefix).
    A_omega, b_omega : np.ndarray or None
        Affine map for the ``R_z`` phase block, or ``None`` for real circuits.
    ry_gate_indices, rz_gate_indices : list of int
        Indices into ``template_qc.data`` of the parametric ``RY`` (resp.
        ``RZ``) gates, in the same order as the rows of ``A_theta`` (resp.
        ``A_omega``).
    n_theta, n_omega : int
        Number of tree parameters in each block (``n_omega == 0`` when not
        complex).
    """
    template_qc: QuantumCircuit
    A_theta: np.ndarray
    b_theta: np.ndarray
    A_omega: Optional[np.ndarray]
    b_omega: Optional[np.ndarray]
    ry_gate_indices: List[int]
    rz_gate_indices: List[int]
    n_theta: int
    n_omega: int


def compile_affine(n, active_leaves, reorder=False, complex=False):
    """Precompile a tree-parameter -> circuit-angle affine map.

    See :class:`AffineCompiledCircuit` and :func:`bind_fast`.  Construction
    is a one-time cost; the returned object can then be re-bound cheaply on
    every iteration of an optimiser / shift-rule gradient sweep.

    The set of parametric gate slots is identified positionally: every
    ``RY``/``RZ`` gate in the underlying pruned circuit is associated with
    exactly one row of the corresponding affine map, with the fixed
    ``RY(pi)`` qubit-preparation prefix (one per inactive qubit whose
    constant value is ``1``) skipped.  No reliance is placed on Qiskit's
    type discrimination, so constant-only surviving rows (rows where
    ``A[g, :] == 0`` but ``b[g] != 0``) are handled correctly.
    """
    affine_data = get_affine_blocks(n, active_leaves, reorder=reorder, complex=complex)
    blocks = affine_data['blocks']
    n_fixed_prefix = sum(1 for v in affine_data['inactive_qubit_values'] if v == 1)

    A_theta, b_theta = _extract_affine_map(blocks)
    if complex:
        A_omega, b_omega = _extract_affine_map(affine_data['phase_blocks'])
    else:
        A_omega = b_omega = None

    # Build the parameterised circuit, then bind dummy zeros to produce a
    # fully-numeric template.  The structural layout of gates (count and
    # order of RYs / RZs) is preserved by ``assign_parameters``.
    qc_symbolic = circuit(n, active_leaves, reorder=reorder, complex=complex)
    if qc_symbolic.parameters:
        binds = {p: 0.0 for p in qc_symbolic.parameters}
        template_qc = qc_symbolic.assign_parameters(binds, inplace=False)
    else:
        template_qc = qc_symbolic.copy()

    ry_indices_all = [i for i, ci in enumerate(template_qc.data) if ci.operation.name == 'ry']
    rz_indices_all = [i for i, ci in enumerate(template_qc.data) if ci.operation.name == 'rz']

    # The first ``n_fixed_prefix`` RYs are the qubit-preparation ``RY(pi)``s
    # produced by ``_fixed_qubit_preparation``; everything after is part of
    # the parameterised ``RY`` block in 1-to-1 order with ``A_theta`` rows.
    ry_indices = ry_indices_all[n_fixed_prefix:]
    rz_indices = rz_indices_all if complex else []

    if len(ry_indices) != A_theta.shape[0]:
        raise RuntimeError(
            f"compile_affine: template has {len(ry_indices)} parametric RY gates "
            f"after the {n_fixed_prefix}-gate fixed prefix, but the affine map "
            f"has {A_theta.shape[0]} rows.  This indicates a drift between the "
            f"pruning loop in _build_pruned_block and _extract_affine_map."
        )
    if complex and len(rz_indices) != A_omega.shape[0]:
        raise RuntimeError(
            f"compile_affine: template has {len(rz_indices)} parametric RZ gates "
            f"but the omega affine map has {A_omega.shape[0]} rows."
        )

    return AffineCompiledCircuit(
        template_qc=template_qc,
        A_theta=A_theta,
        b_theta=b_theta,
        A_omega=A_omega,
        b_omega=b_omega,
        ry_gate_indices=ry_indices,
        rz_gate_indices=rz_indices,
        n_theta=A_theta.shape[1],
        n_omega=A_omega.shape[1] if complex else 0,
    )


def bind_fast(compiled, theta, omega=None):
    """Bind a tree-parameter vector onto ``compiled.template_qc`` in place.

    Per call cost: one ``(G_ry, n_theta)`` BLAS matvec (plus the ``omega``
    counterpart when applicable) followed by ``G_ry`` (+ ``G_rz``) in-place
    Python writes.  No ``ParameterExpression`` substitution is performed.

    Parameters
    ----------
    compiled : AffineCompiledCircuit
        As returned by :func:`compile_affine`.
    theta : array-like
        Tree-parameter vector of length ``compiled.n_theta``.
    omega : array-like, optional
        Phase tree-parameter vector of length ``compiled.n_omega``.  Required
        iff the compiled circuit carries an ``omega`` block.

    Returns
    -------
    QuantumCircuit
        ``compiled.template_qc``, mutated in place.  The same object is
        returned every call.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    if theta.size != compiled.n_theta:
        raise ValueError(
            f"bind_fast: theta has size {theta.size}, expected {compiled.n_theta}"
        )

    data = compiled.template_qc.data
    if compiled.ry_gate_indices:
        ry_angles = compiled.A_theta @ theta + compiled.b_theta
        for k, idx in enumerate(compiled.ry_gate_indices):
            data[idx].operation.params[0] = float(ry_angles[k])

    if compiled.n_omega > 0:
        if omega is None:
            raise ValueError(
                "bind_fast: compiled circuit carries an omega block but omega was not provided"
            )
        omega = np.asarray(omega, dtype=float).ravel()
        if omega.size != compiled.n_omega:
            raise ValueError(
                f"bind_fast: omega has size {omega.size}, expected {compiled.n_omega}"
            )
        if compiled.rz_gate_indices:
            rz_angles = compiled.A_omega @ omega + compiled.b_omega
            for k, idx in enumerate(compiled.rz_gate_indices):
                data[idx].operation.params[0] = float(rz_angles[k])
    elif omega is not None and np.asarray(omega).size > 0:
        raise ValueError(
            "bind_fast: omega supplied but compiled circuit has no omega block"
        )

    return compiled.template_qc


# ---------------------------------------------------------------------------
# Coordinate / metric transformations
# ---------------------------------------------------------------------------

