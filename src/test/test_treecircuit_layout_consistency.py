"""Circuit <-> chart layout consistency of the treecircuit module.

The tree-parameter convention is fixed by the qubit-ordering reducer
(``discard_constant_bits`` for ``reorder=False``, ``best_bit_reordering_bnb``
for ``reorder=True``).  The circuit builder, the chart maps (c2p / p2c), the
closed-form metric and the singular-corner initialiser must all use the same
reducer, or the same numeric parameter vector silently means different states
on the two sides (see ``_select_reducer``).

These tests lock that contract:

  1. p2c reproduces the statevector of the compiled circuit for the same
     parameters, for both reorder flags;
  2. c2p inverts p2c;
  3. the closed-form inverse-metric diagonal matches the finite-difference
     Fubini-Study pullback of the actual circuit state;
  4. the witness leaf set genuinely distinguishes the two layouts, so that
     tests 1-3 cannot pass by accident of a symmetric leaf set.

The witness is the singlet-CSF preimage set of the LiH CAS(2e,3o) sector
(6 leaves on 6 qubits), for which branch-and-bound reordering is non-identity.
"""

import numpy as np
import pytest

from qiskit.quantum_info import Statevector

from quantumsymmetry.treecircuit import (
    best_bit_reordering_bnb,
    discard_constant_bits,
    circuit,
    make_cartesian_to_polyspherical,
    make_polyspherical_to_cartesian,
    make_inverse_metric,
)

# Leaf sets: (name, n_qubits, leaves)
WITNESS = ("csf-preimages-lih", 6, [9, 10, 12, 18, 20, 36])
LEAF_SETS = [
    WITNESS,
    # {N=2, Sz=0} determinant sector, m=3 (layouts coincide: identity is optimal)
    ("nsz-sector-m3", 6, [9, 10, 12, 17, 18, 20, 33, 34, 36]),
    # an unstructured sparse set
    ("unstructured", 5, [0, 3, 11, 21, 30]),
]


def _statevector(qc, params):
    return Statevector(qc.assign_parameters(params)).data.real


def _rand_params(num, seed):
    return np.random.default_rng(seed).uniform(0.15, 1.35, size=num)


@pytest.mark.parametrize("name,n,leaves", LEAF_SETS)
@pytest.mark.parametrize("reorder", [False, True])
def test_p2c_matches_circuit(name, n, leaves, reorder):
    qc = circuit(n, active_leaves=leaves, reorder=reorder)
    p2c = make_polyspherical_to_cartesian(n, leaves, reorder=reorder)
    for seed in range(3):
        params = _rand_params(qc.num_parameters, seed)
        psi_circ = _statevector(qc, params)
        psi_chart = np.asarray(p2c(active_vals=params), float)
        assert np.linalg.norm(psi_circ - psi_chart) < 1e-12, (name, reorder, seed)


@pytest.mark.parametrize("name,n,leaves", LEAF_SETS)
@pytest.mark.parametrize("reorder", [False, True])
def test_c2p_inverts_p2c(name, n, leaves, reorder):
    p2c = make_polyspherical_to_cartesian(n, leaves, reorder=reorder)
    c2p = make_cartesian_to_polyspherical(n, leaves, reorder=reorder)
    k = len(leaves)
    for seed in range(3):
        params = _rand_params(k - 1, seed)
        X = np.asarray(p2c(active_vals=params), float)
        back = np.asarray(c2p(X), float)
        assert np.linalg.norm(back - params) < 1e-10, (name, reorder, seed)


@pytest.mark.parametrize("name,n,leaves", LEAF_SETS)
@pytest.mark.parametrize("reorder", [False, True])
def test_metric_matches_circuit_pullback(name, n, leaves, reorder):
    """diag entries of g equal Re<d_i psi|d_j psi> of the actual circuit."""
    qc = circuit(n, active_leaves=leaves, reorder=reorder)
    inv_metric = make_inverse_metric(n, leaves, reorder=reorder)
    params = _rand_params(qc.num_parameters, 7)
    eps = 1e-6
    cols = []
    for i in range(len(params)):
        pp, pm = params.copy(), params.copy()
        pp[i] += eps
        pm[i] -= eps
        cols.append((_statevector(qc, pp) - _statevector(qc, pm)) / (2 * eps))
    J = np.column_stack(cols)
    g_fd = J.T @ J
    g_closed = np.linalg.inv(np.asarray(inv_metric(active_vals=params), float))
    # off-diagonal of the FD pullback must vanish; diagonal must match
    assert np.max(np.abs(g_fd - np.diag(np.diag(g_fd)))) < 1e-7, (name, reorder)
    assert np.max(np.abs(np.diag(g_fd) - np.diag(g_closed))) < 1e-6, (name, reorder)


def test_witness_set_distinguishes_layouts():
    """The witness leaf set must expose a flag mismatch: with circuit built at
    reorder=True but the chart read at reorder=False, the states differ."""
    name, n, leaves = WITNESS
    bnb = best_bit_reordering_bnb(leaves, n)
    dcb = discard_constant_bits(leaves, n)
    assert bnb[3] != dcb[3], "witness set no longer distinguishes the reducers"

    qc = circuit(n, active_leaves=leaves, reorder=True)
    p2c_wrong = make_polyspherical_to_cartesian(n, leaves, reorder=False)
    params = _rand_params(qc.num_parameters, 11)
    psi_circ = _statevector(qc, params)
    psi_wrong = np.asarray(p2c_wrong(active_vals=params), float)
    assert np.linalg.norm(psi_circ - psi_wrong) > 1e-3
