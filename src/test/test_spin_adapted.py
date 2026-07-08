"""Tests for the exact total-spin adaptation (treecircuit.spin).

Dependency-light: numpy + qiskit only (random sector Hamiltonians, no pyscf).
Covers the sector CSF unitary, the spin-adapted ansatz returned by
``MinimalCircuit.from_particle_number(..., total_spin=S)`` (singlet, triplet
and half-integer doublet), the integrated ``minimize_energy`` driver, and the
determinant-chart (gate-angle) tie utilities.
"""

import numpy as np
import numpy.linalg as la
import pytest

from quantumsymmetry import MinimalCircuit, minimize_energy
from quantumsymmetry.treecircuit import (
    ballot_free_tied,
    ballot_leaders,
    constraint_values,
    sector_amplitude_map,
    sector_csf_unitary,
    solve_tied,
    spin_block_columns,
    spin_constraint_matrix,
    tie_jacobian,
    total_spin_expectation,
    woodbury_inverse_metric,
)
from quantumsymmetry.treecircuit.metric import polyspherical_metric_pruned
from quantumsymmetry.treecircuit.tree import chart_topology


def _random_theta(rng, n):
    """Generic interior chart point (away from corners)."""
    return rng.uniform(0.3, 1.2, size=n)


def _spin_block_reference(H, mc):
    """Lowest eigenvalue of H restricted to the spin-S subspace."""
    Hsec = H[np.ix_(mc.sector, mc.sector)]
    return float(np.min(la.eigvalsh(mc.csf_unitary.T @ Hsec @ mc.csf_unitary)))


def _random_hamiltonian(rng, num_qubits):
    dim = 2 ** num_qubits
    A = rng.normal(size=(dim, dim))
    H = (A + A.T) / 2
    return H / la.norm(H, 2)          # unit spectral norm (well-conditioned NG)


# ---------------------------------------------------------------------------
# Sector CSF unitary
# ---------------------------------------------------------------------------

def test_sector_csf_unitary_orthogonal_and_spin_eigenstructure():
    m, na, nb = 3, 2, 2
    leaves, U, labels = sector_csf_unitary(m, na, nb)
    k = len(leaves)
    assert U.shape == (k, k)
    assert np.allclose(U.T @ U, np.eye(k), atol=1e-12)
    # every column is an S^2 eigenvector with the labelled eigenvalue
    for j, lab in enumerate(labels):
        S = float(lab[0])
        psi = np.zeros(2 ** (2 * m))
        psi[np.asarray(leaves)] = U[:, j]
        assert total_spin_expectation(psi, m) == pytest.approx(
            S * (S + 1), abs=1e-12)


def test_ballot_leaders_count_matches_spin_dimension():
    m, na, nb = 3, 2, 2
    leaves, U, labels = sector_csf_unitary(m, na, nb)
    f0 = len(spin_block_columns(labels, 0))
    assert len(ballot_leaders(leaves, m)) == f0


# ---------------------------------------------------------------------------
# Spin-adapted ansatz: singlet
# ---------------------------------------------------------------------------

def test_singlet_exact_spin_and_chart_roundtrip():
    m = 3
    mc = MinimalCircuit.from_particle_number(m, (2, 2), total_spin=0)
    rng = np.random.default_rng(0)
    theta = _random_theta(rng, mc.num_parameters)
    psi = mc.statevector(theta)
    assert la.norm(psi) == pytest.approx(1.0, abs=1e-12)
    assert mc.total_spin_expectation(theta) == pytest.approx(0.0, abs=1e-12)
    # chart round trip
    assert np.allclose(mc.parameters(psi), theta, atol=1e-10)
    # parameter count = (number of spin-0 CSFs) - 1
    _leaves, _U, labels = sector_csf_unitary(m, 2, 2)
    assert mc.num_parameters == len(spin_block_columns(labels, 0)) - 1


def test_singlet_inverse_metric_is_diagonal():
    mc = MinimalCircuit.from_particle_number(3, (2, 2), total_spin=0)
    rng = np.random.default_rng(1)
    G = mc.inverse_metric(_random_theta(rng, mc.num_parameters))
    assert np.allclose(G, np.diag(np.diag(G)), atol=1e-14)


def test_singlet_circuit_realisation_matches_statevector():
    mc = MinimalCircuit.from_particle_number(3, (2, 2), total_spin=0)
    rng = np.random.default_rng(2)
    theta = _random_theta(rng, mc.num_parameters)
    psi = mc.statevector(theta)
    # the determinant-tree gate angles realise the same state
    det_theta = mc.circuit_parameters(theta)
    realised = MinimalCircuit.statevector(mc, det_theta)
    # global sign is a phase
    overlap = abs(float(np.dot(realised, psi)))
    assert overlap == pytest.approx(1.0, abs=1e-12)


def test_singlet_hartree_fock_start():
    mc = MinimalCircuit.from_particle_number(3, (2, 2), total_spin=0)
    theta_hf = mc.hartree_fock_parameters()
    psi = mc.statevector(theta_hf)
    hf = mc.hartree_fock_statevector()
    assert abs(float(np.dot(psi, hf))) == pytest.approx(1.0, abs=1e-12)


def test_singlet_vqe_reaches_spin_block_floor():
    m = 3
    mc = MinimalCircuit.from_particle_number(m, (2, 2), total_spin=0)
    rng = np.random.default_rng(3)
    H = _random_hamiltonian(rng, mc.num_qubits)
    e_ref = _spin_block_reference(H, mc)

    def energy_fn(t):
        psi = mc.statevector(t)
        return float(psi @ (H @ psi))

    result = minimize_energy(mc, None, energy_fn=energy_fn, tol=1e-12)
    assert result.energy == pytest.approx(e_ref, abs=1e-8)
    assert mc.total_spin_expectation(result.optimal_parameters) == pytest.approx(
        0.0, abs=1e-10)


def test_estimator_path_binds_lowered_circuit():
    """The default (Hamiltonian) energy path binds the determinant-tree
    circuit through the classical lowering: one energy evaluation via the
    estimator equals the statevector expectation."""
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp

    m = 3
    mc = MinimalCircuit.from_particle_number(m, (2, 2), total_spin=0)
    rng = np.random.default_rng(9)
    theta = _random_theta(rng, mc.num_parameters)
    ham = SparsePauliOp.from_sparse_list(
        [("Z", [0], 0.7), ("ZZ", [1, 4], -0.4), ("XX", [2, 5], 0.3),
         ("YY", [0, 3], 0.2)], num_qubits=mc.num_qubits)
    est_val = float(np.real(
        StatevectorEstimator().run(
            [(mc.bound_circuit(theta), ham)]).result()[0].data.evs))
    psi = mc.statevector(theta)
    sv_val = float(np.real(psi @ (ham.to_matrix() @ psi)))
    assert est_val == pytest.approx(sv_val, abs=1e-10)


# ---------------------------------------------------------------------------
# General spin: triplet and half-integer doublet
# ---------------------------------------------------------------------------

def test_triplet_exact_spin_and_vqe():
    m, na, nb = 4, 3, 1                       # Sz = 1 highest-weight block
    mc = MinimalCircuit.from_particle_number(m, (na, nb), total_spin=1)
    rng = np.random.default_rng(4)
    theta = _random_theta(rng, mc.num_parameters)
    assert mc.total_spin_expectation(theta) == pytest.approx(2.0, abs=1e-12)
    H = _random_hamiltonian(rng, mc.num_qubits)
    e_ref = _spin_block_reference(H, mc)

    def energy_fn(t):
        psi = mc.statevector(t)
        return float(psi @ (H @ psi))

    result = minimize_energy(mc, None, energy_fn=energy_fn, tol=1e-12)
    assert result.energy == pytest.approx(e_ref, abs=1e-8)
    assert mc.total_spin_expectation(result.optimal_parameters) == pytest.approx(
        2.0, abs=1e-10)


def test_doublet_half_integer_spin():
    m, na, nb = 3, 2, 1                       # Sz = 1/2
    mc = MinimalCircuit.from_particle_number(m, (na, nb), total_spin=0.5)
    rng = np.random.default_rng(5)
    theta = _random_theta(rng, mc.num_parameters)
    assert mc.total_spin_expectation(theta) == pytest.approx(0.75, abs=1e-12)


def test_total_spin_requires_highest_weight():
    with pytest.raises(ValueError, match="highest-weight"):
        MinimalCircuit.from_particle_number(3, (2, 2), total_spin=1)


# ---------------------------------------------------------------------------
# Determinant-chart (gate-angle) tie utilities
# ---------------------------------------------------------------------------

def test_ballot_free_tied_counts_and_lowered_angles_satisfy_constraints():
    m = 3
    mc = MinimalCircuit.from_particle_number(m, (2, 2), total_spin=0)
    support = mc.support
    free_idx, tied_idx, leaders = ballot_free_tied(mc.num_qubits, support, m)
    f = len(mc.embedding)
    assert len(leaders) == f
    assert len(free_idx) == f - 1
    assert len(free_idx) + len(tied_idx) == len(support) - 1

    # the gate angles realising a CSF-chart point satisfy the S+ constraints
    phi = spin_constraint_matrix(m, support)
    amp_fn = sector_amplitude_map(mc.num_qubits, support)
    rng = np.random.default_rng(6)
    det_theta = mc.circuit_parameters(_random_theta(rng, mc.num_parameters))
    assert np.max(np.abs(constraint_values(det_theta, phi, amp_fn))) < 1e-10


def test_solve_tied_recovers_spin_manifold():
    m = 3
    mc = MinimalCircuit.from_particle_number(m, (2, 2), total_spin=0)
    support = mc.support
    free_idx, tied_idx, _ = ballot_free_tied(mc.num_qubits, support, m)
    phi = spin_constraint_matrix(m, support)
    amp_fn = sector_amplitude_map(mc.num_qubits, support)
    rng = np.random.default_rng(7)
    det_theta = mc.circuit_parameters(_random_theta(rng, mc.num_parameters))
    # perturb the tied angles off the manifold, then re-solve (free held)
    perturbed = det_theta.copy()
    perturbed[tied_idx] += rng.uniform(-0.2, 0.2, size=len(tied_idx))
    solved = solve_tied(perturbed, tied_idx, phi, amp_fn)
    assert np.max(np.abs(constraint_values(solved, phi, amp_fn))) < 1e-12
    assert np.allclose(solved[free_idx], det_theta[free_idx])


def test_woodbury_inverse_metric_inverts_induced_metric():
    m = 3
    mc = MinimalCircuit.from_particle_number(m, (2, 2), total_spin=0)
    support = mc.support
    free_idx, tied_idx, _ = ballot_free_tied(mc.num_qubits, support, m)
    phi = spin_constraint_matrix(m, support)
    amp_fn = sector_amplitude_map(mc.num_qubits, support)
    rng = np.random.default_rng(8)
    det_theta = mc.circuit_parameters(_random_theta(rng, mc.num_parameters))

    topo = chart_topology(mc.num_qubits, support)
    w = np.asarray(polyspherical_metric_pruned(
        topo["active_params"], topo["fixed_params"], det_theta,
        topo["fixed_params_vals"])).diagonal()
    J = tie_jacobian(det_theta, free_idx, tied_idx, phi, amp_fn)
    g_f = np.diag(w[free_idx]) + J.T @ np.diag(w[tied_idx]) @ J
    gi_f = woodbury_inverse_metric(w, free_idx, tied_idx, J)
    assert np.allclose(gi_f @ g_f, np.eye(len(free_idx)), atol=1e-9)


# ---------------------------------------------------------------------------
# Driver guards and plain-factory support kwarg
# ---------------------------------------------------------------------------

def test_unsupported_optimizers_raise():
    mc = MinimalCircuit.from_particle_number(3, (2, 2), total_spin=0)
    with pytest.raises(NotImplementedError, match="spin adaptation"):
        minimize_energy(mc, None, energy_fn=lambda t: 0.0, optimizer="rotosolve")


def test_plain_factory_accepts_support_restriction():
    m = 2
    full = MinimalCircuit.from_particle_number(m, (1, 1))
    sub = MinimalCircuit.from_particle_number(m, (1, 1),
                                              support=full.support[:2])
    assert sub.support == [int(s) for s in full.support[:2]]
    assert sub.hartree_fock_parameters() is not None
