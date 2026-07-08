"""Validation of the sector-Haar (Fubini--Study) tree sampler.

Three checks, on several pruned-support geometries:

1.  First moment:  E[|psi><psi|] = P / K  (1-design on the sector).
2.  Second moment: E[(|psi><psi|)^{x2}] = (P x P + SWAP_P) / (K (K+1))
    (2-design on the sector -- what fidelity averaging requires).
3.  Tree sampler vs Gaussian oracle: per-coordinate Kolmogorov--Smirnov
    distance of the theta marginals.

All statevectors are reconstructed through the COMPLEX chart so the test
also exercises the chart conventions (theta ordering, omega gauge).
"""

import numpy as np
import pytest

from quantumsymmetry.treecircuit import MinimalCircuit
from quantumsymmetry.treecircuit.sampling import (
    tree_beta_parameters,
    sample_sector_haar,
    sample_sector_haar_oracle,
)

SEED = 20260611


def _supports():
    """Representative supports: dense, particle-number sector, irregular."""
    full = (3, list(range(8)))                       # full 3-qubit space
    pn = (6, None)                                   # (1,1) sector on 3 spatial orbitals
    irregular = (4, [0, 1, 3, 5, 6, 9, 12])          # irregular pruned support
    return {"full3": full, "pn_sector": pn, "irregular4": irregular}


def _make_circuit(name):
    n, support = _supports()[name]
    if support is None:
        return MinimalCircuit.from_particle_number(3, (1, 1), complex=True)
    return MinimalCircuit(n, sorted(set(support)), complex=True)


def _sample_states(mc, n_samples, sampler, rng):
    """Reconstruct statevectors restricted to the support (K-dim vectors)."""
    theta, omega = sampler(mc.num_qubits, mc.support, n_samples, rng=rng)
    out = np.empty((n_samples, len(mc.support)), dtype=complex)
    for s in range(n_samples):
        psi = mc.statevector(theta[s], omega[s])
        out[s] = psi[mc.support]
    return out


@pytest.mark.parametrize("name", ["full3", "pn_sector", "irregular4"])
def test_states_normalised_and_supported(name):
    mc = _make_circuit(name)
    rng = np.random.default_rng(SEED)
    theta, omega = sample_sector_haar(mc.num_qubits, mc.support, 50, rng=rng)
    for s in range(50):
        psi = mc.statevector(theta[s], omega[s])
        assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-12)
        off = np.delete(psi, mc.support)
        if off.size:
            assert np.max(np.abs(off)) < 1e-12


@pytest.mark.parametrize("name", ["full3", "pn_sector", "irregular4"])
def test_first_moment(name):
    mc = _make_circuit(name)
    K = len(mc.support)
    rng = np.random.default_rng(SEED)
    N = 20000
    vecs = _sample_states(mc, N, sample_sector_haar, rng)
    rho = np.einsum('si,sj->ij', vecs, vecs.conj()) / N
    err = np.max(np.abs(rho - np.eye(K) / K))
    # CLT scale ~ 1/(K sqrt(N)); allow a generous factor.
    assert err < 8.0 / (K * np.sqrt(N)), f"first-moment err {err:.2e}"


@pytest.mark.parametrize("name", ["full3", "pn_sector", "irregular4"])
def test_second_moment(name):
    mc = _make_circuit(name)
    K = len(mc.support)
    rng = np.random.default_rng(SEED)
    N = 20000
    vecs = _sample_states(mc, N, sample_sector_haar, rng)
    # E[(psi psi^dag)^{x2}] on the K-dim sector.
    M = np.zeros((K * K, K * K), dtype=complex)
    for s in range(N):
        v = vecs[s]
        vv = np.outer(v, v).reshape(K * K)
        M += np.outer(vv, vv.conj())
    M /= N
    eye = np.eye(K)
    swap = np.einsum('ij,kl->ikjl', eye, eye).transpose(0, 1, 3, 2)
    target = (np.einsum('ij,kl->ikjl', eye, eye) + swap).reshape(K * K, K * K)
    target = target / (K * (K + 1))
    err = np.max(np.abs(M - target))
    assert err < 10.0 / (K * np.sqrt(N)), f"second-moment err {err:.2e}"


@pytest.mark.parametrize("name", ["full3", "pn_sector", "irregular4"])
def test_tree_matches_oracle_marginals(name):
    """KS distance between tree-sampler and oracle theta marginals."""
    mc = _make_circuit(name)
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED + 1)
    N = 8000
    th_tree, _ = sample_sector_haar(mc.num_qubits, mc.support, N, rng=rng1)
    th_orac, _ = sample_sector_haar_oracle(mc.num_qubits, mc.support, N,
                                           rng=rng2)
    n_active = th_tree.shape[1]
    assert th_orac.shape[1] == n_active
    # Two-sample KS statistic per coordinate.
    for j in range(n_active):
        a = np.sort(th_tree[:, j])
        b = np.sort(th_orac[:, j])
        grid = np.concatenate([a, b])
        cdf_a = np.searchsorted(a, grid, side='right') / N
        cdf_b = np.searchsorted(b, grid, side='right') / N
        ks = np.max(np.abs(cdf_a - cdf_b))
        # 99.9% two-sample KS critical value ~ 1.95*sqrt(2/N).
        assert ks < 1.95 * np.sqrt(2.0 / N), f"coord {j}: KS={ks:.3f}"


def test_beta_parameters_sum():
    """Right+left active-leaf counts at the root equal K."""
    mc = _make_circuit("irregular4")
    params, topo = tree_beta_parameters(mc.num_qubits, mc.support)
    assert params, "no active nodes?"
    n_r, n_l = params[0]
    assert n_r + n_l == topo['n_leaves']
    assert all(r >= 1 and l >= 1 for r, l in params)
