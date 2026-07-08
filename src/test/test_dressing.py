"""Validation of the Schrieffer--Wolff dressing layer and surrogate oracle.

Checks:

1.  ``make_dressing_pool`` accepts ``A^3 = A`` generators and rejects others.
2.  ``apply_dressing`` reproduces ``expm(-i phi A)`` exactly, is unitary, and is
    inverted by its adjoint (plain and symmetric products).
3.  ``decoupling_surrogate`` cost vanishes iff the block is decoupled, its gradient is a
    descent direction, and feeding the (cost, gradient) oracle to a generic
    optimiser drives the held-out leakage to zero on a tangent-complete toy.
"""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import minimize

from quantumsymmetry import (
    MinimalCircuit,
    sample_sector_haar,
    make_dressing_pool,
    apply_dressing,
    dressing_diagnostics,
    decoupling_surrogate,
)
from quantumsymmetry.treecircuit.dressing import _apply_dressing_dagger

SEED = 20260623


def _rot(dim, p, q):
    """Off-block ``A^3 = A`` rotation generator ``i(|q><p| - |p><q|)``."""
    A = np.zeros((dim, dim), dtype=complex)
    A[q, p], A[p, q] = 1.0, -1.0
    return 1j * A


def _probes(n, P, n_samples, rng):
    """Fubini--Study probe states on the model space ``P``."""
    mc = MinimalCircuit(n, P, complex=True)
    th, om = sample_sector_haar(n, P, n_samples, rng=rng)
    return [mc.statevector(th[s], om[s]).astype(complex) for s in range(n_samples)]


def _toy(coupled=True, seed=SEED):
    """3-qubit toy: sector S = {0,1,2,3}, P = {0,1}, Q = {2,3}, tangent-complete pool."""
    n, dim = 3, 8
    P, Q, S = [0, 1], [2, 3], [0, 1, 2, 3]
    rng = np.random.default_rng(seed)
    block = rng.standard_normal((4, 4))
    block = block + block.T
    if not coupled:
        block[0:2, 2:4] = block[2:4, 0:2] = 0.0
    H = np.zeros((dim, dim))
    H[np.ix_(S, S)] = block
    pool = make_dressing_pool([_rot(dim, p, q) for p in P for q in Q])
    return dict(n=n, dim=dim, P=P, Q=Q, H=H.astype(complex), pool=pool)


def test_make_dressing_pool_rejects_non_cubic():
    make_dressing_pool([_rot(8, 0, 2)])                      # ok
    bad = np.zeros((8, 8), dtype=complex)
    bad[0, 1] = 0.7                                           # A^3 != A
    with pytest.raises(ValueError):
        make_dressing_pool([bad])


def test_apply_dressing_matches_expm():
    A = _rot(8, 1, 4)
    pool = make_dressing_pool([A])
    rng = np.random.default_rng(SEED)
    psi = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    for phi in (0.3, -1.1, 2.0):
        assert np.allclose(apply_dressing(pool, np.array([phi]), psi),
                           expm(-1j * phi * A) @ psi, atol=1e-12)


def test_apply_dressing_unitary_and_adjoint():
    t = _toy()
    pool, dim = t["pool"], t["dim"]
    rng = np.random.default_rng(SEED)
    phi = rng.standard_normal(len(pool))
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    for symmetric in (False, True):
        out = apply_dressing(pool, phi, psi, symmetric=symmetric)
        assert np.isclose(np.linalg.norm(out), np.linalg.norm(psi), atol=1e-12)
        back = _apply_dressing_dagger(pool, phi, out, symmetric=symmetric)
        assert np.allclose(back, psi, atol=1e-12)


def test_decoupling_surrogate_cost_zero_iff_decoupled():
    rng = np.random.default_rng(SEED)
    chis = _probes(3, [0, 1], 8, rng)
    zero = np.zeros(4)
    decoupled, coupled = _toy(coupled=False), _toy(coupled=True)
    assert decoupling_surrogate(decoupled["H"], decoupled["pool"], chis, zero)[0] < 1e-18
    assert decoupling_surrogate(coupled["H"], coupled["pool"], chis, zero)[0] > 1e-3


def test_decoupling_surrogate_gradient_is_descent():
    rng = np.random.default_rng(SEED)
    t = _toy(coupled=True)
    chis = _probes(t["n"], t["P"], 8, rng)
    phi = 0.1 * rng.standard_normal(len(t["pool"]))
    cost, grad = decoupling_surrogate(t["H"], t["pool"], chis, phi)
    assert np.linalg.norm(grad) > 1e-6
    moved, _ = decoupling_surrogate(t["H"], t["pool"], chis, phi - 1e-3 * grad)
    assert moved < cost


def test_decoupling_surrogate_decouples_with_generic_optimiser():
    rng = np.random.default_rng(SEED)
    t = _toy(coupled=True)
    L0 = dressing_diagnostics(t["H"], t["P"], t["Q"], t["pool"],
                              np.zeros(len(t["pool"])))["leakage"]
    assert L0 > 1e-3
    chis = _probes(t["n"], t["P"], 8, rng)                    # fixed batch
    res = minimize(lambda p: decoupling_surrogate(t["H"], t["pool"], chis, p),
                   np.zeros(len(t["pool"])), jac=True, method="BFGS")
    Lf = dressing_diagnostics(t["H"], t["P"], t["Q"], t["pool"], res.x)["leakage"]
    assert Lf < 1e-3 and Lf < 0.1 * L0
