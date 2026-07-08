"""Schrieffer--Wolff dressing layer for the binary-tree ansatz.

A *dressing* is an ordered product of exact rotations

    U_ext(phi) = prod_k exp(-i phi_k A_k),

whose anti-Hermitian generators ``A_k`` satisfy ``A_k^3 = A_k`` (the encoded
symmetry-adapted single/double excitations ``i(T - T^dag)``).  The cubic
identity collapses each exponential to the closed form

    exp(-i phi A) = I + (cos phi - 1) A^2 - i sin phi A,

so the dressed state ``U_ext(phi) T(theta) |0>`` -- the (classically simulable)
tree state composed with the dressing -- is prepared exactly.  The dressing is
where genuinely quantum content enters: the relevant expectation values involve
the dressed Hamiltonian ``H~ = U_ext(phi)^dag H U_ext(phi)``.

To decouple a model space ``P`` from its complement ``Q`` (a quantum-native
Schrieffer--Wolff transformation), minimise the Fubini--Study-sampled surrogate
:func:`decoupling_surrogate`, whose cost vanishes exactly at ``Q H~ P = 0`` when the
pool spans the off-block tangent space:

>>> import numpy as np
>>> from scipy.optimize import minimize
>>> from quantumsymmetry import (make_dressing_pool, apply_dressing,
...     decoupling_surrogate, dressing_diagnostics, sample_sector_haar, MinimalCircuit)
>>> # A tangent-complete pool of P<->Q rotation generators i(|q><p| - |p><q|):
>>> def rot(p, q):
...     A = np.zeros((dim, dim), complex); A[q, p], A[p, q] = 1.0, -1.0
...     return 1j * A
>>> pool = make_dressing_pool([rot(p, q) for p in P for q in Q])
>>> # Fubini--Study probe states on the model space P:
>>> mc = MinimalCircuit(n_qubits, P, complex=True)
>>> th, om = sample_sector_haar(n_qubits, P, 2 * len(P))
>>> chis = [mc.statevector(th[s], om[s]) for s in range(len(th))]
>>> res = minimize(lambda phi: decoupling_surrogate(H, pool, chis, phi),
...                [0.0] * len(pool), jac=True)
>>> dressing_diagnostics(H, P, Q, pool, res.x)["leakage"]   # ~ 0

This module depends only on ``numpy``.
"""
from __future__ import annotations

from collections import namedtuple
from typing import List, Sequence

import numpy as np

__all__ = [
    "make_dressing_pool",
    "apply_dressing",
    "dressing_diagnostics",
    "decoupling_surrogate",
]

A3_ATOL = 1e-9       # default A^3 = A validation tolerance
SW_FD_EPS = 1e-4     # default central-difference step for the surrogate gradient

# A pool generator caches its square so the closed-form rotation is one line.
Generator = namedtuple("Generator", ["A", "A2"])


def make_dressing_pool(generators: Sequence[np.ndarray], *, check: bool = True,
                       atol: float = A3_ATOL) -> List[Generator]:
    """Build a dressing pool from anti-Hermitian ``A^3 = A`` generators.

    Each generator may be a dense ``numpy`` array or any object supporting
    ``A @ v``; its square is cached for the closed-form rotation.  With
    ``check=True`` the cubic identity ``A^3 = A`` (which the closed form relies
    on) is verified on a random vector.
    """
    rng = np.random.default_rng(0)
    pool = []
    for k, A in enumerate(generators):
        A2 = A @ A
        if check:
            v = rng.standard_normal(A.shape[0]) + 1j * rng.standard_normal(A.shape[0])
            Av = A @ v
            dev = np.linalg.norm(A2 @ Av - Av) / (np.linalg.norm(Av) + 1e-30)
            if dev > atol:
                raise ValueError(
                    f"generator {k} violates A^3 = A (relative deviation "
                    f"{dev:.2e} > {atol:.0e})")
        pool.append(Generator(A, A2))
    return pool


def _factor_order(n_phi: int, symmetric: bool):
    """Factor sequence ``[(generator index, angle scale), ...]`` for ``U_ext``.

    Plain: first-order Trotterization of ``exp(S)``.  Symmetric: the
    second-order (forward then reversed half-angle) product.
    """
    if symmetric:
        return ([(k, 0.5) for k in range(n_phi)]
                + [(k, 0.5) for k in reversed(range(n_phi))])
    return [(k, 1.0) for k in range(n_phi)]


def _rotate(g: Generator, angle: float, psi: np.ndarray) -> np.ndarray:
    """``exp(-i angle A) psi`` for an ``A^3 = A`` generator."""
    c, s = np.cos(angle), np.sin(angle)
    return psi + (c - 1.0) * (g.A2 @ psi) - 1j * s * (g.A @ psi)


def apply_dressing(pool: Sequence[Generator], phi: np.ndarray, psi: np.ndarray,
                   *, symmetric: bool = False) -> np.ndarray:
    """``U_ext(phi) psi`` as the ordered product of exact rotations."""
    psi = np.asarray(psi, dtype=complex)
    for k, scale in _factor_order(len(pool), symmetric):
        psi = _rotate(pool[k], scale * phi[k], psi)
    return psi


def _apply_dressing_dagger(pool, phi, psi, symmetric=False) -> np.ndarray:
    """``U_ext(phi)^dag psi``: reversed factor order, negated angles (internal)."""
    psi = np.asarray(psi, dtype=complex)
    for k, scale in reversed(_factor_order(len(pool), symmetric)):
        psi = _rotate(pool[k], -scale * phi[k], psi)
    return psi


def dressing_diagnostics(H: np.ndarray, P: Sequence[int], Q: Sequence[int],
                         pool: Sequence[Generator], phi: np.ndarray, *,
                         symmetric: bool = False) -> dict:
    """Held-out Schrieffer--Wolff diagnostics of the dressed Hamiltonian.

    With ``H~ = U_ext(phi)^dag H U_ext(phi)``, returns a dict with the
    block-off-diagonal ``leakage = ||Q H~ P||_F^2 / |P|``, the effective model
    block ``effective_block = P H~ P``, its (Hermitised) ``effective_spectrum``
    and lowest eigenvalue ``gs_energy``.  These are classical diagnostics not
    used by :func:`decoupling_surrogate`.
    """
    P_idx = np.asarray(P, dtype=int)
    Q_idx = np.asarray(Q, dtype=int)
    dim, K_P = H.shape[0], P_idx.size
    cols = np.zeros((dim, K_P), dtype=complex)
    for i, p in enumerate(P_idx):
        e = np.zeros(dim, dtype=complex)
        e[p] = 1.0
        v = apply_dressing(pool, phi, e, symmetric=symmetric)
        cols[:, i] = _apply_dressing_dagger(pool, phi, H @ v, symmetric=symmetric)
    block = cols[P_idx, :]
    spectrum = np.linalg.eigvalsh(0.5 * (block + block.conj().T))
    return {
        "leakage": float(np.linalg.norm(cols[Q_idx, :]) ** 2 / K_P),
        "effective_block": block,
        "effective_spectrum": spectrum,
        "gs_energy": float(spectrum[0]),
    }


def _probe_responses(H, pool, chi, phi, symmetric) -> np.ndarray:
    """Per-probe responses ``g_mu = -2 Im <U^dag H U chi | A_mu chi>``.

    On hardware these are the four-term shift-rule derivatives (``A^3 = A``) of
    the energy of ``U(phi) e^{i s A_mu} |chi>`` at ``s = 0``.
    """
    w = apply_dressing(pool, phi, chi, symmetric=symmetric)
    w = _apply_dressing_dagger(pool, phi, H @ w, symmetric=symmetric)
    return np.array([-2.0 * np.vdot(w, g.A @ chi).imag for g in pool])


def _surrogate_cost(H, pool, chis, phi, symmetric) -> float:
    """Batch surrogate cost ``mean_chi sum_mu g_mu^2``."""
    G = np.array([_probe_responses(H, pool, chi, phi, symmetric) for chi in chis])
    return float(np.mean(np.sum(G * G, axis=1)))


def decoupling_surrogate(H: np.ndarray, pool: Sequence[Generator],
                         chis: Sequence[np.ndarray], phi: np.ndarray, *,
                         symmetric: bool = False, fd_eps: float = SW_FD_EPS):
    """Fubini--Study-sampled Schrieffer--Wolff surrogate cost and gradient.

    For a batch of model-space probe states ``chis`` (sector-Haar states on the
    model space, e.g. via :func:`sample_sector_haar`), evaluates

        C_grad(phi) = mean_chi  sum_mu | d/dphi_mu <chi| H~ |chi> |^2,

    where ``H~ = U_ext(phi)^dag H U_ext(phi)`` and each per-sample response is
    the parameter-shift energy gradient of the dressing acting on ``chi``.
    ``C_grad`` vanishes exactly at the Schrieffer--Wolff decoupling
    ``Q H~ P = 0`` when the pool spans the off-block tangent space.

    Returns ``(cost, grad)`` -- the value and (coordinate central-difference)
    gradient -- an optimiser-agnostic oracle to hand to any minimiser, e.g.
    ``scipy.optimize.minimize(fun, x0, jac=True)``.
    """
    phi = np.asarray(phi, dtype=float)
    chis = [np.asarray(c, dtype=complex) for c in chis]
    cost = _surrogate_cost(H, pool, chis, phi, symmetric)
    grad = np.empty(len(phi))
    for k in range(len(phi)):
        step = np.zeros(len(phi))
        step[k] = fd_eps
        cp = _surrogate_cost(H, pool, chis, phi + step, symmetric)
        cm = _surrogate_cost(H, pool, chis, phi - step, symmetric)
        grad[k] = (cp - cm) / (2.0 * fd_eps)
    return cost, grad
