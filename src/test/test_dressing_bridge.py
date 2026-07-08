"""Validation of the chemistry bridge ``dressing_generators`` on a molecule.

Builds the symmetry-adapted P<->Q excitation pool for H3+ and checks that it
(a) yields ``A^3 = A`` generators (accepted by ``make_dressing_pool``) and
(b) decouples a real encoded block to machine precision through the generic
``decoupling_surrogate`` workflow.  Skipped when the chemistry stack (``pyscf`` /
``openfermion``) is unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytest.importorskip("openfermion")

from openfermion import get_sparse_operator, QubitOperator
from scipy.optimize import minimize

from quantumsymmetry.encoding import Encoding
from quantumsymmetry.treecircuit import (
    dressing_generators, make_dressing_pool, dressing_diagnostics,
    decoupling_surrogate, MinimalCircuit, sample_sector_haar,
)


def _encoded_hamiltonian(enc, nq):
    h = enc.hamiltonian
    if isinstance(h, QubitOperator):
        return get_sparse_operator(h, n_qubits=nq).toarray().astype(complex)
    return h.to_matrix().astype(complex)


def test_dressing_generators_decouple_h3plus():
    enc = Encoding(atom='H 0 0 0; H 0 0 1.0; H 0.94 0 0.5',
                   basis='sto-3g', charge=1)
    nq = enc.nspinorbital - len(enc.target_qubits)
    dim = 1 << nq
    H = _encoded_hamiltonian(enc, nq)
    P, Q = list(range(dim // 2)), list(range(dim // 2, dim))

    pool = make_dressing_pool(dressing_generators(enc, P, Q))   # validates A^3 = A
    assert len(pool) > 0

    L0 = dressing_diagnostics(H, P, Q, pool, np.zeros(len(pool)))["leakage"]
    assert L0 > 1e-3                                            # genuinely coupled

    mc = MinimalCircuit(nq, P, complex=True)
    th, om = sample_sector_haar(nq, P, 2 * len(P), rng=np.random.default_rng(0))
    chis = [mc.statevector(th[s], om[s]) for s in range(len(th))]
    res = minimize(lambda p: decoupling_surrogate(H, pool, chis, p),
                   np.zeros(len(pool)), jac=True, method="BFGS")
    Lf = dressing_diagnostics(H, P, Q, pool, res.x)["leakage"]
    assert Lf < 1e-6 and Lf < 1e-3 * L0                        # decoupled
