# Tests for PeriodicEncoding (BYO-Hamiltonian path).
#
# Reference is the lowest eigenvalue of the Jordan-Wigner Hamiltonian
# restricted to the target Boolean symmetry sector. PySCF crystal drivers
# are intentionally not exercised here: they are too slow for unit tests
# and are covered by the CAS/crystals/ benchmarks.

import numpy as np
from numpy import isclose
from numpy.linalg import eigvalsh
from openfermion import FermionOperator, QubitOperator, jordan_wigner
from openfermion.linalg import get_sparse_operator

from ..quantumsymmetry import PeriodicEncoding
from ..quantumsymmetry.periodic import (
    spin_parity_generators,
    half_translation_generators,
)


# ---------------------------------------------------------------------------
# Tiny Hubbard model in the Bloch basis on a (Z_2)^d torus mesh.
# Spin-orbital indexing (matches openfermion / quantumsymmetry):
#     so_index(band, k_index, sigma) = 2 * (band * n_k + k_index) + sigma
# ---------------------------------------------------------------------------

def _so(band, k, sigma, n_k):
    return 2 * (band * n_k + k) + sigma


def _build_hubbard_bloch(t, U, mesh):
    """1-band Hubbard model on a (Z_2)^d torus, in the Bloch basis.

    `mesh` is a length-3 tuple with entries in {1, 2}, matching the
    convention used by ``half_translation_generators``: active axes have
    size 2; bit ordering follows PySCF's ``make_kpts`` (axis 0 slowest).
    """
    mesh = tuple(int(x) for x in mesh)
    if len(mesh) != 3 or any(m not in (1, 2) for m in mesh):
        raise ValueError(f"mesh must be length-3 with entries in {{1,2}}; got {mesh}")
    n_k = int(np.prod(mesh))
    n_so = 2 * n_k

    # Bit position of axis `ax` inside k_index (PySCF order: axis 0 slowest).
    def _bit(k_index, ax):
        if mesh[ax] != 2:
            return 0
        weight = 1
        for ax2 in range(ax + 1, 3):
            weight *= mesh[ax2]
        return (k_index // weight) % 2

    H = FermionOperator()
    # Hopping diagonal in k: eps(k) = -2 t sum_active_ax (-1)^{m_ax(k)}
    for k in range(n_k):
        eps_k = sum(-2.0 * t * (-1) ** _bit(k, ax) for ax in range(3) if mesh[ax] == 2)
        if eps_k != 0.0:
            for sigma in (0, 1):
                p = _so(0, k, sigma, n_k)
                H += FermionOperator(((p, 1), (p, 0)), eps_k)
    # Hubbard U with bitwise momentum conservation: per active axis
    #   bit(k1) + bit(k2) = bit(k3) + bit(k4) mod 2.
    # Since PySCF mesh axes of size 2 occupy distinct, contiguous bit
    # positions of k_index (LSB = last active axis), this is equivalent to
    # plain XOR on the integer k_index.
    if U != 0.0:
        coeff = U / n_k
        for k1 in range(n_k):
            for k2 in range(n_k):
                for k3 in range(n_k):
                    k4 = k1 ^ k2 ^ k3
                    p1 = _so(0, k1, 0, n_k)
                    p2 = _so(0, k2, 0, n_k)
                    p3 = _so(0, k3, 1, n_k)
                    p4 = _so(0, k4, 1, n_k)
                    H += FermionOperator(((p1, 1), (p3, 1), (p4, 0), (p2, 0)), coeff)
    return H, n_so


def _sector_lowest(fermion_hamiltonian, n_so, generators, signs):
    """Lowest eigenvalue of the JW Hamiltonian restricted to the sector
    where each Z-string generator takes its prescribed +/-1 sign."""
    H = get_sparse_operator(jordan_wigner(fermion_hamiltonian),
                            n_qubits=n_so).toarray()
    dim = H.shape[0]
    keep = np.ones(dim, dtype=bool)
    for gen, sgn in zip(generators, signs):
        z_mask = np.array([1 if gen[q] == -1 else 0 for q in range(n_so)], dtype=int)
        eigs = np.empty(dim, dtype=int)
        for x in range(dim):
            bits = np.array([(x >> q) & 1 for q in range(n_so)], dtype=int)
            eigs[x] = 1 if (int(np.dot(z_mask, bits)) % 2) == 0 else -1
        keep &= (eigs == int(sgn))
    return float(eigvalsh(H[np.ix_(keep, keep)])[0])


def _reduced_lowest(qubit_op, n_qubits):
    if n_qubits == 0:
        return float(qubit_op.terms.get((), 0.0).real)
    H = get_sparse_operator(qubit_op, n_qubits=n_qubits).toarray()
    return float(eigvalsh(H)[0])


# ---------------------------------------------------------------------------
# Standard generator set for a half-filled 1-band Hubbard model
# at the Gamma sector (P_up = P_dn = (-1)^N_sigma, half-translations = +1).
# ---------------------------------------------------------------------------

def _half_filled_gens(mesh, n_so, nelectron_up, nelectron_down):
    g_spin, l_spin = spin_parity_generators(n_so)
    g_trans, l_trans = half_translation_generators(mesh, n_bands=1)
    gens = g_spin + g_trans
    signs = [(-1) ** nelectron_up, (-1) ** nelectron_down] + [1] * len(g_trans)
    labels = l_spin + l_trans
    return gens, signs, labels


# ===========================================================================
# Tests
# ===========================================================================

def test_hubbard_2sites_1d():
    """Smallest non-trivial: 1-band Hubbard on a 2-site 1D cell.
    n_so=4, generators=3 -> reduced to 1 qubit."""
    mesh = (1, 1, 2)
    H_fop, n_so = _build_hubbard_bloch(t=1.0, U=2.0, mesh=mesh)
    nelectron_up = nelectron_down = 1  # half filling
    gens, signs, labels = _half_filled_gens(mesh, n_so, nelectron_up, nelectron_down)

    enc = PeriodicEncoding(
        fermion_hamiltonian=H_fop, nspinorbital=n_so,
        nelectron_up=nelectron_up, nelectron_down=nelectron_down,
        symmetry_generators=gens, signs=signs,
        symmetry_generator_labels=labels, name='Hubbard 1D 2-site',
    )
    n_red = n_so - len(enc.target_qubits)
    e_ref = _sector_lowest(H_fop, n_so, gens, signs)
    e_red = _reduced_lowest(enc.hamiltonian, n_red)
    assert isclose(e_red, e_ref, atol=1e-10)


def test_hubbard_2x2_2d():
    """1-band Hubbard on a 2x2 cell. n_so=8, generators=4 -> 4 qubits."""
    mesh = (1, 2, 2)
    H_fop, n_so = _build_hubbard_bloch(t=1.0, U=2.0, mesh=mesh)
    nelectron_up = nelectron_down = 2
    gens, signs, labels = _half_filled_gens(mesh, n_so, nelectron_up, nelectron_down)

    enc = PeriodicEncoding(
        fermion_hamiltonian=H_fop, nspinorbital=n_so,
        nelectron_up=nelectron_up, nelectron_down=nelectron_down,
        symmetry_generators=gens, signs=signs,
        symmetry_generator_labels=labels, name='Hubbard 2D 2x2',
    )
    n_red = n_so - len(enc.target_qubits)
    assert n_red == n_so - len(gens)  # all 4 generators independent
    e_ref = _sector_lowest(H_fop, n_so, gens, signs)
    e_red = _reduced_lowest(enc.hamiltonian, n_red)
    assert isclose(e_red, e_ref, atol=1e-10)


def test_periodic_apply_number_operator():
    """`apply` on a single-mode number operator produces the expected
    Z-string projection (mirrors molecular `test_encoding_apply`)."""
    mesh = (1, 1, 2)
    H_fop, n_so = _build_hubbard_bloch(t=1.0, U=2.0, mesh=mesh)
    gens, signs, labels = _half_filled_gens(mesh, n_so, 1, 1)
    enc = PeriodicEncoding(
        fermion_hamiltonian=H_fop, nspinorbital=n_so,
        nelectron_up=1, nelectron_down=1,
        symmetry_generators=gens, signs=signs,
        symmetry_generator_labels=labels,
    )
    # n_0 = 0.5 * (I - Z_0). After projection of all stabilizer qubits the
    # surviving operator is still 0.5*(I - Z_*) on the unique data qubit, or
    # a constant if mode 0 happens to be a stabilizer qubit.
    reduced = enc.apply(FermionOperator('0^ 0'))
    n_red = n_so - len(enc.target_qubits)
    # Sanity: <reduced> equals <n_0> averaged over the target sector.
    H_n = get_sparse_operator(jordan_wigner(FermionOperator('0^ 0')),
                              n_qubits=n_so).toarray()
    if n_red == 0:
        val_red = float(reduced.terms.get((), 0.0).real)
    else:
        val_red = float(eigvalsh(get_sparse_operator(reduced, n_qubits=n_red).toarray())[0])
    # Compare to lowest eigenvalue of n_0 inside the sector.
    dim = 1 << n_so
    keep = np.ones(dim, dtype=bool)
    for gen, sgn in zip(gens, signs):
        z_mask = np.array([1 if gen[q] == -1 else 0 for q in range(n_so)], dtype=int)
        eigs = np.array([1 if (int(np.dot(z_mask, np.array([(x >> q) & 1 for q in range(n_so)]))) % 2) == 0
                         else -1 for x in range(dim)], dtype=int)
        keep &= (eigs == int(sgn))
    val_ref = float(eigvalsh(H_n[np.ix_(keep, keep)])[0])
    assert isclose(val_red, val_ref, atol=1e-10)


def test_periodic_BK_spectrum_invariant():
    """JW vs Bravyi-Kitaev reduction of the same periodic problem must
    give the same spectrum (mirrors molecular test_H2_CAS_quick_BK_..)."""
    mesh = (1, 2, 2)
    H_fop, n_so = _build_hubbard_bloch(t=1.0, U=2.0, mesh=mesh)
    gens, signs, labels = _half_filled_gens(mesh, n_so, 2, 2)
    common = dict(
        fermion_hamiltonian=H_fop, nspinorbital=n_so,
        nelectron_up=2, nelectron_down=2,
        symmetry_generators=gens, signs=signs,
        symmetry_generator_labels=labels,
    )
    enc_jw = PeriodicEncoding(bravyi_kitaev=False, **common)
    enc_bk = PeriodicEncoding(bravyi_kitaev=True, **common)
    n_red = n_so - len(enc_jw.target_qubits)
    H_jw = get_sparse_operator(enc_jw.hamiltonian, n_qubits=n_red).toarray()
    H_bk = get_sparse_operator(enc_bk.hamiltonian, n_qubits=n_red).toarray()
    assert np.allclose(np.sort(eigvalsh(H_jw)), np.sort(eigvalsh(H_bk)))


def test_periodic_qiskit_output():
    """output_format='qiskit' returns a SparsePauliOp with same spectrum."""
    from qiskit.quantum_info import SparsePauliOp
    mesh = (1, 1, 2)
    H_fop, n_so = _build_hubbard_bloch(t=1.0, U=2.0, mesh=mesh)
    gens, signs, labels = _half_filled_gens(mesh, n_so, 1, 1)

    enc_of = PeriodicEncoding(
        fermion_hamiltonian=H_fop, nspinorbital=n_so,
        nelectron_up=1, nelectron_down=1,
        symmetry_generators=gens, signs=signs,
        symmetry_generator_labels=labels,
        output_format='openfermion',
    )
    enc_qk = PeriodicEncoding(
        fermion_hamiltonian=H_fop, nspinorbital=n_so,
        nelectron_up=1, nelectron_down=1,
        symmetry_generators=gens, signs=signs,
        symmetry_generator_labels=labels,
        output_format='qiskit',
    )
    n_red = n_so - len(enc_of.target_qubits)
    H_qk = enc_qk.hamiltonian
    # PauliSumOp wraps SparsePauliOp; either way .to_matrix() works.
    M_qk = np.asarray(H_qk.to_matrix())
    e_qk = float(eigvalsh(M_qk)[0])
    e_of = _reduced_lowest(enc_of.hamiltonian, n_red)
    assert isclose(e_qk, e_of, atol=1e-10)


def test_periodic_pyscf_h2_gamma():
    """End-to-end PySCF-driven path: H2 in a vacuum cell, Gamma-only.

    Exercises the full ``build_periodic_inputs`` pipeline (cell build, KRHF,
    supercell folding, point-group detection, MO-basis Hamiltonian) and
    checks that the reduced Hamiltonian's lowest eigenvalue matches the
    sector ED of the JW Hamiltonian. Slowest test in the file (~10 s),
    intentionally minimal so it stays comparable to the slowest molecular
    tests (O2/C2H4 CAS quick). Larger crystals are covered by
    ``CAS/crystals/`` benchmarks.
    """
    a = np.diag([3.0, 8.0, 8.0])
    atom = [['H', (0.0, 0.0, 0.0)], ['H', (0.7414, 0.0, 0.0)]]
    enc = PeriodicEncoding(
        atom=atom, a=a, basis='sto-3g', kpts=(1, 1, 1), name='H2-gamma',
    )
    n_red = enc.nspinorbital - len(enc.target_qubits)
    e_ref = _sector_lowest(
        enc.fermion_hamiltonian, enc.nspinorbital,
        enc.symmetry_generators, enc.signs,
    )
    e_red = _reduced_lowest(enc.hamiltonian, n_red)
    assert isclose(e_red, e_ref, atol=1e-9)
