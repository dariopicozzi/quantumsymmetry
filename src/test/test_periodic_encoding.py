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
    _iter_commuting_pg_subsets,
    _restricted_active_involutions,
    _select_max_rank_periodic_plan,
    _supercell_mo_to_kpt_mo,
    _translated_space_group_ops,
    build_cell,
    fold_to_supercell,
    spin_parity_generators,
    half_translation_generators,
)


def test_translated_coset_orientation_and_odd_mesh_acceptance():
    T = np.array([[0.0, 1.0], [1.0, 0.0]])
    U = np.array([[1.0, 2.0], [3.0, 4.0]])
    reps = [('g', U, False), ('centering', T, True)]
    translated = list(_translated_space_group_ops(
        [('T[0,0,0]', np.eye(2)), ('T[1,0,0]', T)],
        reps,
    ))
    assert translated[0][0] == 'g'
    assert np.array_equal(translated[0][1], U)
    assert translated[1][0] == 'T[1,0,0].g'
    assert np.array_equal(translated[1][1], T @ U)
    fractional = list(_translated_space_group_ops(
        [('T[0,0,0]', np.eye(2))], reps,
        translations=True,
    ))
    assert [label for label, _ in fractional] == ['centering']

    cell = build_cell(
        atom='He 0 0 0', a=np.diag([4.0, 4.0, 4.0]),
        basis={'He': [[0, [1.0, 1.0]]]}, kpts_mesh=(3, 1, 1),
    )
    assert len(cell.make_kpts([3, 1, 1])) == 3


def test_active_restriction_accepts_only_exact_active_involutions():
    """Order is tested after restriction, with active/core leakage rejected."""
    S = np.eye(4)
    C_active = np.eye(4)[:, :2]
    C_empty = np.empty((4, 0))
    C_occ = C_active[:, :1]
    rotation_90 = np.array([[0.0, -1.0], [1.0, 0.0]])

    # Globally order four, but diag(+1,-1) on the invariant active space.
    active_involution = np.block([
        [np.diag([1.0, -1.0]), np.zeros((2, 2))],
        [np.zeros((2, 2)), rotation_90],
    ])
    accepted, _ = _restricted_active_involutions(
        [('active', active_involution)], C_active, C_empty, C_occ, S,
    )
    assert [label for label, _ in accepted] == ['active']

    # A quarter turn on the active space squares to -I and is not accepted.
    active_order_four = np.block([
        [rotation_90, np.zeros((2, 2))],
        [np.zeros((2, 2)), np.eye(2)],
    ])
    accepted, _ = _restricted_active_involutions(
        [('order-four', active_order_four)], C_active, C_empty, C_empty, S,
    )
    assert accepted == []

    # Preserving the active space is insufficient if the frozen core leaks.
    core = np.eye(4)[:, [2]]
    core_leaking = np.eye(4)
    core_leaking[[2, 3]] = core_leaking[[3, 2]]
    accepted, _ = _restricted_active_involutions(
        [('core-leak', core_leaking)], C_active, core, C_occ, S,
    )
    assert accepted == []


def test_active_commutation_and_coupled_translation_deduplication():
    """Compatibility and translation rank are properties of active actions."""
    S = np.eye(5)
    C_active = np.eye(5)[:, :3]
    C_empty = np.empty((5, 0))
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.diag([1.0, -1.0])
    U1 = np.block([
        [np.diag([-1.0, 1.0, 1.0]), np.zeros((3, 2))],
        [np.zeros((2, 3)), X],
    ])
    U2 = np.block([
        [np.diag([1.0, -1.0, 1.0]), np.zeros((3, 2))],
        [np.zeros((2, 3)), Z],
    ])
    ops, _ = _restricted_active_involutions(
        [('u1', U1), ('u2', U2)], C_active, C_empty, C_empty, S,
    )
    assert not np.allclose(U1 @ U2, U2 @ U1)
    assert any(len(subset) == 2 for subset in _iter_commuting_pg_subsets(ops))

    # Two coordinate halves may have the same nontrivial active action;
    # their diagonal square root can supply a second independent action.
    H = np.diag([-1.0, -1.0, 1.0])
    D = np.diag([-1.0, 1.0, -1.0])
    translations, _ = _restricted_active_involutions(
        [('hx', H), ('hy', H.copy()), ('diagonal-root', D)],
        np.eye(3), np.empty((3, 0)), np.empty((3, 0)), np.eye(3),
    )
    assert [label for label, _ in translations] == ['hx', 'diagonal-root']


def test_active_periodic_planner_uses_restricted_boolean_image():
    """The planner combines active-commuting translation and coset actions."""
    rotation_90 = np.array([[0.0, -1.0], [1.0, 0.0]])
    translation = np.block([
        [np.diag([-1.0, 1.0, 1.0]), np.zeros((3, 2))],
        [np.zeros((2, 3)), np.eye(2)],
    ])
    globally_order_four = np.block([
        [np.diag([1.0, -1.0, 1.0]), np.zeros((3, 2))],
        [np.zeros((2, 3)), rotation_90],
    ])
    plan = _select_max_rank_periodic_plan(
        np.eye(5), np.zeros(5), np.zeros(5), np.eye(5),
        [('g', globally_order_four)], [('T', translation)],
        active_indices_0=[0, 1, 2], core_indices_0=[],
    )
    assert plan['total_rank'] == 4
    assert set(plan['active_labels']) == {'T', 'g'}


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

def test_explicit_422_fold_mesh_roundtrip():
    """The fold and inverse phase must use the caller's explicit k-mesh.

    PySCF's automatic inference can identify a (4,2,2) mesh as (3,2,2) when
    bounded by the cell's real-space image estimate.  A one-AO mock KRHF
    object catches that regression without running an SCF calculation.
    """
    from pyscf.pbc import scf as pbcscf

    mesh = (4, 2, 2)
    cell = build_cell(
        atom="He 0 0 0",
        a=np.diag([4.5, 4.5, 4.5]),
        basis={"He": [[0, [1.0, 1.0]]]},
        kpts_mesh=mesh,
    )
    kpts = cell.make_kpts(mesh)
    kmf = pbcscf.KRHF(cell, kpts=kpts, exxdiv="ewald")
    overlap_k = np.asarray(kmf.get_ovlp())
    kmf.mo_coeff = [
        np.asarray([[1.0 / np.sqrt(overlap_k[k, 0, 0])]], dtype=complex)
        for k in range(len(kpts))
    ]
    kmf.mo_energy = [np.asarray([float(k)]) for k in range(len(kpts))]
    kmf.mo_occ = [np.asarray([0.0]) for _ in range(len(kpts))]

    mf_sc, _order, n_k, _n_bands = fold_to_supercell(kmf, kmesh=mesh)
    assert n_k == int(np.prod(mesh))
    assert mf_sc.cell.natm == int(np.prod(mesh)) * cell.natm

    C = np.asarray(mf_sc.mo_coeff)
    overlap_sc = np.asarray(mf_sc.get_ovlp())
    assert np.allclose(C.conj().T @ overlap_sc @ C, np.eye(n_k), atol=1e-10)

    C_at_k = _supercell_mo_to_kpt_mo(mf_sc, kmf, C)
    hcore_k = np.asarray(kmf.get_hcore())
    hcore_from_k = sum(
        C_at_k[k].conj().T @ hcore_k[k] @ C_at_k[k]
        for k in range(n_k)
    )
    hcore_sc = C.conj().T @ np.asarray(mf_sc.get_hcore()) @ C
    assert np.allclose(hcore_from_k, hcore_sc, atol=1e-9)


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
