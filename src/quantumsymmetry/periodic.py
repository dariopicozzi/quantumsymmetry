"""Periodic / crystalline extension of quantumsymmetry.

Provides a PySCF-driven path that mirrors the molecular flow
(``run_pyscf`` -> ``get_hamiltonian`` -> ``find_symmetry_generators``)
for crystals.  The pipeline is:

    1. Build a ``pyscf.pbc.gto.Cell`` from user-supplied
       ``atom + a + basis + kpts (+ pseudo)``.
    2. Run k-point restricted Hartree-Fock (``KRHF``) with an
       auto-selected density-fitting backend (FFTDF for all-electron,
       GDF when a pseudopotential is provided).
    3. Fold the k-mesh into a Gamma-point supercell with
       ``pyscf.pbc.tools.k2gamma``.  This gives us a regular Mole-style
       SCF object whose AO->MO transformation we can run through the
       existing ``ao2mo`` machinery, while preserving the (band, k, sigma)
       indexing needed for diagonal half-translation symmetries.
    4. Permute spin-orbital indices into (band, k_index, sigma) order.
    5. Detect the largest abelian Z2^k symmetry subgroup made of
            (a) spin parities P_up, P_dn,
            (b) one half-supercell translation per even-mesh axis,
            (c) Boolean point-group descent for the symmorphic point group.
    6. Build the second-quantised FermionOperator and the +/-1 generator
       arrays in the (band, k_index, sigma) interleaved indexing.

Monkhorst-Pack mesh restrictions: every axis dimension must be either 1
or even.  A Z_2 half-supercell translation along axis i exists if and only
if the mesh size n_i along that axis is even, since translation by half the
supercell period (n_i / 2) * a_i has order 2 in the supercell periodic
boundary conditions and acts as +/-1 on every Bloch state.  Odd mesh sizes
(n_i = 3, 5, ...) would give Z_{n_i} (not Z_2) translation symmetries and
are rejected.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Iterable

import numpy as np
from openfermion import FermionOperator


# ---------------------------------------------------------------------------
# Cell + KRHF
# ---------------------------------------------------------------------------

def build_cell(atom, a, basis, kpts_mesh, pseudo=None, spin=0, charge=0,
               unit='Angstrom', verbose=0, **cell_kwargs):
    """Build and return a `pyscf.pbc.gto.Cell` object.

    Parameters
    ----------
    atom : str | list
        PySCF atom specification (same format as `gto.Mole.atom`).
    a : array_like, shape (3, 3)
        Lattice vectors as rows.
    basis : str | dict
        Gaussian basis set.
    kpts_mesh : tuple[int, int, int]
        Monkhorst-Pack mesh.  Each entry must be 1 or even.  Even-axis
        sizes (2, 4, 6, ...) admit a Z_2 half-supercell translation along
        that axis; odd sizes > 1 give Z_n translations and are not
        supported by the SAE Z_2 framework.
    pseudo : str | dict | None
        PySCF pseudopotential.  When provided the SCF will use GDF.
    """
    from pyscf.pbc import gto as pbcgto

    mesh = tuple(int(x) for x in kpts_mesh)
    if len(mesh) != 3:
        raise ValueError(f"kpts_mesh must be a 3-tuple; got {mesh}")
    for m in mesh:
        if m < 1:
            raise ValueError(f"kpts_mesh entries must be >= 1; got {mesh}")
        if m > 1 and m % 2 != 0:
            raise ValueError(
                f"kpts_mesh entries must be 1 or even (odd axis sizes > 1 "
                f"do not admit Z_2 half-supercell translation symmetries); "
                f"got {mesh}"
            )

    cell = pbcgto.Cell()
    cell.atom = atom
    cell.a = np.asarray(a, dtype=float)
    cell.basis = basis
    if pseudo is not None:
        cell.pseudo = pseudo
    cell.spin = int(spin)
    cell.charge = int(charge)
    cell.unit = unit
    cell.verbose = int(verbose)
    for k, v in cell_kwargs.items():
        setattr(cell, k, v)
    cell.build()
    return cell


def run_krhf(cell, kpts_mesh, df='auto', exxdiv='ewald'):
    """Run KRHF on `cell` over the requested Monkhorst-Pack mesh.

    Returns ``(kmf, kpts)``.
    """
    from pyscf.pbc import scf as pbcscf
    from pyscf.pbc import df as pbcdf

    kpts = cell.make_kpts(list(kpts_mesh))
    kmf = pbcscf.KRHF(cell, kpts=kpts, exxdiv=exxdiv)

    if df == 'auto':
        df = 'GDF' if cell.pseudo else 'FFTDF'
    df = df.upper()
    if df == 'FFTDF':
        kmf.with_df = pbcdf.FFTDF(cell, kpts)
    elif df == 'GDF':
        kmf.with_df = pbcdf.GDF(cell, kpts)
    elif df == 'MDF':
        kmf.with_df = pbcdf.MDF(cell, kpts)
    else:
        raise ValueError(f"Unknown df backend: {df!r}")

    kmf.kernel()
    if not kmf.converged:
        warnings.warn("KRHF did not converge", RuntimeWarning)
    return kmf, kpts


# ---------------------------------------------------------------------------
# Fold to Gamma-point supercell (preserves (band, k) labelling)
# ---------------------------------------------------------------------------

def fold_to_supercell(kmf):
    """Fold a KRHF object into a Gamma-point supercell RHF object.

    Returns
    -------
    mf_sc : pyscf.pbc.scf.RHF
        Gamma-point RHF on the supercell with `mo_coeff` aligned to the
        Bloch (band, k) order produced by ``k2gamma``.
    band_k_order : list[tuple[int, int]]
        For each spatial supercell MO column ``i``, the pair
        ``(band_index, k_index)`` it descended from.
    n_k : int
    n_bands : int
    """
    from pyscf.pbc.tools import k2gamma

    mf_sc = k2gamma.k2gamma(kmf)

    n_k = len(kmf.kpts)
    # Each k-point contributes the same number of spatial MOs (full window).
    n_mo_per_k = kmf.mo_coeff[0].shape[1]
    n_bands = n_mo_per_k

    # k2gamma stacks the Bloch MOs as columns ordered first by k, then by band.
    # Rebuild that ordering explicitly so downstream code does not rely on it.
    band_k_order = [(b, k) for k in range(n_k) for b in range(n_bands)]
    return mf_sc, band_k_order, n_k, n_bands


# ---------------------------------------------------------------------------
# (band, k, sigma) interleaved spin-orbital permutation
# ---------------------------------------------------------------------------

def interleaved_spinorbital_order(band_k_order, n_k, n_bands):
    """Return a permutation `perm` such that
    ``so_index_supercell[i] = perm[so_index_(band, k, sigma)]``.

    The target (band, k, sigma) ordering is::

        so_target(b, k, sigma) = 2 * (b * n_k + k) + sigma

    i.e. interleaved spin (alpha = 0 even, beta = 1 odd).  The supercell
    MO ordering produced by ``fold_to_supercell`` is::

        so_source(col, sigma) = 2 * col + sigma
    """
    n_orb = n_k * n_bands
    perm = np.empty(2 * n_orb, dtype=int)
    # Inverse map: target -> source
    col_of = {bk: i for i, bk in enumerate(band_k_order)}
    for b in range(n_bands):
        for k in range(n_k):
            col = col_of[(b, k)]
            for sigma in (0, 1):
                src = 2 * col + sigma
                tgt = 2 * (b * n_k + k) + sigma
                perm[tgt] = src
    return perm


# ---------------------------------------------------------------------------
# Bloch-basis FermionOperator from supercell SCF
# ---------------------------------------------------------------------------

def _supercell_ao_eri(mf_sc):
    """Return the Gamma-point supercell AO ERI as an (nao,)*4 ndarray.

    GDF's ``get_eri`` can be very slow on supercells (it materialises a
    dense AO ERI from the 3-index DF tensor).  For a pure Gamma point on
    a fully-folded supercell, FFTDF is dramatically faster and gives the
    same numerical answer.  We always build a fresh FFTDF here, ignoring
    whatever DF backend ``mf_sc`` inherited from its k-point parent.

    Pseudopotentials are handled by ``cell.pseudo`` on the FFTDF cell, so
    no additional plumbing is required.
    """
    from pyscf.pbc import df as pbcdf
    cell = mf_sc.cell
    fft = pbcdf.FFTDF(cell, np.zeros((1, 3)))
    eri = fft.get_eri(compact=False).reshape(cell.nao, cell.nao,
                                              cell.nao, cell.nao)
    if np.iscomplexobj(eri):
        eri = eri.real
    return eri


def _build_interleaved_fermion_op(h1, eri, constant=0.0, tol=1e-14):
    """Construct a FermionOperator on ``2*n_mo`` interleaved spinorbitals
    from spatial ``h1`` (n_mo, n_mo) and chemist-order ``eri[p,s,q,r]``
    (n_mo,)*4 by direct dict assembly.

    Equivalent to the explicit nested-loop construction
    ``H = sum_pq h1[p,q] (a^_{p,s}^ a_{q,s}) +
         sum_pqrs 0.5 eri[p,s,q,r] (...)``
    used elsewhere in this module, but ~50x faster because it avoids
    string parsing in ``FermionOperator(str)`` and Python-level
    ``__iadd__`` dict merges.
    """
    terms = {}
    n_mo = h1.shape[0]

    # 1-body: same-spin only.  Indices in interleaved order: 2p (alpha),
    # 2p+1 (beta).  FermionOperator term tuples: ((idx, action), ...).
    for p in range(n_mo):
        for q in range(n_mo):
            v = float(h1[p, q])
            if abs(v) < tol:
                continue
            terms[((2 * p, 1), (2 * q, 0))] = v
            terms[((2 * p + 1, 1), (2 * q + 1, 0))] = v

    # 2-body: chemist (pq|rs) -> physicist a^_p a^_q a_r a_s with
    # coefficient 0.5 * eri[p,s,q,r].  Mirror the molecular code below.
    for p in range(n_mo):
        for q in range(n_mo):
            for r in range(n_mo):
                for s in range(n_mo):
                    v = float(eri[p, s, q, r])
                    if abs(v) < tol:
                        continue
                    half = 0.5 * v
                    # aa
                    terms[((2 * p, 1), (2 * q, 1),
                           (2 * r, 0), (2 * s, 0))] = half
                    # ab + ba (two copies, weight v each, mirroring the
                    # 0.5*v + 0.5*v decomposition used by the explicit loop)
                    terms[((2 * p, 1), (2 * q + 1, 1),
                           (2 * r + 1, 0), (2 * s, 0))] = v
                    # bb
                    terms[((2 * p + 1, 1), (2 * q + 1, 1),
                           (2 * r + 1, 0), (2 * s + 1, 0))] = half

    if abs(constant) > 0.0:
        terms[()] = float(constant) + terms.get((), 0.0)

    H = FermionOperator()
    H.terms = terms
    return H


def _supercell_mo_eri(mf_sc, mo_coeff):
    """Return MO-basis ERI for the supplied MO block, transforming
    directly via the FFTDF ``ao2mo`` path (avoids materialising the full
    AO ERI tensor).

    Returns ``(n_mo,)*4`` real ndarray in chemist (pq|rs) order.
    """
    from pyscf.pbc import df as pbcdf
    cell = mf_sc.cell
    n_mo = mo_coeff.shape[1]
    fft = pbcdf.FFTDF(cell, np.zeros((1, 3)))
    eri_mo = fft.ao2mo(mo_coeff, compact=False)
    if np.iscomplexobj(eri_mo):
        eri_mo = eri_mo.real
    return eri_mo.reshape(n_mo, n_mo, n_mo, n_mo)


def _supercell_mo_to_kpt_mo(mf_sc, kmf, mo_sc):
    """Convert supercell MO coefficients (``(n_ao_sc, n_mo)`` real or
    complex) to per-k primitive AO MO coefficients.

    Uses the k2gamma Bloch unitary
    ``phi^SC_{R,nu} = sum_k conj(phase[R,k]) phi^prim_{k,nu}``,
    where ``phase[R,k] = exp(i k.R) / sqrt(N_R)`` and ``N_R = N_k`` is
    the number of unit cells in the supercell.

    Returns
    -------
    mo_at_k : list of complex ndarray
        Length ``n_k`` list, each of shape ``(n_ao_prim, n_mo)``.
    """
    from pyscf.pbc.tools import k2gamma as k2g
    cell_prim = kmf.cell
    kpts = np.asarray(kmf.kpts)
    _scell, phase = k2g.get_phase(cell_prim, kpts)  # phase: (N_R, n_k)
    n_R, n_k = phase.shape
    n_ao_prim = cell_prim.nao
    n_mo = mo_sc.shape[1]
    if mo_sc.shape[0] != n_R * n_ao_prim:
        raise RuntimeError(
            f"mo_sc has {mo_sc.shape[0]} rows but expected N_R * n_ao_prim = "
            f"{n_R} * {n_ao_prim} = {n_R * n_ao_prim}"
        )
    mo_z = mo_sc.astype(complex, copy=False)
    # mo_at_k[k][nu, p] = sum_R conj(phase[R,k]) * mo_sc[R*n_ao_prim+nu, p]
    mo_resh = mo_z.reshape(n_R, n_ao_prim, n_mo)
    # einsum over R: 'Rnp,Rk->knp' with conj on phase
    mo_at_k_arr = np.einsum('Rnp,Rk->knp', mo_resh, phase.conj())
    return [np.ascontiguousarray(mo_at_k_arr[k]) for k in range(n_k)]


def _kpt_mo_eri(kmf, mo_at_k):
    """Compute MO-basis ERI on the supercell-Gamma MO block, using the
    per-k primitive AO ERIs from KRHF's density fitting.

    Implementation: triple loop over (ki,kj,kk), with kl set by momentum
    conservation, calling ``kmf.with_df.ao2mo`` per (ki,kj,kk,kl) tuple
    and summing into the supercell-Gamma MO ERI.  Backend-agnostic
    (works with both GDF and FFTDF) and memory-safe (each call returns
    only an ``(n_mo,)*4`` block, never the full ``(N_k^3, n_mo^4)``
    workspace that ``ao2mo_7d`` would allocate).

    Returns
    -------
    eri_mo : (n_mo,)*4 real ndarray
    """
    from pyscf.pbc.lib.kpts_helper import get_kconserv

    cell = kmf.cell
    kpts = np.asarray(kmf.kpts)
    n_k = len(kpts)
    n_mo = mo_at_k[0].shape[1]

    kconserv = get_kconserv(cell, kpts)

    eri_mo = np.zeros((n_mo, n_mo, n_mo, n_mo), dtype=complex)
    for ki in range(n_k):
        for kj in range(n_k):
            for kk in range(n_k):
                kl = int(kconserv[ki, kj, kk])
                eri_block = kmf.with_df.ao2mo(
                    [mo_at_k[ki], mo_at_k[kj], mo_at_k[kk], mo_at_k[kl]],
                    kpts=[kpts[ki], kpts[kj], kpts[kk], kpts[kl]],
                    compact=False,
                )
                eri_mo += np.asarray(eri_block).reshape(
                    n_mo, n_mo, n_mo, n_mo)
    eri_mo /= n_k

    if np.max(np.abs(eri_mo.imag)) > 1e-7:
        raise RuntimeError(
            f"k-point MO ERI has non-trivial imaginary part "
            f"(max |Im| = {np.max(np.abs(eri_mo.imag)):.3e}); supercell MOs "
            "may not be real-valued at Gamma."
        )
    return eri_mo.real


def _supercell_mo_eri_via_kpts(mf_sc, kmf, mo_coeff):
    """Drop-in replacement for ``_supercell_mo_eri`` that goes through
    the per-k-point primitive AO ERIs.  Much faster for large supercell
    AO counts because it never materialises a supercell-AO ERI tensor.
    """
    mo_at_k = _supercell_mo_to_kpt_mo(mf_sc, kmf, mo_coeff)
    return _kpt_mo_eri(kmf, mo_at_k)


def _mos_are_bloch_pure(mf_sc, kmf, mo_coeff, tol=1e-6):
    """Return True iff every column of ``mo_coeff`` has weight on a single
    k-point (Bloch-pure to within ``tol`` for the second-largest weight).

    The kpts ERI / J-K route assumes Bloch-pure MOs.  Symmetry-adapted
    MOs that are eigenvectors of only a partial set of half-translation
    operators can be coherent superpositions of two or more k-points
    (e.g. when the lattice has primitive-translation generators that do
    not commute with every PG irrep projector, like FCC/BCC/diamond).
    """
    if mo_coeff.shape[1] == 0:
        return True
    mo_at_k = _supercell_mo_to_kpt_mo(mf_sc, kmf, mo_coeff)
    S_k = np.asarray(kmf.get_ovlp())
    n_k = S_k.shape[0]
    for p in range(mo_coeff.shape[1]):
        weights = np.array([
            float(np.abs(
                mo_at_k[k][:, p].conj() @ S_k[k] @ mo_at_k[k][:, p]
            ).real)
            for k in range(n_k)
        ])
        s = np.sort(weights)[::-1]
        if s[1] > tol:
            return False
    return True


def supercell_fermion_hamiltonian(mf_sc, band_k_order, n_k, n_bands):
    """Build the second-quantised FermionOperator on the supercell, with
    spin-orbital indexing in the (band, k_index, sigma) interleaved order.

    Returns
    -------
    fermion_hamiltonian : FermionOperator
    n_spinorbital : int
    nelectron : int
    """
    from pyscf import ao2mo

    mol = mf_sc.cell  # cell behaves like Mole for ao2mo purposes
    mo_coeff = np.asarray(mf_sc.mo_coeff)
    if mo_coeff.dtype != float:
        if np.max(np.abs(mo_coeff.imag)) > 1e-9:
            raise RuntimeError(
                "Supercell MO coefficients are complex; k2gamma should yield real MOs at Gamma."
            )
        mo_coeff = mo_coeff.real

    # 1-electron core Hamiltonian in supercell AO basis
    hcore_ao = mf_sc.get_hcore()
    if np.iscomplexobj(hcore_ao):
        hcore_ao = hcore_ao.real
    h1_mo = mo_coeff.T @ hcore_ao @ mo_coeff

    # When KRHF is run with exxdiv='ewald' (the PySCF default), the SCF
    # total energy includes a per-electron Madelung shift
    #     E_HF^{ewald} = E_HF^{raw} - (1/2) * N_e * xi_M.
    # We encode this as the operator -(xi_M / 2) * sum_p n_p so that the
    # FermionOperator agrees with kmf.e_tot in *every* particle-number
    # sector (and the global GS does not collapse onto the vacuum).
    if getattr(mf_sc, 'exxdiv', None) == 'ewald' and mol.dimension > 0:
        from pyscf.pbc.tools import madelung as _madelung
        xi_M = float(_madelung(mol, np.zeros((1, 3))))
        h1_mo = h1_mo - 0.5 * xi_M * np.eye(h1_mo.shape[0])

    # 2-electron integrals: transform AO->MO directly via FFTDF.ao2mo
    n_mo = mo_coeff.shape[1]
    eri_mo = _supercell_mo_eri(mf_sc, mo_coeff)

    # Permute spatial orbitals into (band, k) order
    n_orb = n_k * n_bands
    assert n_mo == n_orb
    col_of = {bk: i for i, bk in enumerate(band_k_order)}
    p_perm = np.array([col_of[(b, k)] for b in range(n_bands) for k in range(n_k)],
                      dtype=int)
    h1_mo = h1_mo[np.ix_(p_perm, p_perm)]
    eri_mo = eri_mo[np.ix_(p_perm, p_perm, p_perm, p_perm)]

    # Build FermionOperator with interleaved spin (alpha=0, beta=1)
    H = FermionOperator()
    for p in range(n_orb):
        for q in range(n_orb):
            v = h1_mo[p, q]
            if abs(v) < 1e-14:
                continue
            H += v * FermionOperator(f'{2*p}^ {2*q}')
            H += v * FermionOperator(f'{2*p + 1}^ {2*q + 1}')

    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    v = eri_mo[p, s, q, r]
                    if abs(v) < 1e-14:
                        continue
                    H += 0.5 * v * FermionOperator(f'{2*p}^ {2*q}^ {2*r} {2*s}')
                    H += v * FermionOperator(f'{2*p}^ {2*q + 1}^ {2*r + 1} {2*s}')
                    H += 0.5 * v * FermionOperator(f'{2*p + 1}^ {2*q + 1}^ {2*r + 1} {2*s + 1}')

    H += float(mf_sc.energy_nuc())

    n_spinorbital = 2 * n_orb
    nelectron = int(round(mf_sc.cell.tot_electrons(nkpts=1)))
    return H, n_spinorbital, nelectron


# ---------------------------------------------------------------------------
# Boolean Z2^k symmetry generators
# ---------------------------------------------------------------------------

def _so_index(b, k, sigma, n_k):
    return 2 * (b * n_k + k) + sigma


def spin_parity_generators(n_spinorbital):
    """Return (P_up, P_dn) as +/-1 arrays of length n_spinorbital.

    P_up has -1 on every alpha (even) qubit, P_dn on every beta (odd) qubit.
    """
    P_up = np.ones(n_spinorbital, dtype=int)
    P_dn = np.ones(n_spinorbital, dtype=int)
    P_up[0::2] = -1
    P_dn[1::2] = -1
    return [P_up, P_dn], ['P_up', 'P_dn']


def half_translation_generators(kpts_mesh, n_bands):
    """One half-supercell translation per even-mesh axis.

    For mesh axis `i` with even size n_i, the generator is the Z_2
    half-supercell translation T_{(n_i/2) a_i}, which acts as
    (-1)^{j_i(k)} on every Bloch state of k-index j_i along axis i.
    Returns (gens, labels).
    """
    mesh = tuple(int(x) for x in kpts_mesh)
    n_k = int(np.prod(mesh))

    # An axis admits a Z_2 half-supercell translation iff its mesh size
    # is even.  For an axis of size n_i, the relevant Z_2 generator
    # T_{(n_i/2) a_i} has eigenvalue (-1)^{j_i} on a Bloch state at the
    # j_i-th k-point along axis i (j_i = 0, ..., n_i - 1).  The parity
    # of j_i is therefore the bit-pattern entry for that axis.
    # PySCF's `cell.make_kpts` orders k_index with the first axis
    # iterating slowest, so axis-0 j-index is the most significant digit.
    even_axes = [i for i, m in enumerate(mesh) if m % 2 == 0 and m >= 2]
    bit_of_k = []  # list (n_k,) of length-3 bit tuples
    for k_index in range(n_k):
        bits = []
        for ax in range(3):
            if mesh[ax] % 2 == 0 and mesh[ax] >= 2:
                # weight = product of mesh[ax+1:]
                weight = 1
                for ax2 in range(ax + 1, 3):
                    weight *= mesh[ax2]
                j_ax = (k_index // weight) % mesh[ax]
                bits.append(j_ax % 2)
            else:
                bits.append(0)
        bit_of_k.append(tuple(bits))

    n_so = 2 * n_bands * n_k
    gens, labels = [], []
    for ax in even_axes:
        gen = np.ones(n_so, dtype=int)
        for b in range(n_bands):
            for k in range(n_k):
                if bit_of_k[k][ax]:
                    for sigma in (0, 1):
                        gen[_so_index(b, k, sigma, n_k)] = -1
        gens.append(gen)
        labels.append(f'T_a{ax}/2')
    return gens, labels


def point_group_z2_generators(kmf, band_k_order, n_k, n_bands):
    """Detect Boolean (Z2-only) point-group symmetries.

    For the v1 implementation we only handle generators that act
    diagonally on the (band, k_index, sigma) basis with eigenvalue +/-1.
    A point-group operation g is included if and only if every Bloch MO
    is an eigenvector of g with eigenvalue +/-1 (within tolerance).

    Returns (gens, labels).  May be empty.
    """
    # Conservative v1: skip detection.  Half-translations + spin parities
    # already cover the model-Hamiltonian benchmarks; full point-group
    # descent for ab-initio crystals will be layered in once the basic
    # path is validated.
    return [], []


# ---------------------------------------------------------------------------
# Strategy A: reuse the molecular pipeline on the supercell as a symmetry
# analyser.  See periodic.build_periodic_inputs for how this is plumbed in.
# ---------------------------------------------------------------------------

def _build_supercell_mol(cell_sc, symmetry=True):
    """Build a `gto.Mole` from the supercell's atoms + basis, with
    `symmetry=True`.  Refuses to silently reorient the cell -- if PySCF
    would reorient/translate the atoms, we raise so the user can align
    their input frame manually.

    Descends any non-Boolean point group to its maximal Boolean subgroup
    supported by ``core.get_character_table`` (C1, Cs, C2, Ci, C2v, C2h,
    D2, D2h).  The descent table covers all 32 crystallographic point
    groups plus the linear and icosahedral groups that PySCF can detect.
    """
    # Maximal Boolean subgroup of every non-Boolean group PySCF may detect.
    # Target groups are the ones in core.get_character_table: C1, Cs, C2,
    # Ci, C2v, C2h, D2, D2h.
    _BOOLEAN_DESCENT = {
        # Linear / spherical (already handled, kept for completeness)
        'Dooh': 'D2h', 'SO3': 'D2h', 'Coov': 'C2v',
        # Cubic
        'Oh': 'D2h', 'Th': 'D2h', 'O': 'D2', 'Td': 'C2v', 'T': 'D2',
        # Tetragonal
        'D4h': 'D2h', 'D4': 'D2', 'C4h': 'C2h', 'C4v': 'C2v',
        'C4': 'C2', 'S4': 'C2', 'D2d': 'C2v',
        # Hexagonal
        'D6h': 'D2h', 'D6': 'D2', 'C6h': 'C2h', 'C6v': 'C2v', 'C6': 'C2',
        # Trigonal
        'D3h': 'C2v', 'D3d': 'C2h', 'D3': 'C2',
        'C3v': 'Cs', 'C3h': 'Cs', 'S6': 'Ci', 'C3': 'C1',
        # Icosahedral (rare in crystals)
        'Ih': 'D2h', 'I': 'D2',
    }
    from pyscf import gto
    atom_list = []
    coords_bohr = cell_sc.atom_coords()  # always in Bohr
    for i, sym in enumerate(cell_sc.elements):
        atom_list.append([sym, tuple(coords_bohr[i])])

    def _build(group_spec):
        mol = gto.Mole()
        mol.atom = atom_list
        mol.unit = 'Bohr'
        mol.basis = cell_sc.basis
        mol.spin = int(cell_sc.spin)
        mol.charge = int(cell_sc.charge)
        mol.symmetry = group_spec
        mol.verbose = 0
        mol.build()
        return mol

    mol_sc = _build(bool(symmetry))
    if symmetry and mol_sc.groupname in _BOOLEAN_DESCENT:
        mol_sc = _build(_BOOLEAN_DESCENT[mol_sc.groupname])

    if symmetry:
        new_coords = mol_sc.atom_coords()
        if (new_coords.shape != coords_bohr.shape
                or np.max(np.abs(new_coords - coords_bohr)) > 1e-6):
            raise RuntimeError(
                "PySCF reoriented the supercell when applying `symmetry=True`. "
                "The periodic SCF was performed in a different frame, so the "
                "AO bases no longer match.\n"
                "Workaround: align your `atom`+`a` inputs to PySCF's standard "
                "orientation (typically: principal axis along z, mirror planes "
                "containing the coordinate axes, inversion centre at origin)."
            )
    return mol_sc


def _atom_permutation_under_translation(cell_sc, prim_a_vec, atol=1e-4):
    """Return the atom permutation induced by translating the supercell
    atoms by the *primitive* lattice vector ``prim_a_vec`` (in Bohr),
    interpreted modulo the supercell lattice vectors.

    Returns
    -------
    perm : (n_atom,) int ndarray
        ``perm[A]`` is the index of the atom that atom ``A`` lands on
        after translation.

    Raises
    ------
    RuntimeError if ``prim_a_vec`` is not a sub-lattice vector of the
    supercell (i.e. the translated atoms do not coincide with original
    atom positions modulo the supercell lattice).
    """
    coords = cell_sc.atom_coords()  # (n_atom, 3) in Bohr
    a_sc = np.asarray(cell_sc.a, dtype=float)  # rows are supercell lattice vectors
    if cell_sc.unit.lower().startswith('b'):
        a_sc_bohr = a_sc
    else:
        # 'Angstrom' or 'A' -> Bohr
        a_sc_bohr = a_sc / 0.52917720859
    elements = cell_sc.elements
    n_atom = len(elements)

    # Build fractional coords in supercell lattice basis
    inv_a = np.linalg.inv(a_sc_bohr.T)  # so inv_a @ r gives fractional coords
    frac = (inv_a @ coords.T).T  # (n_atom, 3)
    frac_t = (inv_a @ (coords + prim_a_vec).T).T

    perm = -np.ones(n_atom, dtype=int)
    for A in range(n_atom):
        for B in range(n_atom):
            if elements[A] != elements[B]:
                continue
            d = frac_t[A] - frac[B]
            d -= np.round(d)
            if np.max(np.abs(d)) < atol:
                perm[A] = B
                break
        if perm[A] < 0:
            raise RuntimeError(
                f"No atom-image found for atom {A} ({elements[A]}) under translation "
                f"by {prim_a_vec} Bohr; primitive vector is not a sub-lattice vector "
                f"of the supercell."
            )
    if len(set(perm.tolist())) != n_atom:
        raise RuntimeError(
            "Atom permutation under translation is not a bijection "
            f"(perm={perm.tolist()})"
        )
    return perm


def _ao_translation_matrix(cell_sc, atom_perm):
    """Return the (n_ao, n_ao) permutation matrix T such that the AO
    basis transforms as ``phi'_mu = sum_nu T_{mu,nu} phi_nu`` under the
    given atom permutation.

    Each AO on atom A is mapped to the corresponding AO (same shell,
    same m-component, same primitive position-within-atom) on atom
    ``atom_perm[A]``.  This is the AO permutation induced by a pure
    lattice translation; no sign change is applied (translation does
    not flip orbital lobes).
    """
    aoslice = cell_sc.aoslice_by_atom()  # (n_atom, 4): (shell0, shellN, ao0, aoN)
    n_atom = aoslice.shape[0]
    n_ao = cell_sc.nao
    T = np.zeros((n_ao, n_ao), dtype=float)
    for A in range(n_atom):
        ao0_A, aoN_A = int(aoslice[A, 2]), int(aoslice[A, 3])
        B = int(atom_perm[A])
        ao0_B, aoN_B = int(aoslice[B, 2]), int(aoslice[B, 3])
        nA = aoN_A - ao0_A
        nB = aoN_B - ao0_B
        if nA != nB:
            raise RuntimeError(
                f"AO count mismatch under atom permutation A={A}->B={B} "
                f"({nA} vs {nB})"
            )
        # phi'_{ao_B + j} = phi_{ao_A + j}  (translation moves the AO from A to B)
        for j in range(nA):
            T[ao0_B + j, ao0_A + j] = 1.0
    return T


def _build_irrep_projectors(mol_sc, S):
    """Return one (n_ao, n_ao) projector per irrep of ``mol_sc``.

    Each projector ``P`` satisfies ``P psi = sum over irrep components``,
    using the AO inner product ``<a|b> = a^T S b``.  Acts on a column
    vector via ``P @ psi``.
    """
    proj = []
    for sao in mol_sc.symm_orb:
        sao = np.asarray(sao, dtype=float)
        M = sao.T @ S @ sao
        w, v = np.linalg.eigh(M)
        if np.min(w) < 1e-10:
            raise RuntimeError(
                f"Symmetry-adapted basis for an irrep of {mol_sc.groupname} is "
                "linearly dependent in the AO overlap; cannot build a projector."
            )
        M_inv_sqrt = (v / np.sqrt(w)) @ v.T
        sao_orth = sao @ M_inv_sqrt
        proj.append(sao_orth @ sao_orth.T @ S)
    return proj


def _filter_pg_invariant_translations(T_ao_primitives, irrep_projs,
                                      tol=1e-6):
    """Filter the GF(2)-span of primitive half-translations down to the
    subspace of translations that commute with every PG irrep
    projector.

    Parameters
    ----------
    T_ao_primitives : list of np.ndarray
        ``[T_a_i^AO]`` for each primitive lattice direction with even
        mesh.
    irrep_projs : list of np.ndarray
        Output of ``_build_irrep_projectors``.
    tol : float
        Frobenius-norm tolerance for the commutator check.

    Returns
    -------
    T_ao_filtered : list of np.ndarray
        A GF(2) basis of the PG-invariant subspace, expressed as AO
        operators ``T_v^AO = prod_i (T_a_i^AO)^{v_i}``.
    label_combos : list of tuple of int
        For each entry of ``T_ao_filtered``, the indices i of the
        primitive directions whose product gives that operator.

    Notes
    -----
    Brute-force enumeration over the ``2**d`` combinations is fine
    because ``d <= 3`` for any 3D Bravais lattice.
    """
    d = len(T_ao_primitives)
    if d == 0:
        return [], []

    # Enumerate all 2^d non-zero v in F2^d, build T_v^AO and test
    # commutation with every irrep projector.
    passing_v = []  # list of bit-tuples
    passing_T = []
    for code in range(1, 1 << d):
        v = tuple((code >> i) & 1 for i in range(d))
        # Compose T_v^AO as product of primitives (they all commute with
        # each other by construction, so order doesn't matter).
        T_v = None
        for i, vi in enumerate(v):
            if vi:
                T_v = T_ao_primitives[i] if T_v is None else T_v @ T_ao_primitives[i]
        # Commutation check
        ok = True
        for P in irrep_projs:
            comm = T_v @ P - P @ T_v
            if np.linalg.norm(comm, ord='fro') > tol:
                ok = False
                break
        if ok:
            passing_v.append(v)
            passing_T.append(T_v)

    if not passing_v:
        return [], []

    # Reduce to a GF(2) basis: row-reduce the bit-vectors. We pick
    # leading-bit vectors greedily so the resulting basis combos are
    # easy to interpret.
    # bits[i] is a list of length d
    bits = [list(v) for v in passing_v]
    rows = list(range(len(bits)))
    chosen_rows = []  # indices into passing_v
    pivot_col = 0
    while pivot_col < d and rows:
        pivot_row = None
        for r in rows:
            if bits[r][pivot_col] == 1:
                pivot_row = r
                break
        if pivot_row is None:
            pivot_col += 1
            continue
        chosen_rows.append(pivot_row)
        # Eliminate this column from all other rows
        for r in rows:
            if r != pivot_row and bits[r][pivot_col] == 1:
                bits[r] = [(a ^ b) for a, b in zip(bits[r], bits[pivot_row])]
        rows.remove(pivot_row)
        pivot_col += 1

    # `chosen_rows` indexes the basis combos. Re-derive the AO operators
    # from the *original* (unmodified) v's we stored in passing_v.
    T_ao_filtered = [passing_T[r] for r in chosen_rows]
    label_combos = [tuple(i for i, vi in enumerate(passing_v[r]) if vi)
                    for r in chosen_rows]
    return T_ao_filtered, label_combos


def _diagonalise_translations_in_energy_blocks(mo_coeff, mo_energy, S,
                                               T_ao_list, energy_tol=5e-3,
                                               sign_tol=1e-3):
    """Within each energy-degenerate block of MOs, simultaneously
    diagonalise every operator in ``T_ao_list`` (which all commute and
    square to identity, i.e. are involutions).

    Returns ``(new_mo_coeff, k_signs)`` where ``new_mo_coeff`` has the
    same column space and energies as ``mo_coeff`` but each column is
    now a simultaneous +/-1 eigenvector of every translation, and
    ``k_signs`` is an ``(n_mo, n_axes)`` array of +/-1 entries.

    The MO ordering is preserved at the block level (we only re-order
    within each block); columns outside any block keep their original
    index.
    """
    n_ao, n_mo = mo_coeff.shape
    new_coeff = mo_coeff.copy()
    k_signs = np.zeros((n_mo, len(T_ao_list)), dtype=int)

    order = np.argsort(mo_energy)
    blocks = []
    cur = [int(order[0])]
    for j in order[1:]:
        j = int(j)
        if abs(mo_energy[j] - mo_energy[cur[0]]) < energy_tol:
            cur.append(j)
        else:
            blocks.append(cur)
            cur = [j]
    blocks.append(cur)

    for blk in blocks:
        cols = mo_coeff[:, blk]  # (n_ao, k)
        k = len(blk)
        # Iteratively split by each translation: each pass partitions
        # into +1 and -1 subspaces.
        sub_partition = [(np.arange(k), np.zeros(k, dtype=int))]
        # sub_partition is a list of (indices_within_block, sign_so_far_unused)
        for ax, T_ao in enumerate(T_ao_list):
            new_partition = []
            for idxs, _ in sub_partition:
                if idxs.size == 0:
                    continue
                C = cols[:, idxs]  # (n_ao, m)
                # Operator T projected onto this subspace, in the cols basis,
                # using the AO inner product:
                #   M_{ij} = <c_i|S T|c_j> / <c_i|S|c_j_normalised>
                # We assume cols are already S-orthonormal (KRHF MOs are).
                M = C.T @ S @ T_ao @ C  # (m, m)
                # Symmetrise (T is orthogonal under S; numerically symmetric)
                M = 0.5 * (M + M.T)
                w, V = np.linalg.eigh(M)
                # eigenvalues should cluster near +/-1
                if np.max(np.abs(np.abs(w) - 1.0)) > sign_tol:
                    raise RuntimeError(
                        f"Half-translation axis {ax}: encountered eigenvalue "
                        f"{w.tolist()} (expected +/-1) within an "
                        f"energy-degenerate block of size {k}. Translation does "
                        f"not commute with H within the symm-tol energy window."
                    )
                # Replace cols[:, idxs] by C @ V (in the right order)
                C_new = C @ V
                cols[:, idxs] = C_new
                # Split into +1 and -1
                pos = idxs[w > 0]
                neg = idxs[w < 0]
                new_partition.append((pos, None))
                new_partition.append((neg, None))
            sub_partition = new_partition
        # cols now contains simultaneous eigenvectors of all T_ao_list ops
        new_coeff[:, blk] = cols
        # Read off signs by direct evaluation on the rotated columns
        for ax, T_ao in enumerate(T_ao_list):
            for local_j, j in enumerate(blk):
                psi = cols[:, local_j]
                ev = float(psi @ S @ T_ao @ psi) / float(psi @ S @ psi)
                if abs(ev - 1.0) < 1e-3:
                    k_signs[j, ax] = +1
                elif abs(ev + 1.0) < 1e-3:
                    k_signs[j, ax] = -1
                else:
                    raise RuntimeError(
                        f"Post-diagonalisation MO {j} axis {ax} eigenvalue "
                        f"{ev:.6f} is not +/-1 (block size {k})."
                    )
    return new_coeff, k_signs


def _compute_k_signs_per_mo(mo_coeff, S, T_ao_list, tol=1e-4):
    """For each MO column, compute its eigenvalue (+/-1) under each AO
    translation matrix in ``T_ao_list``.

    Returns
    -------
    signs : (n_mo, n_axes) int ndarray of values in {-1, +1}.

    Raises
    ------
    RuntimeError if any MO is not a simultaneous eigenvector of every
    operator in ``T_ao_list`` (i.e. the eigenvalue is not within ``tol``
    of +/-1).
    """
    n_axes = len(T_ao_list)
    n_mo = mo_coeff.shape[1]
    signs = np.zeros((n_mo, n_axes), dtype=int)
    for ax, T_ao in enumerate(T_ao_list):
        # eigenvalue of MO p = (psi^T S T psi) / (psi^T S psi) since T is orthogonal
        # under the AO inner product induced by S (T is a permutation that maps
        # equivalent AOs to equivalent AOs, preserving the overlap matrix).
        ST = S @ T_ao
        for p in range(n_mo):
            psi = mo_coeff[:, p]
            denom = float(psi @ S @ psi)
            if denom < 1e-12:
                raise RuntimeError(f"MO {p} has zero norm under S")
            ev = float(psi @ ST @ psi) / denom
            if abs(ev - 1.0) < tol:
                signs[p, ax] = +1
            elif abs(ev + 1.0) < tol:
                signs[p, ax] = -1
            else:
                raise RuntimeError(
                    f"MO {p} is not an eigenvector of half-translation along axis "
                    f"{ax}: <psi|S T_ao|psi>/<psi|S|psi> = {ev:.6f} (expected +/-1)."
                )
    return signs


def _symmetry_adapt_mos(mol_sc, mo_coeff, mo_energy, S, energy_tol=5e-3,
                        irrep_purity_tol=0.95, k_signs_per_mo=None):
    """Within each degenerate eigenspace of `mo_energy`, rotate the
    columns of `mo_coeff` so that each MO lies in a single irrep of
    `mol_sc`.  Returns ``(new_mo_coeff, irrep_labels)``.

    Notes
    -----
    The supercell Hamiltonian commutes with each point-group operation,
    so within each (numerically) degenerate subspace we can freely
    rotate to a symmetry-adapted basis without changing the SCF energy
    or any subsequent FCI calculation.

    If ``k_signs_per_mo`` is provided (shape ``(n_mo, n_axes)`` with
    +/-1 entries), MOs are first split by their k-signature and only
    rotated within (energy, k-signature) blocks.  This preserves the
    half-translation eigenlabels of every MO across the symmetry
    adaptation.
    """
    n_ao, n_mo = mo_coeff.shape
    new_coeff = mo_coeff.copy()
    labels = [None] * n_mo

    irrep_names = list(mol_sc.irrep_name)
    proj = _build_irrep_projectors(mol_sc, S)

    # Group MOs by approximate energy degeneracy AND, if provided, by
    # k-signature (so that half-translation eigenlabels are preserved).
    order = np.argsort(mo_energy)

    def _ksig(j):
        if k_signs_per_mo is None:
            return ()
        return tuple(int(x) for x in k_signs_per_mo[j])

    groups = []
    cur = [int(order[0])]
    cur_ksig = _ksig(int(order[0]))
    for j in order[1:]:
        j = int(j)
        same_e = abs(mo_energy[j] - mo_energy[cur[0]]) < energy_tol
        same_k = (_ksig(j) == cur_ksig)
        if same_e and same_k:
            cur.append(j)
        else:
            groups.append(cur)
            cur = [j]
            cur_ksig = _ksig(j)
    groups.append(cur)

    # If k_signs were provided, also split blocks that span multiple
    # k-signatures within an energy window (can happen if energy_tol is
    # generous enough to merge near-degenerate same-energy bands across
    # different k-points).
    if k_signs_per_mo is not None:
        refined = []
        for grp in groups:
            by_k = {}
            for j in grp:
                by_k.setdefault(_ksig(j), []).append(j)
            refined.extend(by_k.values())
        groups = refined

    for grp in groups:
        block = mo_coeff[:, grp]  # (n_ao, k)
        k = len(grp)
        if k == 1:
            psi = block[:, 0]
            best_w, best_irr = -1.0, None
            for k_irr, P in enumerate(proj):
                w = float(psi @ S @ P @ psi)
                if w > best_w:
                    best_w, best_irr = w, k_irr
            if best_w < irrep_purity_tol:
                raise RuntimeError(
                    f"MO {grp[0]} (E={mo_energy[grp[0]]:.6f}) is not a pure symmetry "
                    f"eigenstate (max irrep overlap {best_w:.3f} < {irrep_purity_tol}). "
                    "Consider raising energy_tol or checking that the supercell is in "
                    "PySCF's standard orientation."
                )
            labels[grp[0]] = irrep_names[best_irr]
        else:
            # Multi-MO degenerate group: diagonalise each irrep projector
            # restricted to the group, harvest eigenvectors with eigenvalue ~1
            new_cols = []
            new_lbls = []
            for k_irr, P in enumerate(proj):
                M = block.T @ S @ P @ block  # (k, k), Hermitian, eigenvalues in [0, 1]
                w, v = np.linalg.eigh(M)
                for j in range(k):
                    if w[j] > 0.5:
                        psi = block @ v[:, j]
                        psi = psi / np.sqrt(float(psi @ S @ psi))
                        new_cols.append(psi)
                        new_lbls.append(irrep_names[k_irr])
            if len(new_cols) != k:
                raise RuntimeError(
                    f"Symmetry-adaptation failed for degenerate group at "
                    f"E~{mo_energy[grp[0]]:.6f}: got {len(new_cols)} pure-irrep "
                    f"vectors from a degenerate subspace of dimension {k}."
                )
            for j, i in enumerate(grp):
                new_coeff[:, i] = new_cols[j]
                labels[i] = new_lbls[j]
    return new_coeff, labels


def _fermion_hamiltonian_in_mo_basis(mf_sc, mo_coeff, kmf=None):
    """Build the second-quantised FermionOperator of `mf_sc` (a Gamma-point
    supercell SCF) in the supplied MO basis with interleaved spin
    indexing.

    If ``kmf`` is provided and uses GDF/MDF, both the 1-electron and
    2-electron integrals are built via the per-k-point primitive AO ERIs
    cached on ``kmf.with_df`` (avoids evaluating AOs on the supercell
    FFTDF grid).  Otherwise falls back to a fresh FFTDF on ``mf_sc.cell``.
    """
    from pyscf import ao2mo
    from pyscf.pbc.df import GDF as _GDF, MDF as _MDF
    cell = mf_sc.cell
    if mo_coeff.dtype != float:
        if np.max(np.abs(mo_coeff.imag)) > 1e-9:
            raise RuntimeError("Supplied mo_coeff has non-trivial imaginary part.")
        mo_coeff = mo_coeff.real

    n_mo = mo_coeff.shape[1]
    if kmf is not None and isinstance(kmf.with_df, (_GDF, _MDF)):
        n_k = len(kmf.kpts)
        mo_at_k = _supercell_mo_to_kpt_mo(mf_sc, kmf, mo_coeff)
        hcore_k = np.asarray(kmf.get_hcore())  # (n_k, nao_prim, nao_prim)
        h1_c = np.zeros((n_mo, n_mo), dtype=complex)
        for k in range(n_k):
            Ck = mo_at_k[k]
            h1_c += Ck.conj().T @ hcore_k[k] @ Ck
        if np.max(np.abs(h1_c.imag)) > 1e-7:
            raise RuntimeError(
                f"Imag part of h1 from kpts route too large: "
                f"{np.max(np.abs(h1_c.imag)):.2e}"
            )
        h1 = h1_c.real
    else:
        hcore_ao = mf_sc.get_hcore()
        if np.iscomplexobj(hcore_ao):
            hcore_ao = hcore_ao.real
        h1 = mo_coeff.T @ hcore_ao @ mo_coeff

    # Ewald exxdiv shift: see _supercell_fermion_hamiltonian for derivation.
    if getattr(mf_sc, 'exxdiv', None) == 'ewald' and cell.dimension > 0:
        from pyscf.pbc.tools import madelung as _madelung
        xi_M = float(_madelung(cell, np.zeros((1, 3))))
        h1 = h1 - 0.5 * xi_M * np.eye(h1.shape[0])
    # Use the kpts route (per-k primitive-AO ERIs) for GDF/MDF backends,
    # which cache ``_cderi`` from KRHF so each ``ao2mo`` call is cheap.
    # The kpts-ERI formula is correct for any supercell MOs (Bloch-pure
    # or PG-adapted mixtures): D_k = 2 C_k C_k† is the correct k-space
    # density and the triple-k sum reproduces the full 4-center ERI via
    # momentum conservation.  On TRIM-only meshes (every n_i in {1,2})
    # all mo_at_k[k] happen to be real; on larger even meshes per-k MOs
    # are complex but the sum over k pairs still gives a real h1/ERI by
    # Hermiticity, which the imag-part check below verifies.
    # For FFTDF there is no ``_cderi`` cache, so the supercell FFTDF
    # single call is faster than N_k^3 per-block GDF calls.
    from pyscf.pbc.df import GDF as _GDF, MDF as _MDF
    if kmf is not None and isinstance(kmf.with_df, (_GDF, _MDF)):
        eri_mo = _supercell_mo_eri_via_kpts(mf_sc, kmf, mo_coeff)
    else:
        eri_mo = _supercell_mo_eri(mf_sc, mo_coeff)

    H = _build_interleaved_fermion_op(h1, eri_mo, float(mf_sc.energy_nuc()))
    return H


def _active_space_fermion_hamiltonian(mf_sc, mo_coeff, active_indices_0,
                                       kmf=None, integral_backend='kpts'):
    """Build the active-space FermionOperator using periodic ERIs and the
    standard frozen-core mean-field correction to ``h1``.

    Parameters (integral_backend)
    ------------------------------
    'kpts' (default)
        Use k-point GDF/MDF integrals via ``kmf.with_df`` when available.
        Gives the most physically rigorous periodic integrals but produces
        a ~mHa discrepancy with PySCF CASCI (which uses supercell integrals).
    'mf_sc'
        Force the supercell (Gamma-point) integral route: hcore and J/K
        come from ``mf_sc``, 2e integrals from ``mf_sc.with_df`` (GDF/MDF
        on the supercell) when available, otherwise FFTDF.  Matches PySCF
        CASCI to floating-point precision, enabling end-to-end validation.

    Parameters
    ----------
    mf_sc : pyscf.pbc.scf.RHF
        Gamma-folded supercell SCF (output of ``fold_to_supercell``).
    mo_coeff : (nao, nmo) real ndarray
        Symmetry-adapted MO coefficients in the supercell AO basis.
        Each column must be either doubly occupied or empty at HF.
    active_indices_0 : list[int]
        Zero-indexed positions (into ``mo_coeff`` columns) of the
        active MOs.

    Returns
    -------
    H_active : openfermion.FermionOperator
        Active-space second-quantised Hamiltonian (spinorbital count
        ``2*len(active_indices_0)``) in interleaved spin order.  Includes
        the constant ``E_core`` so its lowest eigenvalue matches the full
        Hamiltonian's lowest eigenvalue in the (core-occupied, virtual-
        empty) sector to numerical precision.
    n_act : int
        Number of active spatial MOs (i.e. ``len(active_indices_0)``).
    nelec_in_active : int
        Number of electrons in the active window (sum of mo_occ over the
        set of active MOs, rounded).
    e_core : float
        Frozen-core energy + nuclear-nuclear + Madelung shift on core
        electrons.  Already added to ``H_active`` as a constant; returned
        separately for diagnostics.
    active_sorted : list[int]
        ``active_indices_0`` sorted ascending (the order matching the
        spinorbital indexing of ``H_active``).
    """
    from pyscf import ao2mo

    cell = mf_sc.cell
    mo_coeff = np.asarray(mo_coeff)
    if np.iscomplexobj(mo_coeff):
        if np.max(np.abs(mo_coeff.imag)) > 1e-9:
            raise RuntimeError("mo_coeff has non-trivial imaginary part")
        mo_coeff = mo_coeff.real

    mo_occ = np.asarray(mf_sc.mo_occ)
    if mo_occ.ndim > 1:
        raise RuntimeError(
            "Expected scalar mo_occ from a Gamma-folded supercell; got an array."
        )

    n_mo_total = mo_coeff.shape[1]
    active_set = set(int(i) for i in active_indices_0)
    if not active_set.issubset(range(n_mo_total)):
        raise ValueError(
            f"active_indices_0 {sorted(active_set)} out of range [0, {n_mo_total - 1}]"
        )
    # Anything outside the active window must be either fully doubly occupied
    # (treated as frozen core) or empty (dropped).  Fractional occupations are
    # not supported in this v1.
    core_indices = []
    for i in range(n_mo_total):
        if i in active_set:
            continue
        if mo_occ[i] > 1.5:
            core_indices.append(i)
        elif mo_occ[i] > 0.5:
            raise NotImplementedError(
                f"Fractional / singly-occupied MO {i} (occ={mo_occ[i]:.3f}) "
                "not in active window; not yet supported."
            )
    active_sorted = sorted(active_set)
    mo_core = mo_coeff[:, core_indices]
    mo_act = mo_coeff[:, active_sorted]
    n_act = mo_act.shape[1]
    nelec_in_active = int(round(sum(mo_occ[i] for i in active_sorted)))

    # ---- 1e: hcore + frozen-core mean-field ----
    # When kmf with a cached DF (GDF/MDF) is available, evaluate hcore and
    # the frozen-core J/K natively on the primitive cell at each k, reusing
    # ``kmf.with_df``.  This avoids (i) running ``mf_sc.get_jk`` on the
    # supercell FFTDF (which can blow up to many GB even for small cores)
    # and (ii) building hcore on the supercell FFT grid.
    #
    # Math: with ``mo_at_k = _supercell_mo_to_kpt_mo(mf_sc, kmf, mo_sc)`` and
    # ``D_core_k[k] = 2 * mo_core_k[k] @ mo_core_k[k].conj().T``,
    #   Tr[D_sc @ M_sc]                          = sum_k Tr[D_core_k @ M_k]
    #   <p|M_sc|q>_act = sum_k mo_act_k[k]^dagger M_k mo_act_k[k]
    # because the k2gamma phase satisfies sum_R phase[R,k] phase[R,k']^*
    # = delta_{k,k'} (unitary in (R,k)).
    from pyscf.pbc.df import GDF as _GDF, MDF as _MDF
    use_kpts_1e = (
        integral_backend == 'kpts'
        and kmf is not None
        and isinstance(kmf.with_df, (_GDF, _MDF))
    )

    if use_kpts_1e:
        n_k_loc = len(kmf.kpts)
        mo_act_at_k = _supercell_mo_to_kpt_mo(mf_sc, kmf, mo_act)
        hcore_k = np.asarray(kmf.get_hcore())  # (n_k, nao_prim, nao_prim)

        if mo_core.shape[1] > 0:
            mo_core_at_k = _supercell_mo_to_kpt_mo(mf_sc, kmf, mo_core)
            D_core_k = np.array([
                2.0 * mo_core_at_k[k] @ mo_core_at_k[k].conj().T
                for k in range(n_k_loc)
            ])
            J_core_k, K_core_k = kmf.get_jk(dm_kpts=D_core_k)
            J_core_k = np.asarray(J_core_k)
            K_core_k = np.asarray(K_core_k)
            veff_core_k = J_core_k - 0.5 * K_core_k
            h1_eff_k = hcore_k + veff_core_k
        else:
            D_core_k = None
            veff_core_k = None
            h1_eff_k = hcore_k

        h1_act_c = np.zeros((n_act, n_act), dtype=complex)
        for k in range(n_k_loc):
            Ck = mo_act_at_k[k]
            h1_act_c += Ck.conj().T @ h1_eff_k[k] @ Ck
        if np.max(np.abs(h1_act_c.imag)) > 1e-7:
            raise RuntimeError(
                f"Imag part of active h1 from kpts route too large: "
                f"{np.max(np.abs(h1_act_c.imag)):.2e}"
            )
        h1_act = h1_act_c.real
        # Keep handles for the e_core trace below.
        _ec_kpts = (D_core_k, hcore_k, veff_core_k)
        # Sentinel so the legacy AO-trace branch is skipped.
        D_core = "kpts" if mo_core.shape[1] > 0 else None
        veff_core = None
    else:
        hcore_ao = mf_sc.get_hcore()
        if np.iscomplexobj(hcore_ao):
            hcore_ao = hcore_ao.real

        if mo_core.shape[1] > 0:
            D_core = 2.0 * mo_core @ mo_core.T  # closed-shell density of frozen core
            # FFTDF buffers the AO ERI on a dense FFT grid; large supercells
            # (many core MOs, many AOs) can overflow the default 4 GB budget.
            # Temporarily raise the limit to avoid the buffer-too-small error.
            _df = mf_sc.with_df
            _old_mem = getattr(_df, 'max_memory', 4000)
            if hasattr(_df, 'max_memory'):
                _df.max_memory = max(_old_mem, 64000)
            J_core, K_core = mf_sc.get_jk(dm=D_core)
            if hasattr(_df, 'max_memory'):
                _df.max_memory = _old_mem
            if np.iscomplexobj(J_core): J_core = J_core.real
            if np.iscomplexobj(K_core): K_core = K_core.real
            veff_core = J_core - 0.5 * K_core
            h1_eff_ao = hcore_ao + veff_core
        else:
            D_core = None
            veff_core = None
            h1_eff_ao = hcore_ao

        h1_act = mo_act.T @ h1_eff_ao @ mo_act
        _ec_kpts = None

    # Ewald exxdiv shift on active MOs (mirrors _fermion_hamiltonian_in_mo_basis)
    xi_M = 0.0
    if getattr(mf_sc, 'exxdiv', None) == 'ewald' and cell.dimension > 0:
        from pyscf.pbc.tools import madelung as _madelung
        xi_M = float(_madelung(cell, np.zeros((1, 3))))
        h1_act = h1_act - 0.5 * xi_M * np.eye(n_act)

    # ---- 2e integrals on active block only ----
    # The kpts route reuses the primitive-cell DF intermediate cached on
    # ``kmf.with_df`` -- a big win for GDF/MDF (where ``_cderi`` is built
    # once during KRHF and reused for all 512 (ki,kj,kk,kl) tuples).  For
    # FFTDF there is no such cache: each call rebuilds the FFT
    # intermediate from scratch, making the supercell route faster.
    from pyscf.pbc.df import GDF, MDF
    use_kpts_route = (
        integral_backend == 'kpts'
        and kmf is not None
        and isinstance(kmf.with_df, (GDF, MDF))
    )
    if use_kpts_route:
        eri_act = _supercell_mo_eri_via_kpts(mf_sc, kmf, mo_act)
    else:
        eri_act = _supercell_mo_eri(mf_sc, mo_act)

    # ---- E_core ----
    if D_core is not None:
        if _ec_kpts is not None:
            D_core_k, hcore_k, veff_core_k = _ec_kpts
            n_k_loc = D_core_k.shape[0]
            e1c = sum(
                np.einsum('ij,ji->', D_core_k[k], hcore_k[k])
                for k in range(n_k_loc)
            )
            e2c = 0.5 * sum(
                np.einsum('ij,ji->', D_core_k[k], veff_core_k[k])
                for k in range(n_k_loc)
            )
            if abs(e1c.imag) > 1e-6 or abs(e2c.imag) > 1e-6:
                raise RuntimeError(
                    f"E_core has spurious imag part: 1e={e1c.imag:.2e} "
                    f"2e={e2c.imag:.2e}"
                )
            e_core_1e = float(e1c.real)
            e_core_2e = float(e2c.real)
        else:
            e_core_1e = float(np.einsum('ij,ji->', D_core, hcore_ao))
            e_core_2e = 0.5 * float(np.einsum('ij,ji->', D_core, veff_core))
        # NOTE on Madelung: PySCF's get_jk with exxdiv='ewald' already bakes
        # the Ewald shift into K (K += madelung * S @ D @ S).  Tracing
        # 0.5 * Tr[D_core @ veff_core] therefore already contains the core
        # Madelung correction (-0.5 * madelung * N_core_e).  We must NOT
        # add an explicit e_core_madelung -- doing so would double-count.
        # The h1_act subtraction below handles the Madelung shift for the
        # ACTIVE electrons only (they are not represented in K_core).
        e_core = e_core_1e + e_core_2e
    else:
        e_core = 0.0
    e_core += float(mf_sc.energy_nuc())

    # ---- Build active-space FermionOperator (interleaved spin) ----
    H = _build_interleaved_fermion_op(h1_act, eri_act, e_core)
    return H, n_act, nelec_in_active, e_core, active_sorted


def _active_space_number_penalty(n_act, n_up_target, n_dn_target, strength=2.0):
    """FermionOperator for ``strength*(N_up - t_up)^2 + strength*(N_dn - t_dn)^2``.

    Expanding (N - t)^2 = (1-2t)*N + 2*sum_{p<q} n_p*n_q + t^2, using the
    fermion idempotency n_p^2 = n_p and the identity
    n_i*n_j = a†_i a†_j a_j a_i  (i < j, different spin-orbital indices).

    The penalty is zero for the target (n_up_target, n_dn_target) particle
    sector and positive everywhere else, making the physical ground state the
    minimum of the qubit Hamiltonian.  In periodic active-space calculations
    the frozen-core Coulomb field can make h1_act diagonal positive, which
    would otherwise let the vacuum (0 active electrons) be the minimum.
    """
    from openfermion import FermionOperator
    const = strength * float(n_up_target ** 2 + n_dn_target ** 2)
    pen = FermionOperator('', const)
    for p in range(n_act):
        pu, pd = 2 * p, 2 * p + 1
        pen += FermionOperator(f'{pu}^ {pu}', strength * (1.0 - 2 * n_up_target))
        pen += FermionOperator(f'{pd}^ {pd}', strength * (1.0 - 2 * n_dn_target))
        for q in range(p + 1, n_act):
            qu, qd = 2 * q, 2 * q + 1
            pen += FermionOperator(((pu, 1), (qu, 1), (qu, 0), (pu, 0)), 2 * strength)
            pen += FermionOperator(((pd, 1), (qd, 1), (qd, 0), (pd, 0)), 2 * strength)
    return pen


# ---------------------------------------------------------------------------
# Space-group-aware extra Z2 generators
# ---------------------------------------------------------------------------
#
# The ``mol_sc``-based PG detection only sees operations that map the *finite*
# 16-atom (etc.) supercell cluster back to itself with a fixed origin.  Many
# space-group operations of the infinite crystal map atoms only modulo a
# supercell lattice vector, so they're invisible to the molecular detector.
# These helpers enumerate the full primitive space group via PySCF's
# ``cell.lattice_symmetry``, lift each involution to a supercell-AO orthogonal
# matrix, and add any new GF(2)-independent ones as extra Z2 generators.

def _axis_label(R):
    """Return a short axis label for a Z2 rotation/reflection R (3x3 integer
    matrix in primitive lattice coords).  Falls back to a generic tag when no
    obvious axis exists."""
    R = np.asarray(R, dtype=float)
    det = int(round(np.linalg.det(R)))
    tr = int(round(np.trace(R)))
    if det == -1 and tr == -3:
        return 'i'
    # Eigendecomposition: pick eigenvector with eigenvalue +1 (C2 axis) or
    # -1 (mirror normal).
    w, v = np.linalg.eig(R)
    target = +1 if det == 1 else -1
    idx = int(np.argmin(np.abs(w - target)))
    ax = v[:, idx].real
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    # Snap small components to 0 and unit components to ±1 for readability
    ax_snap = np.where(np.abs(ax) < 0.2, 0, np.sign(ax))
    s = ''.join('+' if c > 0 else ('-' if c < 0 else '0') for c in ax_snap)
    if det == 1 and tr == -1:
        return f'C2[{s}]'
    if det == -1 and tr == +1:
        return f'σ[{s}]'
    return f'op(det={det:+d},tr={tr:+d})'


def _extra_supercell_pg_ops(prim_cell, mf_sc, S, atol_pos=1e-4,
                             ortho_tol=1e-6):
    """Enumerate space-group involutions of the *primitive* cell that close
    on the supercell as orthogonal AO operators.

    Returns
    -------
    ops : list of (label, U_ao) tuples
        Each ``U_ao`` is an (n_ao_sc, n_ao_sc) real matrix that satisfies
        ``U^T S U = S`` (orthogonal w.r.t. AO overlap).
    """
    from pyscf.pbc import gto as _pbcgto
    cell_sym = _pbcgto.Cell()
    cell_sym.atom = prim_cell.atom
    cell_sym.a = prim_cell.a
    cell_sym.basis = prim_cell.basis
    if getattr(prim_cell, 'pseudo', None) not in (None, {}):
        cell_sym.pseudo = prim_cell.pseudo
    cell_sym.unit = prim_cell.unit
    cell_sym.spin = int(prim_cell.spin)
    cell_sym.charge = int(prim_cell.charge)
    cell_sym.space_group_symmetry = True
    cell_sym.symmorphic = False
    cell_sym.verbose = 0
    cell_sym.build()
    ls = cell_sym.lattice_symmetry

    sc_cell = mf_sc.cell
    coords_sc = sc_cell.atom_coords()  # Bohr
    elements_sc = list(sc_cell.elements)
    n_atom_sc = len(elements_sc)

    # Lattices in Bohr
    def _to_bohr(a, unit):
        a = np.asarray(a, dtype=float)
        if str(unit).lower().startswith('a'):
            return a / 0.52917720859
        return a
    a_prim_bohr = _to_bohr(prim_cell.a, prim_cell.unit)
    a_sc_bohr = _to_bohr(sc_cell.a, sc_cell.unit)
    inv_prim_T = np.linalg.inv(a_prim_bohr.T)
    inv_sc_T = np.linalg.inv(a_sc_bohr.T)

    # Per-atom shell list: (l, nctr, ao_start)
    ao_loc = sc_cell.ao_loc_nr()
    atom_shells = {A: [] for A in range(n_atom_sc)}
    for shell_id in range(sc_cell.nbas):
        A = int(sc_cell.bas_atom(shell_id))
        l = int(sc_cell.bas_angular(shell_id))
        nctr = int(sc_cell.bas_nctr(shell_id))
        ao_start = int(ao_loc[shell_id])
        atom_shells[A].append((l, nctr, ao_start))

    n_ao = sc_cell.nao
    out = []
    seen_signatures = set()  # avoid duplicates from equivalent ops

    for op_idx, op in enumerate(ls.ops):
        if op.is_eye:
            continue
        R = np.asarray(op.rot, dtype=int)
        if not np.array_equal(R @ R, np.eye(3, dtype=int)):
            continue  # not an involution in the rotation part
        t = np.asarray(op.trans, dtype=float)

        # Cartesian rotation/translation
        R_cart = a_prim_bohr.T @ R.astype(float) @ inv_prim_T
        t_cart = a_prim_bohr.T @ t

        # Atom permutation on the supercell, modulo supercell lattice
        new_coords = (R_cart @ coords_sc.T).T + t_cart
        perm = -np.ones(n_atom_sc, dtype=int)
        ok = True
        for A in range(n_atom_sc):
            r_new = new_coords[A]
            for B in range(n_atom_sc):
                if elements_sc[B] != elements_sc[A]:
                    continue
                d = inv_sc_T @ (r_new - coords_sc[B])
                d -= np.round(d)
                if np.max(np.abs(d)) < atol_pos:
                    perm[A] = B
                    break
            if perm[A] < 0:
                ok = False
                break
        if not ok or len(set(perm.tolist())) != n_atom_sc:
            continue

        # Build U_ao using PySCF's per-shell Wigner-D matrices
        Dmat_per_l = ls.Dmats[op_idx]
        U = np.zeros((n_ao, n_ao), dtype=float)
        for A in range(n_atom_sc):
            B = int(perm[A])
            shells_A = atom_shells[A]
            shells_B = atom_shells[B]
            if len(shells_A) != len(shells_B):
                ok = False; break
            for sA, sB in zip(shells_A, shells_B):
                lA, nctrA, ao_A = sA
                lB, nctrB, ao_B = sB
                if (lA, nctrA) != (lB, nctrB):
                    ok = False; break
                D = np.asarray(Dmat_per_l[lA], dtype=float)
                m_dim = 2 * lA + 1
                for c in range(nctrA):
                    U[ao_B + c*m_dim:ao_B + (c+1)*m_dim,
                      ao_A + c*m_dim:ao_A + (c+1)*m_dim] = D
            if not ok:
                break
        if not ok:
            continue

        # Sanity: U should be orthogonal under S
        residual = float(np.linalg.norm(U.T @ S @ U - S, ord='fro')) / np.sqrt(S.shape[0])
        if residual > ortho_tol:
            continue

        # Deduplicate by a signature based on the action on a probe vector
        # (cheap; the atom_perm + R together fully identify the op anyway).
        sig = (tuple(perm.tolist()), tuple(R.flatten().tolist()))
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)

        label = _axis_label(R)
        out.append((label, U))
    return out


def _refine_blocks_with_extra_ops(mo_coeff, S, candidate_ops, blocks,
                                   eig_tol=1e-2, spill_tol=1e-2):
    """For each candidate op (label, U_ao), check that it acts block-diagonally
    on the current ``blocks`` partition of MO indices, that its block-eigenvalues
    are ±1, and (if so) rotate within each block to diagonalise it.

    Returns
    -------
    new_mo_coeff : (n_ao, n_mo) ndarray
    new_blocks   : list of list of MO indices (refined by accepted ops)
    accepted     : list of (label, sign_vec) where sign_vec is +/-1 per MO.
    """
    n_mo = mo_coeff.shape[1]
    accepted = []
    cur_coeff = mo_coeff.copy()
    cur_blocks = [list(b) for b in blocks]

    for label, U in candidate_ops:
        # MO-basis representation of U
        M = cur_coeff.T @ S @ U @ cur_coeff
        M = (M + M.T) / 2  # symmetrise (involution -> symmetric in MO basis up to numerics)

        # Check block-diagonal structure: off-diagonal-block norm small
        ok = True
        per_block = []  # list of (block_indices, eigvals, eigvecs)
        for block in cur_blocks:
            sub = M[np.ix_(block, block)]
            other = [i for i in range(n_mo) if i not in block]
            if other:
                spill = M[np.ix_(other, block)]
                if float(np.linalg.norm(spill, 'fro')) > spill_tol:
                    ok = False; break
            w, v = np.linalg.eigh(sub)
            if not np.all(np.abs(np.abs(w) - 1.0) < eig_tol):
                ok = False; break
            per_block.append((block, w, v))
        if not ok:
            continue

        # Apply rotations and refine blocks
        sign_vec = np.zeros(n_mo, dtype=int)
        new_coeff = cur_coeff.copy()
        new_blocks = []
        for block, w, v in per_block:
            block_arr = np.asarray(block)
            new_coeff[:, block_arr] = cur_coeff[:, block_arr] @ v
            signs = np.where(w > 0, +1, -1).astype(int)
            for j_local, j in enumerate(block):
                sign_vec[j] = int(signs[j_local])
            pos = [block[k] for k in range(len(block)) if signs[k] > 0]
            neg = [block[k] for k in range(len(block)) if signs[k] < 0]
            if pos: new_blocks.append(pos)
            if neg: new_blocks.append(neg)

        # GF(2) independence check vs already-accepted sign vectors
        if accepted:
            existing = np.array([sv for _, sv in accepted])  # (k, n_mo) ±1
            existing_b = (existing < 0).astype(int)  # GF(2)
            new_b = (sign_vec < 0).astype(int)
            # Solve A^T x = new_b mod 2 ; if solvable -> dependent
            from numpy.linalg import matrix_rank
            stacked = np.vstack([existing_b, new_b]) % 2
            r_with = _gf2_rank(stacked)
            r_without = _gf2_rank(existing_b)
            if r_with == r_without:
                continue  # dependent

        accepted.append((label, sign_vec))
        cur_coeff = new_coeff
        cur_blocks = new_blocks

    return cur_coeff, cur_blocks, accepted


def _gf2_rank(M):
    """Rank over GF(2) of an integer matrix M (entries 0/1)."""
    A = np.asarray(M, dtype=np.uint8) % 2
    A = A.copy()
    n_row, n_col = A.shape
    r = 0
    col = 0
    while r < n_row and col < n_col:
        pivot = None
        for i in range(r, n_row):
            if A[i, col] == 1:
                pivot = i; break
        if pivot is None:
            col += 1; continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for i in range(n_row):
            if i != r and A[i, col] == 1:
                A[i] = (A[i] ^ A[r])
        r += 1
        col += 1
    return r


def _generators_from_supercell_symmetry(mol_sc, label_orb_symm, n_spinorbital,
                                         nelectron_up, nelectron_down):
    """Run the molecular `find_symmetry_generators` on the symmetry-adapted
    supercell and return ``(gens, signs, labels)`` ready for `_setup_periodic`.

    The point-group analysis is delegated to the existing molecular code,
    which automatically detects the largest abelian Z2^k subgroup.
    Half-translations along even-mesh axes that manifest as supercell
    point-group elements (e.g. inversion centred between equivalent atoms)
    are picked up automatically.
    """
    from .core import (find_symmetry_generators, get_character_table,
                       find_ground_state_irrep)

    point_group_name = mol_sc.groupname
    character_table, conj_labels, irrep_labels, _ = get_character_table(point_group_name)

    # Build a fake mo_occ vector consistent with (n_up + n_dn) electrons in the
    # lowest spatial MOs.  Only used by find_ground_state_irrep to pick the
    # target irrep; we use double occupation of the lowest n_pair orbitals.
    n_mo = n_spinorbital // 2
    n_pair = min(nelectron_up, nelectron_down)
    n_single = abs(nelectron_up - nelectron_down)
    mo_occ = np.zeros(n_mo)
    for i in range(n_pair):
        mo_occ[i] = 2.0
    for i in range(n_pair, n_pair + n_single):
        mo_occ[i] = 1.0

    irrep = find_ground_state_irrep(label_orb_symm, mo_occ, character_table, irrep_labels)

    nelec = nelectron_up + nelectron_down
    spin = nelectron_up - nelectron_down
    (gen_labels, gen_strings, target_qubits, gens, signs, descs) = find_symmetry_generators(
        mol_sc, irrep, label_orb_symm, CAS_qubits=None,
        nelec_spin_override=(nelec, spin)
    )
    # `gens` may be a list of arrays or a 2D array; coerce to list of np.int arrays
    gens_out = [np.asarray(g, dtype=int) for g in gens]
    signs_out = [int(s) for s in signs]
    return gens_out, signs_out, list(gen_labels), irrep



def build_periodic_inputs(atom, a, basis, kpts_mesh, pseudo=None,
                          spin=0, charge=0, df='auto', exxdiv='ewald',
                          active_bands=None, verbose=0, name='crystal',
                          symmetry=True,
                          symm_energy_tol=5e-3, symm_purity_tol=0.95,
                          active_mos=None, integral_backend='kpts'):
    """End-to-end builder for the periodic-from-PySCF path.

    Returns a dict suitable for feeding into ``Encoding(periodic=True, **out)``.

    Parameters
    ----------
    active_mos : list[int] | None
        If provided, restrict the second-quantised Hamiltonian to the
        active window spanned by these MOs (1-indexed positions in the
        symmetry-adapted MO set, mirroring the molecular ``active_mo``
        argument).  Doubly occupied MOs outside the window are treated as
        frozen core (their mean-field contribution is folded into the
        active 1-electron Hamiltonian and a constant ``E_core``);
        unoccupied MOs outside the window are dropped.  Symmetry detection
        runs on the active-MO subset only.
    symmetry : bool
        If True (default) the supercell-as-molecule point group is
        detected and used to build the maximal Boolean Z2^k subgroup
        (Strategy A in the design notes).  Half-translations along
        even-mesh axes that map to supercell point-group elements
        (typically inversion through atom midpoints) are picked up
        automatically.

        If False, only spin parities and the explicit half-translation
        generators are used; the FermionOperator is built in the
        original (band, k_index, sigma) Bloch basis.
    """
    import time as _time
    _t0 = _time.time()
    def _stage(msg):
        print(f"  [{(_time.time()-_t0):7.1f}s] {msg}", flush=True)
    _stage('build_cell')
    cell = build_cell(atom, a, basis, kpts_mesh, pseudo=pseudo,
                      spin=spin, charge=charge, verbose=verbose)
    _stage('run_krhf')
    kmf, _ = run_krhf(cell, kpts_mesh, df=df, exxdiv=exxdiv)
    _stage('krhf done; e_tot=%.6f' % kmf.e_tot)

    if active_bands is not None:
        raise NotImplementedError(
            "active_bands restriction not yet implemented; pass active_bands=None for v1."
        )

    _stage('fold_to_supercell')
    mf_sc, band_k_order, n_k, n_bands = fold_to_supercell(kmf)
    _stage('fold done')
    nelec = int(round(mf_sc.cell.tot_electrons(nkpts=1)))
    nelec_up = (nelec + spin) // 2
    nelec_dn = (nelec - spin) // 2

    if not symmetry:
        # Bloch-basis path (no point-group descent).  Half-translations
        # are applied as explicit \u00b11 generators on (band, k, sigma) qubits.
        H, n_so, _ = supercell_fermion_hamiltonian(mf_sc, band_k_order, n_k, n_bands)
        spin_gens, spin_labels = spin_parity_generators(n_so)
        trans_gens, trans_labels = half_translation_generators(kpts_mesh, n_bands)
        gens = spin_gens + trans_gens
        labels = spin_labels + trans_labels
        signs = []
        for lbl in labels:
            if lbl == 'P_up':
                signs.append(1 if nelec_up % 2 == 0 else -1)
            elif lbl == 'P_dn':
                signs.append(1 if nelec_dn % 2 == 0 else -1)
            else:
                signs.append(1)
        return dict(
            fermion_hamiltonian=H, nspinorbital=n_so,
            nelectron_up=nelec_up, nelectron_down=nelec_dn,
            symmetry_generators=gens, signs=signs,
            symmetry_generator_labels=labels, name=name,
            _kmf=kmf, _mf_sc=mf_sc, _n_k=n_k, _n_bands=n_bands,
        )

    # ------------- Strategy A: molecular pipeline on supercell -------------
    _stage('build supercell Mol (symmetry detection)')
    mol_sc = _build_supercell_mol(mf_sc.cell, symmetry=True)
    _stage(f'mol_sc PG = {mol_sc.groupname}')

    # Symmetry-adapt the k2gamma MOs so each one lies in a single irrep of mol_sc
    S = mf_sc.get_ovlp()
    if np.iscomplexobj(S):
        S = S.real
    mo_coeff = np.asarray(mf_sc.mo_coeff)
    if mo_coeff.dtype != float:
        mo_coeff = mo_coeff.real
    mo_energy = np.asarray(mf_sc.mo_energy).real

    # Build half-supercell-translation generators in MO basis. For an
    # axis with even mesh size n_i, the Z_2 generator is the translation
    # by half the supercell period, T_{(n_i/2) a_i}, where a_i is the
    # primitive lattice vector along that axis.  This translation has
    # order 2 in the supercell periodic boundary conditions and acts as
    # +/-1 on every MO that descended from KRHF at a definite k-point.
    # Only those F2-combinations T_v = prod_i (T_{(n_i/2) a_i})^{v_i} that
    # COMMUTE with every PG irrep projector can coexist with the molecular
    # point-group adaptation.  On orthorhombic Bravais (cubic, tetragonal,
    # ...) every primitive passes; on FCC/BCC/diamond only certain
    # combinations (e.g. the body-diagonal glide) survive.
    even_axes = [i for i, m in enumerate(kpts_mesh)
                 if int(m) % 2 == 0 and int(m) >= 2]
    T_ao_primitives = []
    prim_a = np.asarray(cell.a, dtype=float)
    if cell.unit.lower().startswith('b'):
        prim_a_bohr = prim_a
    else:
        prim_a_bohr = prim_a / 0.52917720859
    for ax in even_axes:
        n_ax = int(kpts_mesh[ax])
        # Half of the supercell period along axis ax: shift by (n_ax // 2)
        # primitive lattice vectors.  For n_ax=2 this equals one primitive
        # lattice vector (preserving the original behaviour); for n_ax=4
        # it is two primitive vectors; etc.
        half_shift = (n_ax // 2) * prim_a_bohr[ax]
        atom_perm = _atom_permutation_under_translation(
            mf_sc.cell, half_shift
        )
        T_ao_primitives.append(_ao_translation_matrix(mf_sc.cell, atom_perm))

    if T_ao_primitives:
        _stage(f'building {len(T_ao_primitives)} primitive AO translations + irrep projectors')
        irrep_projs = _build_irrep_projectors(mol_sc, S)
        _stage(f'filtering PG-invariant translations among 2^{len(T_ao_primitives)}-1 combos')
        T_ao_list, T_label_combos = _filter_pg_invariant_translations(
            T_ao_primitives, irrep_projs
        )
        _stage(f'  -> {len(T_ao_list)} surviving combo(s): {T_label_combos}')
    else:
        T_ao_list, T_label_combos = [], []

    if T_ao_list:
        _stage('diagonalise translations in energy-degenerate blocks')
        mo_coeff, k_signs_per_mo = _diagonalise_translations_in_energy_blocks(
            mo_coeff, mo_energy, S, T_ao_list,
            energy_tol=symm_energy_tol,
        )
    else:
        k_signs_per_mo = None

    _stage('symmetry-adapt MOs')
    new_mo_coeff, label_orb_symm = _symmetry_adapt_mos(
        mol_sc, mo_coeff, mo_energy, S,
        energy_tol=symm_energy_tol, irrep_purity_tol=symm_purity_tol,
        k_signs_per_mo=k_signs_per_mo,
    )

    # ---------- Extra space-group generators (beyond mol_sc PG) ----------
    # Enumerate primitive-cell space-group involutions that close on the
    # supercell, lift to supercell-AO matrices via PySCF Wigner-D, and
    # accept those that are GF(2)-independent of the (energy, irrep,
    # k-sign) block partition already established.
    extra_signs_per_mo = None
    extra_op_labels = []
    try:
        _stage('enumerate extra space-group involutions')
        extra_ops = _extra_supercell_pg_ops(cell, mf_sc, S)
        _stage(f'  -> {len(extra_ops)} candidate involution(s) close on supercell')
    except Exception as _e:
        extra_ops = []
        _stage(f'  space-group enumeration skipped: {_e!r}')

    if extra_ops:
        # Build current MO block partition: group by (energy, irrep, k-sign)
        n_mo_total = new_mo_coeff.shape[1]
        order = np.argsort(mo_energy)
        def _mo_key(j):
            k = (round(float(mo_energy[j]) / max(symm_energy_tol, 1e-9)),
                 label_orb_symm[j])
            if k_signs_per_mo is not None:
                k = k + (tuple(int(x) for x in k_signs_per_mo[j]),)
            return k
        cur_blocks = []
        cur_key = None
        for j in order:
            j = int(j)
            kj = _mo_key(j)
            if kj != cur_key:
                cur_blocks.append([j])
                cur_key = kj
            else:
                cur_blocks[-1].append(j)
        _stage(f'  block partition has {len(cur_blocks)} blocks (n_mo={n_mo_total})')
        new_mo_coeff, _refined_blocks, accepted = _refine_blocks_with_extra_ops(
            new_mo_coeff, S, extra_ops, cur_blocks,
        )
        if accepted:
            extra_op_labels = [lbl for lbl, _ in accepted]
            extra_signs_per_mo = np.stack(
                [sv for _, sv in accepted], axis=1
            )  # (n_mo, n_extra)
            _stage(f'  -> {len(accepted)} extra independent generator(s): {extra_op_labels}')
        else:
            _stage('  -> 0 extra independent generators (all dependent on existing)')

    if active_mos is None:
        # Full-space path: build the FermionOperator in the symmetry-adapted
        # MO basis using PERIODIC density-fitted ERIs.
        _stage('full-space fermion Hamiltonian build')
        H = _fermion_hamiltonian_in_mo_basis(mf_sc, new_mo_coeff, kmf=kmf)
        n_mo = new_mo_coeff.shape[1]
        n_so = 2 * n_mo
        active_label_orb_symm = label_orb_symm
        nelec_up_eff = nelec_up
        nelec_dn_eff = nelec_dn
        e_core = 0.0
        active_indices_0 = list(range(n_mo))
    else:
        # Active-space path: rebuild integrals on the active MO block only.
        # Carry frozen-core mean-field correction into h1 and into E_core.
        _stage(f'active-space fermion Hamiltonian build (n_act={len(active_mos)})')
        active_indices_0 = sorted(int(i) - 1 for i in active_mos)
        H, n_mo, nelec_in_active, e_core, _act_sorted = (
            _active_space_fermion_hamiltonian(mf_sc, new_mo_coeff, active_indices_0,
                                              kmf=kmf, integral_backend=integral_backend)
        )
        n_so = 2 * n_mo
        # Active-window electron count and (Sz, parity).
        # Spin within the active window is the global spin (frozen core is
        # closed-shell by construction).
        nelec_up_eff = (nelec_in_active + spin) // 2
        nelec_dn_eff = (nelec_in_active - spin) // 2
        # In periodic active-space calculations the frozen-core Coulomb field
        # can make the h1_act diagonal positive, so the vacuum (0 active
        # electrons) has lower energy than the physical ground state.  Adding
        # this penalty ensures the physical (nelec_up_eff, nelec_dn_eff) sector
        # is the minimum of the qubit Hamiltonian (penalty = 0 there, positive
        # everywhere else), without changing any eigenvalue in that sector.
        H = H + _active_space_number_penalty(
            n_mo, nelec_up_eff, nelec_dn_eff, strength=2.0
        )
        active_label_orb_symm = [label_orb_symm[i] for i in active_indices_0]

    # Detect Boolean Z2^k generators via the molecular pipeline
    _stage('detect supercell-PG generators')
    gens, signs, labels, irrep = _generators_from_supercell_symmetry(
        mol_sc, active_label_orb_symm, n_so, nelec_up_eff, nelec_dn_eff
    )
    _stage('done')

    # Append half-translation generators built from the k-signs of each
    # MO under each surviving (PG-invariant) F2-combination of primitive
    # translations.  Each generator is a +/-1 vector of length n_so with
    # the same +/-1 pattern on alpha (even) and beta (odd) qubits.
    #
    # GF(2) independence check: a half-translation that coincides with a
    # supercell point-group element (e.g. inversion in a centrosymmetric
    # crystal) would already be captured by _generators_from_supercell_symmetry.
    # We skip any generator whose binary support lies in the span of the
    # generators already collected, to avoid adding a linearly dependent
    # (and potentially sign-conflicting) vector.
    if k_signs_per_mo is not None:
        active_k_signs = k_signs_per_mo[active_indices_0]  # (n_act, n_combos)
        mo_occ_active = np.asarray(mf_sc.mo_occ)[active_indices_0]
        existing_b = (
            np.array([(np.asarray(g) < 0).astype(np.uint8) for g in gens],
                     dtype=np.uint8)
            if gens else np.zeros((0, n_so), dtype=np.uint8)
        )
        base_rank = _gf2_rank(existing_b) if existing_b.size else 0
        for combo_local, combo in enumerate(T_label_combos):
            sign_per_mo = active_k_signs[:, combo_local]
            gen_vec = np.empty(n_so, dtype=int)
            for p, s_p in enumerate(sign_per_mo):
                gen_vec[2 * p] = int(s_p)
                gen_vec[2 * p + 1] = int(s_p)
            new_b = (gen_vec < 0).astype(np.uint8)
            stacked = (np.vstack([existing_b, new_b[None, :]])
                       if existing_b.size else new_b[None, :])
            if _gf2_rank(stacked) == base_rank:
                continue  # GF(2)-dependent on existing generators
            # Sign: (-1)^{n_electrons_in_antisymmetric_k-sector} at the HF
            # reference.  For KRHF, mo_occ values are 0 or 2 (closed-shell),
            # so n_antisym is always even and the sign is always +1.  The
            # explicit computation is kept for correctness with future
            # open-shell extensions (KROHF etc.).
            n_antisym = int(sum(
                round(float(mo_occ_active[p]))
                for p in range(len(sign_per_mo)) if sign_per_mo[p] == -1
            ))
            signs.append((-1) ** n_antisym)
            gens.append(gen_vec)
            ax_str = '+'.join(f'a{even_axes[i]}' for i in combo)
            labels.append(f'T_({ax_str})/2')
            existing_b = stacked
            base_rank += 1

    # Extra space-group generators (must come AFTER the half-translation
    # block so that their GF(2) independence is assessed against the
    # full set of existing generators (parities + PG + half-translations).
    if extra_signs_per_mo is not None:
        active_extra = extra_signs_per_mo[active_indices_0]  # (n_act, n_extra)
        # GF(2) basis from all currently-appended generators (full n_so each).
        existing_b = np.array(
            [(np.asarray(g) < 0).astype(int) for g in gens], dtype=np.uint8
        ) % 2 if gens else np.zeros((0, n_so), dtype=np.uint8)
        base_rank = _gf2_rank(existing_b) if existing_b.size else 0
        for k_extra, lbl in enumerate(extra_op_labels):
            sign_per_mo = active_extra[:, k_extra]
            gen_vec = np.empty(n_so, dtype=int)
            for p, s_p in enumerate(sign_per_mo):
                gen_vec[2 * p] = int(s_p)
                gen_vec[2 * p + 1] = int(s_p)
            new_b = (gen_vec < 0).astype(np.uint8)
            stacked = np.vstack([existing_b, new_b[None, :]]) if existing_b.size else new_b[None, :]
            if _gf2_rank(stacked) == base_rank:
                continue  # GF(2)-dependent on existing -> skip
            gens.append(gen_vec)
            signs.append(1)
            labels.append(lbl)
            existing_b = stacked
            base_rank += 1

    return dict(
        fermion_hamiltonian=H, nspinorbital=n_so,
        nelectron_up=nelec_up_eff, nelectron_down=nelec_dn_eff,
        symmetry_generators=gens, signs=signs,
        symmetry_generator_labels=labels, name=name,
        _kmf=kmf, _mf_sc=mf_sc, _mol_sc=mol_sc,
        _n_k=n_k, _n_bands=n_bands,
        _label_orb_symm=active_label_orb_symm, _irrep=irrep,
        _full_label_orb_symm=label_orb_symm,
        _active_indices_0=active_indices_0,
        _e_core=e_core,
        _new_mo_coeff=new_mo_coeff,
    )
