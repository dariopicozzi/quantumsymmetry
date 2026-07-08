"""Symmetry sieving of computational basis states and excitation listing.

Selects the active leaves consistent with the molecular symmetries and
derives the excitation operators used to build chemistry ansatze.
"""

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from functools import partial
from dataclasses import dataclass
from typing import List, Optional


__all__ = [
    'sieve_states_by_symmetry',
    'get_excitations_from_states',
    'dressing_generators',
]


def sieve_states_by_symmetry(atom, basis, charge=0, spin=0, irrep=None, order='standard'):
    """Filter computational-basis states by spatial and spin symmetry using PySCF.

    Requires the ``pyscf`` and ``quantumsymmetry`` packages.
    """
    from pyscf import gto, scf, symm
    from quantumsymmetry.core import find_ground_state_irrep, get_character_table

    mol = gto.Mole()
    mol.atom = atom
    mol.symmetry = True
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.build()
    if mol.groupname == 'Dooh' or mol.groupname == 'SO3':
        mol.symmetry = 'D2h'
        mol.build()
    if mol.groupname == 'Coov':
        mol.symmetry = 'C2v'
        mol.build()
    mf = scf.RHF(mol)
    mf.kernel()

    label_orb_symm = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
    character_table, conj_labels, irrep_labels, conj_descriptions = get_character_table(mol.groupname)
    if irrep is None:
        irrep = find_ground_state_irrep(label_orb_symm, mf.mo_occ, character_table, irrep_labels)
    n_elec = mol.nelectron
    n_alpha = (mol.nelectron + mol.spin)//2
    n_beta = (mol.nelectron - mol.spin)//2

    output = []

    if order == 'standard':  
        for state in range(4**mol.nao):
            bitstring = bin(state)[2:].zfill(2*mol.nao)
            bitstring = [int(x) for x in bitstring]
            state_n_elec = sum(bitstring)
            if state_n_elec == n_elec:
                state_n_alpha = sum(bitstring[x] for x in range(0, len(bitstring), 2))
                state_n_beta = n_elec - n_alpha
                if state_n_alpha == n_alpha and state_n_beta == n_beta:
                    state_irrep = find_ground_state_irrep(label_orb_symm, [bitstring[2*x] + bitstring[2*x+1] for x in range(mol.nao)][::-1], character_table, irrep_labels)
                    if state_irrep == irrep:
                        output.append(state)

    elif order == 'qiskit':
        for state in range(4**mol.nao):
            bitstring = bin(state)[2:].zfill(2*mol.nao)
            bitstring = [int(x) for x in bitstring]
            state_n_elec = sum(bitstring)
            if state_n_elec == n_elec:
                state_n_alpha = sum(bitstring[:mol.nao])
                state_n_beta = n_elec - n_alpha
                if state_n_alpha == n_alpha and state_n_beta == n_beta:
                    state_irrep = find_ground_state_irrep(label_orb_symm, [bitstring[x] + bitstring[mol.nao + x] for x in range(mol.nao)][::-1], character_table, irrep_labels)
                    if state_irrep == irrep:
                        output.append(state)

    return output     

def get_excitations_from_states(n, states, reference_state=None):
    """Enumerate excitations relative to a reference determinant.

    Requires the ``pyscf`` package.
    """
    if reference_state is None:
        reference_state = states[0]
    ground_state_bitstring = bin(reference_state)[2:].zfill(n)
    ground_state_bitstring = [int(x) for x in ground_state_bitstring][::-1]
    excitations = []
    for state in states:
        bitstring = bin(state)[2:].zfill(n)
        bitstring = [int(x) for x in bitstring][::-1]
        excitation1,  excitation2 = [], []
        for i in range(n):
            if ground_state_bitstring[i] == 1 and bitstring[i] == 0:
                excitation1.append(i)
            elif ground_state_bitstring[i] == 0 and bitstring[i] == 1:
                excitation2.append(i)
        if excitation1 != [] and excitation2 != []:
            excitations.append((tuple(excitation1), tuple(excitation2)))
    return excitations


def dressing_generators(encoding, P, Q, tol=1e-9, gs_tol=1e-8):
    """Symmetry-adapted P->Q excitation generators (``A^3 = A``) for a dressing.

    The chemistry bridge to the dressing layer: builds the generalized single-
    and double-excitation generators ``A = i(T - T^dag)`` over the molecular
    spin-orbitals, maps each through the symmetry-adapted ``encoding``, and
    screens them down to the off-block ``P``<->``Q`` tangent space.  Generators
    that leave the sector ``P u Q``, that act only within ``P``, or that are
    null are dropped, and the survivors are pruned by ordered Gram-Schmidt on
    their off-block parts ``Q A P``, so the count equals the off-block tangent
    dimension.  Each surviving generator satisfies ``A^3 = A``.

    ``P`` and ``Q`` are computational-basis indices (the model space and its
    complement) in the encoded space.  Returns a list of sparse generator
    matrices, ready for :func:`~quantumsymmetry.make_dressing_pool` and hence
    :func:`~quantumsymmetry.decoupling_surrogate`.  Requires ``openfermion``.

    Example
    -------
    >>> from quantumsymmetry import (Encoding, make_dressing_pool, decoupling_surrogate,
    ...                              dressing_generators)
    >>> enc = Encoding(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g')
    >>> pool = make_dressing_pool(dressing_generators(enc, P, Q))
    """
    from itertools import combinations
    from openfermion import FermionOperator, QubitOperator, get_sparse_operator
    import scipy.sparse as sp

    P = [int(p) for p in P]
    Q = [int(q) for q in Q]
    support = sorted(set(P) | set(Q))
    n_so = encoding.nspinorbital
    n_qubits = n_so - len(encoding.target_qubits)
    out_idx = np.array([i for i in range(1 << n_qubits) if i not in support])

    def encoded(fermionic):
        op = encoding.apply(fermionic)
        if op is None:
            return None
        if isinstance(op, QubitOperator):
            return get_sparse_operator(op, n_qubits=n_qubits).tocsr()
        return sp.csr_matrix(op.to_matrix(sparse=True))     # qiskit SparsePauliOp

    def candidates():
        # Generalized singles and doubles, pre-filtered to conserve N_alpha and
        # N_beta (spin = index parity); the rest leak out of the sector anyway.
        for i, a in combinations(range(n_so), 2):
            if i % 2 == a % 2:
                T = FermionOperator(((a, 1), (i, 0)))
                yield 1j * (T - FermionOperator(((i, 1), (a, 0))))
        pairs = list(combinations(range(n_so), 2))
        for (i, j), (a, b) in combinations(pairs, 2):
            if sorted((i % 2, j % 2)) == sorted((a % 2, b % 2)):
                T = FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)))
                yield 1j * (T - FermionOperator(((i, 1), (j, 1), (b, 0), (a, 0))))

    generators, gs_basis = [], []
    for fermionic in candidates():
        A = encoded(fermionic)
        if A is None or A.nnz == 0 or np.abs(A.data).max() < tol:
            continue
        leak = A[out_idx][:, support] if out_idx.size else None
        if leak is not None and leak.nnz and np.abs(leak.data).max() > tol:
            continue                                        # leaves the sector P u Q
        offblock = A[Q][:, P]
        if not (offblock.nnz and np.abs(offblock.data).max() > tol):
            continue                                        # no P <-> Q coupling
        vec = np.asarray(offblock.todense()).ravel()
        v = vec / np.linalg.norm(vec)
        for _ in range(2):                                  # ordered Gram-Schmidt
            for b in gs_basis:
                v = v - np.vdot(b, v) * b
        if np.linalg.norm(v) < gs_tol:
            continue                                        # off-block action already spanned
        gs_basis.append(v / np.linalg.norm(v))
        generators.append(A)
    return generators
