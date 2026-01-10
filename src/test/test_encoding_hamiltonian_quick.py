# tests/test_quick_cas.py

from ..quantumsymmetry import Encoding
from numpy.linalg import eigvalsh
from numpy import isclose
import numpy as np
from pyscf import gto, scf, mcscf, fci

def compare_energies_quick(atom, basis, charge=0, spin=0, CAS=None, active_mo=None, symmetry=True):
    # SAE/CAS Hamiltonian via Encoding with quick_CAS=True
    encoding = Encoding(
        atom=atom, basis=basis, charge=charge, spin=spin,
        symmetry=symmetry, verbose=False, CAS=CAS,
        quick_CAS=True,                      # <- key difference
        output_format='qiskit', active_mo=active_mo
    )
    H = encoding.hamiltonian.to_matrix()
    lowest_eigenvalue = eigvalsh(H)[0]

    # Reference: PySCF FCI/CASCI
    mol = gto.Mole(atom=atom, basis=basis, charge=charge, spin=spin)
    mol.build(verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    if CAS is None:
        ci = fci.FCI(mf).run(verbose=0)
        ground_state_energy = ci.e_tot
    else:
        cas = mcscf.CASCI(mf, *CAS)
        if active_mo is None:
            cas.run(verbose=0)
        else:
            mo = cas.sort_mo(active_mo)
            cas.kernel(mo)
        ground_state_energy = cas.e_tot

    return isclose(lowest_eigenvalue, ground_state_energy)

# -------------------------
# Quick-CAS mirrored tests
# -------------------------

# H2O, STO-3G, CAS(4,4)
def test_H2O_CAS_quick():
    assert compare_energies_quick(
        atom='O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786',
        basis='sto-3g',
        CAS=(4, 4)
    )

def test_H2_CAS_quick_BK_spectrum_invariant():
    atom = 'H 0 0 0; H 0.7414 0 0'
    basis = 'sto-3g'
    CAS = (2, 2)
    enc_jw = Encoding(atom=atom, basis=basis, verbose=False, CAS=CAS, quick_CAS=True, output_format='qiskit', bravyi_kitaev=False)
    enc_bk = Encoding(atom=atom, basis=basis, verbose=False, CAS=CAS, quick_CAS=True, output_format='qiskit', bravyi_kitaev=True)
    e1 = eigvalsh(enc_jw.hamiltonian.to_matrix())
    e2 = eigvalsh(enc_bk.hamiltonian.to_matrix())
    assert np.allclose(np.sort(e1), np.sort(e2))

# H2O, STO-3G, CAS(4,4), no symmetry
def test_H2O_CAS_no_symmetry_quick():
    assert compare_energies_quick(
        atom='O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786',
        basis='sto-3g',
        CAS=(4, 4),
        symmetry=False
    )

# H2O, STO-3G, CAS(4,4), active MO selection
def test_H2O_CAS_active_mo_quick():
    assert compare_energies_quick(
        atom='O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786',
        basis='sto-3g',
        active_mo=[3, 4, 6, 7],
        CAS=(4, 4)
    )

# O2, STO-3G, CAS(4,4)
def test_O2_CAS_quick():
    assert compare_energies_quick(
        atom='O 0 0 0; O 0 0 1.189',
        spin=2,
        basis='sto-3g',
        CAS=(4, 4)
    )

# O2, STO-3G, CAS(4,4), no symmetry
def test_O2_CAS_no_symmetry_quick():
    assert compare_energies_quick(
        atom='O 0 0 0; O 0 0 1.189',
        spin=2,
        basis='sto-3g',
        CAS=(4, 4),
        symmetry=False
    )

# BeH2, STO-3G, CAS(4,4)
def test_BeH2_CAS_quick():
    assert compare_energies_quick(
        atom=(('H', (-1.326, 0., 0.)), ('Be', (0., 0., 0.)), ('H', (1.326, 0., 0.))),
        basis='sto-3g',
        CAS=(4, 4)
    )

# C2H4 (ethene), STO-3G, CAS(8,8)
def test_C2H4_CAS_quick():
    assert compare_energies_quick(
        atom='C 0 0 0.6695; C 0 0 -0.6695; '
             'H 0 0.9289 1.2321; H 0 -0.9289 1.2321; '
             'H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321',
        basis='sto-3g',
        CAS=(8, 8)
    )
