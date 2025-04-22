from ..quantumsymmetry import Encoding
from numpy.linalg import eigvalsh
from numpy import isclose
from pyscf import gto, scf, fci, mcscf

def compare_energies(atom, basis, charge = 0, spin = 0, CAS = None, active_mo = None, symmetry = True):
    #Finds lowest eigenvalue of symmetry-adapted qubit Hamiltonian
    encoding = Encoding(atom = atom, basis = basis, charge = charge, spin = spin, symmetry= symmetry, verbose = False, CAS = CAS, output_format = 'qiskit', active_mo = active_mo)
    H = encoding.hamiltonian
    H = H.to_matrix()
    print(type(H))
    lowest_eigenvalue = eigvalsh(H)[0]

    #Calculates full configuration interaction/CASCI ground state energy from PySCF
    mol = gto.Mole(atom = atom, basis = basis, charge = charge, spin = spin)
    mol.build(verbose = 0)
    mf = scf.RHF(mol).run(verbose = 0)
    if CAS == None:
        ci = fci.FCI(mf).run(verbose = 0)
        ground_state_energy = ci.e_tot
    else:
        cas = mcscf.CASCI(mf, *CAS)
        if active_mo == None:
            cas.run(verbose = 0)
        else:
            mo = cas.sort_mo(active_mo)
            cas.kernel(mo)
        ground_state_energy = cas.e_tot
    print(lowest_eigenvalue, ground_state_energy)
    return isclose(lowest_eigenvalue, ground_state_energy)

#Hydrogen molecule (H2) with STO-3G basis
def test_H2():
    assert compare_energies(
    atom = 'H 0 0 0; H 0.7414 0 0',
    basis = 'sto-3g')

#Hydrogen molecule (H2) with double-zeta basis
def test_H2_DZ():
    assert compare_energies(
    atom = 'H 0 0 0; H 0.7414 0 0',
    basis = 'dz')

#Trihydrogen diamer (H3+) with STO-3G basis
def test_H3():
    assert compare_energies(
    atom = 'H 0 0.377 0; H -0.435 -0.377 0; H 0.435 -0.377 0',
    basis = 'sto-3g',
    charge = 1)

#Water molecule (H2O) with STO-3G basis
def test_H2O():
    assert compare_energies(
    atom = 'O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786',
    basis = 'sto-3g')

#Water molecule (H2O) with STO-3G basis and CAS(4, 4)
def test_H2O_CAS():
    assert compare_energies(
    atom = 'O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786',
    basis = 'sto-3g',
    CAS = (4, 4))

#Water molecule (H2O) with STO-3G basis, CAS(4, 4) and no symmetry
def test_H2O_CAS_no_symmetry():
    assert compare_energies(
    atom = 'O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786',
    basis = 'sto-3g',
    CAS = (4, 4),
    symmetry = False)

#Water molecule (H2O) with STO-3G basis and CAS(4, 4) and active molecular orbital selection
def test_H2O_CAS_active_mo():
    assert compare_energies(
    atom = 'O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786',
    basis = 'sto-3g',
    active_mo = [3,4,6,7],
    CAS = (4, 4))

#Oxygen molecule (O2) with STO-3G basis and CAS(4, 4)
def test_O2_CAS():
    assert compare_energies(
    atom = 'O 0 0 0; O 0 0 1.189',
    spin = 2,
    basis = 'sto-3g',
    CAS = (4, 4))

#Oxygen molecule (O2) with STO-3G basis, CAS(4, 4) and no symmetry
def test_O2_CAS_no_symmetry():
    assert compare_energies(
    atom = 'O 0 0 0; O 0 0 1.189',
    spin = 2,
    basis = 'sto-3g',
    CAS = (4, 4),
    symmetry = False)

#Beryllium hydrade (BeH2) with STO-3G basis and CAS(4, 4)
def test_BeH2_CAS():
    assert compare_energies(
    atom = (('H', (-1.326, 0., 0.)), ('Be', (0., 0., 0.)), ('H', (1.326, 0., 0.))),
    basis = 'sto-3g',
    CAS = (4, 4))

#Ethene molecule (C₂H₄) with STO-3G basis and CAS(8, 8)
def test_BeH2_CAS():
    assert compare_energies(
    atom = 'C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321',
    basis = 'sto-3g',
    CAS = (8, 8))