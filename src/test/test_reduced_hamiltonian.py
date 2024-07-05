from ..quantumsymmetry import reduced_hamiltonian
from numpy.linalg import eigvalsh
from numpy import isclose
from pyscf import gto, scf, fci, mcscf

def compare_energies(atom, basis, charge = 0, spin = 0, CAS = None):
    #Finds lowest eigenvalue of symmetry-adapted qubit Hamiltonian
    H = reduced_hamiltonian(atom = atom, basis = basis, charge = charge, spin = spin, verbose = False, CAS = CAS, output_format = 'qiskit')
    H = H.to_matrix()
    lowest_eigenvalue = eigvalsh(H)[0]

    #Calculates full configuration interaction/CASCI ground state energy from PySCF
    mol = gto.Mole(atom = atom, basis = basis, charge = charge, spin = spin)
    mol.build(verbose = 0)
    mf = scf.RHF(mol).run(verbose = 0)
    if CAS == None:
        ci = fci.FCI(mf).run(verbose = 0)
        ground_state_energy = ci.e_tot
    else:
        cas = mcscf.CASCI(mf, *CAS).run(verbose = 0)
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

#Oxygen molecule (O2) with STO-3G basis and CAS(4, 4)
def test_O2_CAS():
    assert compare_energies(
    atom = 'O 0 0 0; O 0 0 1.189',
    spin = 2,
    basis = 'sto-3g',
    CAS = (4, 4))

#Beryllium hydrade (BeH2) with STO-3G basis and CAS(4, 4)
def test_BeH2_CAS():
    assert compare_energies(
    atom = (('H', (-1.326, 0., 0.)), ('Be', (0., 0., 0.)), ('H', (1.326, 0., 0.))),
    basis = 'sto-3g',
    CAS = (4, 4))