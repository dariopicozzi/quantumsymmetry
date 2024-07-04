from quantumsymmetry import reduced_hamiltonian, make_encoding, apply_encoding
from numpy.linalg import eigvalsh
from numpy import isclose
from pyscf import gto, scf, fci, mcscf
from openfermion import FermionOperator, QubitOperator

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

def test_make_encoding():
    #Finds lowest eigenvalue of symmetry-adapted qubit Hamiltonian
    encoding = make_encoding(atom = 'H 0 0 0; H 0.7414 0 0', basis = 'sto-3g')
    operator = FermionOperator('0^ 0')
    reduced_operator = apply_encoding(operator = operator, encoding = encoding)
    assert reduced_operator == 0.5*QubitOperator('Z0') + 0.5

def test_make_encoding_CAS():
    #Finds lowest eigenvalue of symmetry-adapted qubit Hamiltonian
    encoding = make_encoding(atom = 'O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786', basis = 'sto-3g', CAS = (4, 4))
    operator = FermionOperator('4^ 6^ 6 4')
    reduced_operator = apply_encoding(operator = operator, encoding = encoding)
    assert reduced_operator == -0.5*QubitOperator('Z0 Z1 Z3') + 0.5

def test_H2():
    assert compare_energies(atom = 'H 0 0 0; H 0.7414 0 0', basis = 'sto-3g')

def test_H2_DZ():
    assert compare_energies(atom = 'H 0 0 0; H 0.7414 0 0', basis = 'dz')

def test_H3():
    assert compare_energies(atom = 'H 0 0.377 0; H -0.435 -0.377 0; H 0.435 -0.377 0', basis = 'sto-3g', spin = 0, charge = 1)

def test_H2O():
    assert compare_energies(atom = 'O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786', basis = 'sto-3g')

def test_H2O_CAS():
    assert compare_energies(atom = 'O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786', basis = 'sto-3g', CAS = (4, 4))