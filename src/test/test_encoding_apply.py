from ..quantumsymmetry import Encoding
from openfermion import FermionOperator, QubitOperator

def test_encoding_apply():
    #Finds lowest eigenvalue of symmetry-adapted qubit Hamiltonian
    encoding = Encoding(atom = 'H 0 0 0; H 0.7414 0 0', basis = 'sto-3g')
    operator = FermionOperator('0^ 0')
    reduced_operator = encoding.apply(operator)
    assert reduced_operator == 0.5*QubitOperator('Z0') + 0.5

def test_encoding_apply_CAS():
    #Finds lowest eigenvalue of symmetry-adapted qubit Hamiltonian
    encoding = Encoding(atom = 'O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786', basis = 'sto-3g', CAS = (4, 4))
    operator = FermionOperator('4^ 6^ 6 4')
    reduced_operator = encoding.apply(operator)
    assert reduced_operator == -0.5*QubitOperator('Z0 Z1 Z3') + 0.5