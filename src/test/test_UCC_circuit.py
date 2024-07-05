from ..quantumsymmetry import reduced_hamiltonian, UCC_SAE_circuit
from ..quantumsymmetry.qiskit_converter import get_num_particles_spin_orbitals
from qiskit import BasicAer, Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE
from qiskit_nature.circuit.library import UCC, HartreeFock
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.drivers import UnitsType
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.circuit.library import UCC
from qiskit_nature.circuit.library import HartreeFock
from pyscf import gto, scf
from numpy import isclose

def UCC_JW_VQE(atom, basis, charge = 0, spin = 0):
    num_particles, num_spin_orbitals = get_num_particles_spin_orbitals(atom = atom, charge = charge, spin = spin, basis = basis)

    driver = PySCFDriver(
        atom= atom,
        unit=UnitsType.ANGSTROM,
        charge=charge, spin=spin,
        basis=basis)

    qubit_transformation = QubitConverter(JordanWignerMapper())

    qubit_operator = qubit_transformation.convert(ElectronicStructureProblem(driver).second_q_ops()[0])

    initial_state = HartreeFock(
        qubit_operator.num_qubits,
        num_particles,
        qubit_transformation)

    ansatz = UCC(
        excitations = "sd",
        num_particles = num_particles,
        num_spin_orbitals = qubit_operator.num_qubits,
        initial_state = initial_state,
        qubit_converter = qubit_transformation,
        reps = 1,
    )

    vqe = VQE(
        ansatz = ansatz,
        quantum_instance = Aer.get_backend("aer_simulator_statevector")
        )

    vqe_result =vqe.compute_minimum_eigenvalue(qubit_operator)

    energy1 = vqe_result.optimal_value
    qubits1 = ansatz.num_qubits
    depth1 = ansatz.decompose().decompose().decompose().decompose().depth()
    
    return energy1
    
def UCC_SAE_VQE(atom, basis, charge = 0, spin = 0, CAS = None):
    qubit_operator = reduced_hamiltonian(atom = atom, basis = basis, charge = charge, spin = spin, CAS = CAS, output_format = 'qiskit', verbose = False)
    ansatz = UCC_SAE_circuit(atom = atom, basis = basis, charge = charge, spin = spin, CAS = CAS)

    backend = BasicAer.get_backend('statevector_simulator')
    quantum_instance=QuantumInstance(backend=backend)

    initial_point = [0]*ansatz.num_parameters

    vqe = VQE(
        ansatz = ansatz,
        initial_point = initial_point,
            quantum_instance=quantum_instance)

    vqe_result = vqe.compute_minimum_eigenvalue(qubit_operator)

    energy2 = vqe_result.optimal_value
    qubits2 = ansatz.num_qubits
    depth2 = ansatz.decompose().decompose().decompose().decompose().depth()
    
    return energy2

def get_energy_nuc(atom, basis, charge = 0, spin = 0):
    mol = gto.Mole()
    mol.atom = atom
    mol.symmetry = True
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    energy_nuc = mf.energy_nuc()
    
    return energy_nuc

def test_UCC_circuit_H3():
    #Parameters for H3+ in the sto-3g basis
    atom = 'H 0 0.377 0; H -0.435 -0.377 0; H 0.435 -0.377 0'
    charge = 1
    basis = 'sto3g'

    energy1 = UCC_JW_VQE(atom = atom, basis = basis, charge = charge)
    energy2 = UCC_SAE_VQE(atom = atom, basis = basis, charge = charge)
    energy_nuc = get_energy_nuc(atom = atom, basis = basis, charge = charge)
    
    assert isclose(energy1, energy2 - energy_nuc)

def test_UCC_circuit_H2O_CAS():
    #Parameters for H3+ in the sto-3g basis
    atom = 'O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786'
    basis = 'sto3g'
    CAS = (4, 4)

    energy = UCC_SAE_VQE(atom = atom, basis = basis, CAS = CAS)
    
    assert isclose(energy, -74.97227790592032)