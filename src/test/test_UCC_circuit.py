from ..quantumsymmetry import Encoding
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from numpy import isclose

def UCC_VQE(atom, basis, charge = 0, spin = 0, CAS = None, encoding_type = 'SAE'):
    driver = PySCFDriver(
        atom= atom,
        unit=DistanceUnit.ANGSTROM,
        charge=charge,
        spin=spin,
        basis=basis)
    
    problem = driver.run()

    if encoding_type == 'SAE':
        encoding = Encoding(atom = atom, basis = basis, charge = charge, spin = spin, CAS = CAS, output_format = 'qiskit')
        mapper = encoding.qiskit_mapper
        initial_state = encoding.HF_circuit
    
    elif encoding_type == 'JW':
        mapper = JordanWignerMapper()
        initial_state = HartreeFock(
            num_spatial_orbitals = problem.num_spatial_orbitals,
            num_particles = problem.num_particles,
            qubit_mapper = mapper
            )
    
    ansatz = UCCSD(
        num_spatial_orbitals = problem.num_spatial_orbitals,
        num_particles = problem.num_particles,
        qubit_mapper = mapper,
        initial_state = initial_state
        )

    vqe = VQE(
        estimator = Estimator(),
        ansatz = ansatz,
        optimizer = SLSQP()
   )

    solver = GroundStateEigensolver(mapper, vqe)
    vqe_result = solver.solve(problem)

    energy = vqe_result.total_energies[0]
    #qubits = ansatz.num_qubits
    #depth = ansatz.decompose().decompose().decompose().decompose().depth()
    
    return energy

def test_UCC_circuit_H3():
    #Parameters for H3+ in the sto-3g basis
    atom = 'H 0 0.377 0; H -0.435 -0.377 0; H 0.435 -0.377 0'
    charge = 1
    spin = 0
    basis = 'sto3g'
    
    energy_JW = UCC_VQE(atom = atom, basis = basis, charge = charge, encoding_type = 'JW')
    energy_SAE = UCC_VQE(atom = atom, basis = basis, charge = charge, encoding_type = 'SAE')
    
    assert isclose(energy_JW, energy_SAE)

def test_UCC_circuit_H2O_CAS():
    #Parameters for H2O in the sto-3g basis with CAS(4, 4)
    atom = 'O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786'
    basis = 'sto3g'
    charge = 0
    spin = 0
    CAS = (4, 4)

    energy = UCC_VQE(atom = atom, basis = basis, charge = charge, spin = spin, CAS = CAS, encoding_type = 'SAE')

    assert isclose(energy, -74.97227790592032)