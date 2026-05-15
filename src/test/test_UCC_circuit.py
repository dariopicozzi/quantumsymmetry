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

def UCC_VQE(atom, basis, charge = 0, spin = 0, CAS = None, quick_CAS = False, encoding_type = 'SAE', symmetry = True):
    driver = PySCFDriver(
        atom= atom,
        unit=DistanceUnit.ANGSTROM,
        charge=charge,
        spin=spin,
        basis=basis)
    
    problem = driver.run()

    if encoding_type == 'SAE':
        encoding = Encoding(atom = atom, basis = basis, charge = charge, spin = spin, CAS = CAS, quick_CAS = quick_CAS, symmetry=symmetry, output_format = 'qiskit')
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

    energy = UCC_VQE(atom = atom, basis = basis, charge = charge, encoding_type = 'SAE')

    # FCI (exact) energy for H3+ in STO-3G at this geometry.
    # The JW UCCSD VQE is not used here: with a full (unreduced) qubit space
    # SLSQP reliably gets stuck ~11.6 mHa above FCI for this molecule, making
    # a JW-vs-SAE comparison an unreliable test.
    assert isclose(energy, -1.2613448700)

def test_UCC_circuit_H2O_CAS():
    #Parameters for H2O in the sto-3g basis with CAS(4, 4)
    atom = 'O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786'
    basis = 'sto3g'
    charge = 0
    spin = 0
    CAS = (4, 4)

    energy = UCC_VQE(atom = atom, basis = basis, charge = charge, spin = spin, CAS = CAS, encoding_type = 'SAE')

    assert isclose(energy, -74.97227790592032)

def test_UCC_circuit_H2O_CAS_quick():
    #Parameters for H2O in the sto-3g basis with CAS(4, 4)
    atom = 'O 0 0 0.1197; H 0 0.7616 -0.4786; H 0 -0.7616 -0.4786'
    basis = 'sto3g'
    charge = 0
    spin = 0
    CAS = (4, 4)

    energy = UCC_VQE(atom = atom, basis = basis, charge = charge, spin = spin, CAS = CAS, quick_CAS = True, encoding_type = 'SAE')

    assert isclose(energy, -74.97227790592032)

def test_UCC_circuit_H2_no_symmetry():
    #Parameters for H2 in the sto-3g basis with no symmetry
    atom = 'H 0 0 0; H 0.7414 0 0'
    charge = 0
    spin = 0
    basis = 'dz'
    CAS = (2, 2)
    symmetry = False

    energy = UCC_VQE(atom = atom, basis = basis, charge = charge, spin = spin, encoding_type = 'SAE', CAS = CAS, symmetry = symmetry)
    
    assert isclose(energy, -1.1331269927574912)
