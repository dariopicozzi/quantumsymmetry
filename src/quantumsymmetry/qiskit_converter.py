from .core import *
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit.circuit.quantumcircuit import QuantumCircuit
from pyscf import gto, scf
from itertools import combinations
from qiskit_nature.second_q.circuit.library.ansatzes import UCC

def apply_encoding_mapper(operator, suppress_none=True):
    return convert_encoding(operator)

def convert_encoding(operators, suppress_none=True, check_commutes=False, num_particles=None, sector_locator=None):
    if type(operators) == FermionicOp:
        operator = operators
        output = 0
        operator = fix_qubit_order_convention(operator)
        encoded_operator = apply_encoding(operator = operator, encoding = SymmetryAdaptedEncoding_encoding, output_format = 'qiskit')
        if type(encoded_operator) != int and type(encoded_operator) != None:
            output = encoded_operator
    elif type(operators) == list:
        output = list()
        for operator in operators:
            operator = fix_qubit_order_convention(operator)
            encoded_operator = apply_encoding(operator = operator, encoding = SymmetryAdaptedEncoding_encoding, output_format = 'qiskit')
            if type(encoded_operator) != int and type(encoded_operator) != None:
                output.append(encoded_operator)
    return output

def transform(driver, operators):
    output1 = driver
    output2 = list()
    if operators == None:
        return output1, None
    for operator in operators:
        operator = fix_qubit_order_convention(operator)
        encoded_operator = apply_encoding(operator = operator, encoding = SymmetryAdaptedEncoding_encoding, output_format = 'qiskit')
        if type(encoded_operator) != int and type(encoded_operator) != None:
            output2.append(encoded_operator)
    return output1, output2

def fix_qubit_order_convention(input):
    # Remap spin-orbital indices from qiskit-nature's blocked ordering
    # (alpha-block [0, N/2) then beta-block [N/2, N)) to the interleaved
    # ordering the symmetry-adapted encoding expects (alpha spatial k -> 2k,
    # beta spatial k -> 2k+1).  Ported to the qiskit-nature 0.7.x FermionicOp
    # sparse-label API (the legacy ``.to_list()``/dense-label path is gone).
    N = input.register_length
    half = N // 2

    def _remap(i):
        return 2 * i if i < half else 2 * (i - half) + 1

    new_terms = {}
    for label, coeff in input.items():
        if label == "":
            new_label = ""
        else:
            new_tokens = []
            for tok in label.split():
                op, idx = tok.split("_")
                new_tokens.append(f"{op}_{_remap(int(idx))}")
            new_label = " ".join(new_tokens)
        new_terms[new_label] = new_terms.get(new_label, 0) + coeff
    return FermionicOp(new_terms, num_spin_orbitals=N)
    
def SymmetryAdaptedEncodingQubitConverter(encoding):
    global SymmetryAdaptedEncoding_encoding
    SymmetryAdaptedEncoding_encoding = encoding
    qubit_transformation = QubitMapper()
    qubit_transformation.map = apply_encoding_mapper
    return qubit_transformation

def HartreeFockCircuit(encoding, atom, basis, charge = 0, spin = 0, irrep = None, CAS = None, natural_orbitals = False):
    """Circuit to prepare the Hartree-Fock state. The circuit is a number of Pauli X gates.

    Args:
        encoding (_type_):
        atom (_type_): _description_
        basis (_type_): _description_
        charge (int, optional): _description_. Defaults to 0.
        spin (int, optional): _description_. Defaults to 0.
        irrep (_type_, optional): _description_. Defaults to None.
        CAS (_type_, optional): _description_. Defaults to None.
        natural_orbitals (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
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

    if natural_orbitals == True:
        mymp = mp.UMP2(mf).run(verbose = 0)
        noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
        mf.mo_coeff = natorbs

    b = HartreeFock_ket(mf.mo_occ)

    if len(encoding) == 2:
        encoding, CAS_encoding = encoding
    else:
        CAS_encoding = None
    
    tableau, tableau_signs, target_qubits = encoding
    tableau = np.array(tableau)
    tableau_signs = np.array(tableau_signs)
    n = len(tableau)//2
    ZZ_block = (-tableau[:n, :n] + 1)//2
    sign_vector = (-tableau_signs[:n]+ 1)//2
    string_b = f'{b:0{n}b}'
    b_list = list(string_b)[::-1]
    for i in range(len(b_list)):
        b_list[i] = int(b_list[i])
    c_list = np.matmul(ZZ_block, b_list + sign_vector)[::-1] % 2
    string_c = ''.join(str(x) for x in c_list)
    target_qubits.sort(reverse = True)
    for qubit in target_qubits:
        l = len(string_c)
        string_c = string_c[:l - qubit - 1] + string_c[l - qubit:]

    if CAS_encoding != None:
        CAS_tableau, CAS_tableau_signs, CAS_target_qubits = CAS_encoding
        CAS_target_qubits.sort(reverse = True)
        for qubit in CAS_target_qubits:
            l = len(string_c)
            string_c = string_c[:l - qubit - 1] + string_c[l - qubit:]
    
    output = QuantumCircuit(len(string_c))
    for i, bit in enumerate(string_c[::-1]):
        if bit == '1':
            output.x(i)
    return output

def make_fermionic_excitation_ops(reference_state):
    #Single and double excitation generators about a Jordan-Wigner reference
    #determinant, as OpenFermion FermionOperators i(T +/- T^dag).  (OpenFermion's
    #unambiguous '<mode>^ <mode>' labels are used in place of qiskit-nature's
    #dense FermionicOp labels; apply_encoding accepts FermionOperator directly.)
    reference_state = list(reference_state)
    reference_state.reverse()
    occ   = [i for i, x in enumerate(reference_state) if x == '1']
    unocc = [i for i, x in enumerate(reference_state) if x == '0']

    operators = []

    #singles: i (T + T^dag) with T = a^dag_p a_q
    for (p,) in combinations(occ, 1):
        for (q,) in combinations(unocc, 1):
            excitation = (1j * FermionOperator(f'{p}^ {q}')
                          + 1j * FermionOperator(f'{q}^ {p}'))
            if excitation not in operators:
                operators.append(excitation)

    #doubles: i (T - T^dag) with T = a^dag_p a^dag_p2 a_q a_q2
    for (p, p2) in combinations(occ, 2):
        for (q, q2) in combinations(unocc, 2):
            excitation = (1j * FermionOperator(f'{p}^ {p2}^ {q} {q2}')
                          - 1j * FermionOperator(f'{q2}^ {q}^ {p2} {p}'))
            if excitation not in operators:
                operators.append(excitation)

    return operators

def make_excitation_ops(reference_state, encoding):
    operators = []
    excitations = make_fermionic_excitation_ops(reference_state)
    for excitation in excitations:
        op = apply_encoding(encoding = encoding, operator = excitation, output_format = 'qiskit')
        operators.append(op)
    return operators

def get_num_particles_spin_orbitals(atom, basis, charge = 0, spin = 0):
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
    number_up = (mol.nelectron + mol.spin)//2
    number_down = (mol.nelectron - mol.spin)//2
    num_particles = (number_up, number_down)
    num_spin_orbitals = len(mf.mo_coeff)*2
    
    return num_particles, num_spin_orbitals

def UCC_SAE_circuit(atom, basis, charge = 0, spin = 0, irrep = None, CAS = None, natural_orbitals = False, excitations = "sd"):
    encoding = make_encoding(atom = atom, basis = basis, charge = charge, spin = spin, CAS = CAS, irrep = irrep, natural_orbitals = natural_orbitals)
    num_particles, num_spin_orbitals = get_num_particles_spin_orbitals(atom = atom, basis = basis, charge = charge, spin = spin)    
    initial_state = HartreeFockCircuit(encoding = encoding, atom = atom, basis = basis, charge = charge, spin = spin, CAS = CAS, irrep = irrep, natural_orbitals = natural_orbitals)
    qubit_converter = SymmetryAdaptedEncodingQubitConverter(encoding)    
    ansatz = UCC(
        excitations = excitations,
        num_particles = num_particles,
        initial_state = initial_state,
        num_spatial_orbitals = num_spin_orbitals // 2,
        qubit_mapper = qubit_converter,
    )
    return ansatz
