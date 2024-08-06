from .core import *
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit.circuit.quantumcircuit import QuantumCircuit
from pyscf import gto, scf
from itertools import combinations
from qiskit_nature.second_q.circuit.library.ansatzes import UCC

def apply_encoding_mapper(operator, suppress_none=True):
    apply_encoding(operator = operator, encoding = SymmetryAdaptedEncoding_encoding, output_format = 'qiskit')

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
    output = 0
    input.display_format="dense"
    N = input.register_length
    input_list = input.to_list()
    for x in range(len(input_list)):
        output_label = str()
        input_label = input_list[x][0]
        input_label = input_label[::-1]
        for j in range(N//2):
            output_label += input_label[j]
            output_label += input_label[N//2 + j]
        output += FermionicOp([(output_label[::-1], input_list[x][1])], display_format='dense')
    return output
    
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

def swap_plus_and_minuses(input):
    output = str()
    for s in input:
        if s == '+':
            output += '-'
        elif s == '-':
            output += '+'
        else:
            output += s
    return output

def make_fermionic_excitation_ops(reference_state):
    number_of_qubits = len(reference_state)
    occ = []
    unocc = []
    reference_state = list(reference_state)
    reference_state.reverse()

    #get occupations
    for i, x in enumerate(reference_state):
        if x == '0':
            unocc.append(i)
        if x == '1':
            occ.append(i)

    #singles
    operators_s = []
    for perm_plus in list(combinations(occ, 1)):
        for perm_minus in list(combinations(unocc, 1)):
            if len(set(perm_plus).union(set(perm_minus))) == 2:
                operator = ['I']*number_of_qubits
                for i in perm_plus:
                    operator[i] = '+'
                for i in perm_minus:
                    operator[i] = '-'
                operator = ''.join(operator)
                operators_s.append(operator)

    #doubles
    operators_d = []
    for perm_plus in list(combinations(occ, 2)):
        for perm_minus in list(combinations(unocc, 2)):
            operator = ['I']*number_of_qubits
            for i in perm_plus:
                operator[i] = '+'
            for i in perm_minus:
                operator[i] = '-'
            operator = ''.join(operator)
            operators_d.append(operator)
    
    #create fermionic operators
    operators2 = []
    for operator in operators_s:
        excitation = FermionicOp([(operator, 1j), (swap_plus_and_minuses(operator), 1j)], register_length = number_of_qubits, display_format='dense')
        if excitation not in operators2:
            operators2.append(excitation)

    for operator in operators_d:
        excitation = FermionicOp([(operator, 1j), (swap_plus_and_minuses(operator), -1j)], register_length = number_of_qubits, display_format='dense')
        if excitation not in operators2:
            operators2.append(excitation)

    return operators2

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
        num_spin_orbitals = num_spin_orbitals,
        qubit_converter = qubit_converter,
    )
    return ansatz
