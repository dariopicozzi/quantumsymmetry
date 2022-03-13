import numpy as np
from itertools import combinations
from random import randint
from pyscf import gto, scf, symm, tools, ao2mo
from openfermion import QubitOperator, FermionOperator, jordan_wigner, utils, linalg
from qiskit import opflow, quantum_info
from IPython.core.display import display, HTML
from tabulate import tabulate

def get_character_table(point_group_name):
    """Gets the character table for a Boolean molecular symmetry point group

    Args:
        point_group_name (str): the name of the symmetry point group (for example 'C2v')

    Returns:
        character_table (2D array): the square 2D array of 1s and  -1s whose elements represent whether a given irreducible representation (corresponding to the rows of the character table)  is  symmetric (+1) or antisymmetric (-1) under the action of a given point-group element (corresponding to the columns of the character table)
        conj_labels (1D array): names of the elements of the given point group (for example ['E', 'C₂(z)', 'σᵥ(xz)', 'σᵥ(yz)'])
        irrep_labels (1D array): names of the irreducible representation of the given point group (for example ['A1', 'A2', 'B1', 'B2'])
        conj_descriptions (1D array): one-line descriptions of point-group elements
    """
    #Point group C1 (e.g. bromochlorofluromethane)
    if point_group_name == 'C1':
        character_table = [[1]]
        conj_labels = ['E']
        irrep_labels = ['A']
        conj_descriptions = ['Identity element']

    #Point group Cs (e.g. formylfluoride)
    if point_group_name == 'Cs':
        character_table = [[1, 1], [1, -1]]
        conj_labels = ['E', 'σₕ']
        irrep_labels = ['A\'', 'A"']
        conj_descriptions = ['The identity element', 'Reflection across the horizontal mirror plane']

    #Point group C2 (e.g. hydrogen peroxide)
    if point_group_name == 'C2':
        character_table = [[1, 1], [1, -1]]
        conj_labels = ['E', 'C₂']
        irrep_labels = ['A', 'B']
        conj_descriptions = ['Identity element', '180-degree rotation along the principal axis']
        
    #Point group Ci (e.g. tartatic acid)
    if point_group_name == 'Ci':
        character_table = [[1, 1], [1, -1]]
        conj_labels = ['E', 'i']
        irrep_labels = ['Ag', 'Au']
        conj_descriptions = ['Identity element', 'Reflection through the origin (inversion)']

    #Point group C2v (e.g. water)
    if point_group_name == 'C2v':
        character_table = [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]
        conj_labels = ['E', 'C₂(z)', 'σᵥ(xz)', 'σᵥ(yz)']
        irrep_labels = ['A1', 'A2', 'B1', 'B2']
        conj_descriptions = ['Identity element', '180-degree rotation along the principal axis', 'Reflection across the xz mirror plane', 'Reflection across the yz mirror plane']

    #Point group C2h (e.g. butane)
    if point_group_name == 'C2h':
        character_table = [[1, 1, 1, 1], [1,-1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
        conj_labels = ['E', 'C₂(z)', 'i', 'σₕ']
        irrep_labels = ['Ag', 'Bg', 'Au', 'Bu']
        conj_descriptions = ['Identity element', '180-degree rotation along the principal axis', 'Reflection through the origin (inversion)', 'Reflection across the horizontal mirror plane']

    #Point group D2 (e.g. biphenyl)
    if point_group_name == 'D2':
        character_table = [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]
        conj_labels = ['E', 'C₂(z)', 'C₂(y)', 'C₂(x)']
        irrep_labels = ['A', 'B1', 'B2', 'B3']
        conj_descriptions = ['Identity element', '180-degree rotation along the z-axis', '180-degree rotation along the y-axis', '180-degree rotation along the x-axis']

    #Point group D2h (e.g. ethylene)
    if point_group_name == 'D2h':
        character_table = [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, -1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, -1], [1, -1, -1, 1, 1, -1, -1, 1], [1, 1, 1, 1, -1, -1, -1, -1], [1, 1, -1, -1, -1, -1, 1, 1], [1, -1, 1, -1, -1, 1, -1, 1], [1, -1, -1, 1, -1, 1, 1, -1]]
        conj_labels = ['E', 'C₂(z)', 'C₂(y)', 'C₂(x)', 'i', 'σ(xy)', 'σ(xz)', 'σ(yz)']
        irrep_labels = ['Ag', 'B1g', 'B2g', 'B3g', 'Au', 'B1u', 'B2u', 'B3u']
        conj_descriptions = ['Identity element', '180-degree rotation along the z-axis', '180-degree rotation along the y-axis', '180-degree rotation along the x-axis', 'Reflection through the origin (inversion)', 'Reflection across the xy mirror plane', 'Reflection across the xz mirror plane', 'Reflection across the yz mirror plane']

    return character_table, conj_labels, irrep_labels, conj_descriptions

def get_generators(point_group_name, irrep_set):
    """Returns an array generators_labels of 3<=n<=1 symmetry point-group elements and an array target_irrep_labels of the same number of irreducible representations such that:
    - each representation in the target_irrep_labels is antisymmetric with respect to exactly one of the point-group elements in generators_labels, and it is symmetric with respect to the others
    - all representations in the target_irrep_labels are antisymmetric with respect to different point-group elements in generators_labels

    Args:
        point_group_name (str): name of the point group as a string (for example 'C2v')
        irrep_set (list): only those point group irreps that correspond to orbitals (in general a subset of all point group irreducible representations for the given point group)

    Returns:
        target_irrep_labels: the irreducible representations that satisfy the condition given above 
        generators_labels: the point-group elements that satisfy the condition given above
    """
    character_table, conj_labels, irrep_labels, conj_descriptions = get_character_table(point_group_name)

    for number_of_generators in range(3, 0, -1):
        for target_irrep_labels in combinations(irrep_set, number_of_generators):
            for generators_labels in combinations(conj_labels[1:], number_of_generators):

                target_irrep_labels = list(target_irrep_labels)
                generators_labels = list(generators_labels)

                target_irrep_indices = list()
                generators_indices = list()

                for target_irrep_label in target_irrep_labels:
                    target_irrep_indices.append(irrep_labels.index(target_irrep_label))
                for generators_label in generators_labels:
                    generators_indices.append(conj_labels.index(generators_label))

                condition = True

                for i in target_irrep_indices:
                    sum = 0
                    for j in generators_indices:
                        sum += character_table[i][j]
                    if sum != len(generators_indices) - 2:
                        condition = False

                target_irrep_labels2 = list()
                generators_labels2 = list()
                
                if condition == True:
                    for i in target_irrep_indices:
                        for j in generators_indices:
                            if character_table[i][j] == -1:
                                target_irrep_labels2.append(irrep_labels[i])
                                generators_labels2.append(conj_labels[j])
                    return target_irrep_labels2, generators_labels2

def find_symmetry_generators(mol, mf, irrep):
    """Returns the symmetry generators computed using the get_generators function, as well as the eigensector of interest for each symmetry generator

    Args:
        mol (PySCF Mole object): PySCF Mole object containing information on the input molecule
        mf (PySCF SCF object): PySCF SCF object containing information on self-consistent field (SCF) methods calculations for the Mole object
        irrep (string): string identifying the irreducible representation of the molecular point group of interest (for instance 'A1' for the point group C2v)

    Returns:
        symmetry_generator_labels (array): the labels (as strings) of the symmetry generators as elements of the symmetry group
        symmetry_generators_strings (array): strings for the symmetry generators as Z Pauli operators acting on the qubit space
        target_qubits (array): the target qubits of the corresponding generators to eliminate as an array of integers
        symmetry_generators (array): the symmetry generators for the full Boolean group encoded as arrays of 1s (identiy) and -1 (Z Pauli operator)
        signs (array): the eigensector of interest for each symmetry generator encoded as an array of 1s (+) and -1s (-)
        descriptions (array): one-line descriptions for each symmetry generator
    """
    point_group_name = mol.groupname
    orbital_labels = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
    number_of_qubits = 2*len(orbital_labels)

    symmetry_generators = list()
    target_irrep_labels = list()
    target_qubits = list()
    symmetry_generator_labels = list()
    target_irrep_indices = list()
    generators_indices = list()
    descriptions = list()

    character_table, conj_labels, irrep_labels, conj_descriptions = get_character_table(point_group_name)

    #the number of generators is an integer n such that 0 <= n <= 3
    number_of_generators = len(character_table).bit_length() - 1

    #creates a set irrep_set containing the irreps of the molecular orbitals
    irrep_set = set(orbital_labels)
    #removes the trivial representation from irrep_set
    irrep_set.remove(irrep_labels[0])

    target_irrep_labels, generators_labels = get_generators(mol.groupname, irrep_set)

    target_irrep_indices = list()
    generators_indices = list()

    for target_irrep_label in target_irrep_labels:
        target_irrep_indices.append(irrep_labels.index(target_irrep_label))
    for generators_label in generators_labels:
        generators_indices.append(conj_labels.index(generators_label))

    #inserts the location of the target qubits
    for target_irrep_label in target_irrep_labels:
        target_qubits.append(2*list(orbital_labels).index(target_irrep_label) + 1)

    for j in generators_indices:
        symmetry_generator_labels.append(conj_labels[j])
        descriptions.append(conj_descriptions[j])
        antisymmetric_labels = list()
        symmetry_generator = np.empty(number_of_qubits, dtype = int)
        for irrep_label in irrep_labels:
            if character_table[irrep_labels.index(irrep_label)][j] == -1:
                antisymmetric_labels.append(irrep_label)
        for j in range(number_of_qubits):
            if orbital_labels[j//2] in antisymmetric_labels:
                symmetry_generator[j] = -1
            else:
                symmetry_generator[j] = 1
        symmetry_generators.append(symmetry_generator)

    #Take care of parity generators
    for orbital_label in orbital_labels:
        if orbital_label == irrep_labels[0]:
            target_qubits.insert(0,2*list(orbital_labels).index(orbital_label))
            target_qubits.insert(1,2*list(orbital_labels).index(orbital_label) + 1)
            break

    n_alpha = np.empty(number_of_qubits, dtype = int)
    n_beta = np.empty(number_of_qubits, dtype = int)
    for j in range(number_of_qubits):
        n_beta[j] = (-1)**j        
    n_alpha = -1*n_beta

    p_generator = [1]*number_of_qubits
    for generator in symmetry_generators:
        p_generator *= generator
    n_beta *= p_generator

    alpha_labels = set()
    for j, label in enumerate(orbital_labels):
        if p_generator[2*j] == -1:
            alpha_labels.add(label)
    beta_labels = set(orbital_labels) - alpha_labels

    beta_labels = str(beta_labels).replace('\'', '')
    alpha_labels = str(alpha_labels).replace('\'', '')

    #signs of Pα and Pβ
    number_up = (mol.nelectron + mol.spin)//2
    number_down = (mol.nelectron - mol.spin)//2
    
    n_alpha_sign = (-1)**number_up
    n_beta_sign = (-1)**number_down
    
    #irrep optional argument
    if irrep == None:
        irrep = irrep_labels[0]
    else:
        irreps_subset = set(target_irrep_labels)
        irreps_subset.add(irrep_labels[0])
        while len(irreps_subset) < 2**(len(symmetry_generator_labels)- 2):
            for pair in combinations(irreps_subset, 2):
                product = list(np.array(character_table[irrep_labels.index(pair[0])]) * np.array(character_table[irrep_labels.index(pair[1])]))
                irreps_subset.add(irrep_labels[character_table.index(product)])
        if irrep not in irreps_subset:
            exception = "The irrep optional argument can only be one of "
            for x in irreps_subset:
                exception += str(x) + ', '
            exception = exception[:-2] + '.'
            raise Exception(exception)
    
    #calculates and stores the signs for each generator (the eigensectors of interest)
    signs = [n_alpha_sign, n_beta_sign]
    for generator in symmetry_generator_labels:
        signs.append(character_table[irrep_labels.index(irrep)][conj_labels.index(generator)])
    for j in range(len(signs) - 2):
        signs[1] *= signs[j+2]

    symmetry_generators.insert(0, n_alpha)
    symmetry_generators.insert(1, n_beta)

    symmetry_generator_labels.insert(0, 'Pα')
    symmetry_generator_labels.insert(1, 'P\'')

    descriptions.insert(0, 'Parity of electrons with spin α')
    descriptions.insert(1, 'Parity of electrons with spin α for orbitals in irrep ' + alpha_labels[1:-1] + ' and with spin β for orbitals in irrep ' + beta_labels[1:-1])

    irreps_string = str()
    rep_numbers = [0]*len(irrep_labels)
    for orbital_label in orbital_labels:
        j = irrep_labels.index(orbital_label)
        rep_numbers[j] += 1
        orbital_label = str(rep_numbers[j]) + orbital_label.lower()
        irreps_string = orbital_label + ' ' + irreps_string

    symmetry_generators_strings = list()
    for i, generator in enumerate(symmetry_generators):
        string = str()
        for j in range(number_of_qubits):
            if generator[j] == -1:
                string = f'Z{j} ' + string
        if signs[i] == -1:
            string = '- ' + string
        else:
            string = '+ ' + string
        symmetry_generators_strings.append(string)

    return symmetry_generator_labels, symmetry_generators_strings, target_qubits, symmetry_generators, signs, descriptions

def make_clifford_tableau(symmetry_generators, signs, target_qubits):
    """
    Returns the Clifford tableau for the change-of-basis transformation that maps each symmetry generator to a Z Pauli operator acting on a single target qubit
    
    Args:
        symmetry_generators (array): the input symmetry generators encoded as arrays of 1s (identiy) and -1 (Z Pauli operator)
        signs (array): the signs of the corresponding generators in symmetry_generators encoded as an array of 1s (+) and -1s (-)
        target_qubits (array): the target qubits of the corresponding generators to eliminate
    """
    #set number of qubits
    N = len(symmetry_generators[0])
    #creates an empty array
    tableau = np.full((2*N, 2*N), 1)

    #set the rows of the Clifford tableau corresponding to the Z operators on the target qubits
    for n in range(N):
        for i, j in enumerate(target_qubits):
            if n == j:
                tableau[n] = np.concatenate((symmetry_generators[i], [1]*N))
            else:
                tableau[n][n] = -1
    
    #set the rows of the Clifford tableau corresponding to the X operators on the target qubits
    for n in range(N):
        tableau[N + n][N + n] = -1
        for i, j in enumerate(target_qubits):
            #check if symmetry_generator and current X operator anticommute: if they do multiplies the current X-row by an X in the target qubit
            if j != n and symmetry_generators[i][n] == -1:
                tableau[N + n][N + j ] *= -1

    #set the values of the signs in the Clifford tableau
    tableau_signs = np.full((2*N), 1)
    for i, j in enumerate(target_qubits):
        tableau_signs[j] = signs[i]
    
    return tableau, tableau_signs

def make_string(array):
    n = len(array)//2
    output = str() 
    for j in range(n):
        if array[j] == -1:
            output = 'Z' + str(j) + ' ' + output
        if array[n + j] == -1:
            output = 'X' + str(j) + ' ' + output
    return output

def show_tableau(tableau, tableau_signs, html = False):
    """Prints a given Clifford tableau

    Args:
        tableau (array): the input Clifford tableau as a (2n, 2n) array
        tableau_signs (array): the 2n array containing the signs for each row of the tableau
    """
    n = len(tableau)//2
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    if html == False:
        string = str()
        #print Z rows
        for i in range(n):
            string += 'Z' + str(i) + ' → '
            if tableau_signs[i] == -1:
                string += '- '
            string += make_string(tableau[i]) + '\n'
        string += '\n'
        #print X rows
        for i in range(n):
            string += 'X' + str(i) + ' → '
            if tableau_signs[i + n] == -1:
                string += '- '
            string += make_string(tableau[i + n]) + '\n'
        print(string)
    if html == True:
        table = list()
        for i in range(n):
            row = list()
            row.append('Z' + str(i).translate(subscripts))
            row.append(' → ')
            if tableau_signs[i] == -1:
                row.append('-' + make_string(tableau[i]).translate(subscripts))
            if tableau_signs[i] == 1:
                row.append(make_string(tableau[i]).translate(subscripts))
            row.append('')
            row.append('X' + str(i).translate(subscripts))
            row.append(' → ')
            if tableau_signs[i + n] == -1:
                row.append('-' + make_string(tableau[i + n]).translate(subscripts))
            if tableau_signs[i + n] == 1:
                row.append(make_string(tableau[i + n]).translate(subscripts))
            table.append(row)
        return tabulate(table, tablefmt='html')
        

def apply_Clifford_tableau(qubit_operator, tableau, tableau_signs):
    """Applies a Clifford tableau to a qubit operator

    Args:
        qubit_operator (openfermion.QubitOperator): the input qubit operator to be transformed by application of the Clifford tableau
        tableau (2D array): the array in the Clifford tableau
        tableau_signs (1D array): the signs in the Clifford tableau

    Returns:
        openfermion.QubitOperator: the transformed qubit operator
    """
    n = len(tableau)//2
    output = QubitOperator()
    for keys in qubit_operator.terms.keys():
        operator = qubit_operator.terms[keys]
        for key in keys:
            if key[1] == 'Z':
                operator *= tableau_signs[key[0]]*QubitOperator(make_string(tableau[key[0]]))
            if key[1] == 'X':
                operator *= tableau_signs[key[0] + n]*QubitOperator(make_string(tableau[key[0] + n]))
            if key[1] == 'Y':
                operator *= -1j*tableau_signs[key[0]]*tableau_signs[key[0] + n]*QubitOperator(make_string(tableau[key[0]]))*QubitOperator(make_string(tableau[key[0] + n]))
        output += operator
    return output

def simplify_QubitOperator(qubit_operator):
    """Eliminates terms with coefficient 0 from an openfermion.QubitOperator object

    Args:
        qubit_operator (openfermion.QubitOperator): operator to simplify

    Returns:
        openfermion.QubitOperator: simplified operator
    """
    output = QubitOperator()
    for keys in qubit_operator.terms.keys():
        if qubit_operator.terms[keys] != 0:
            output += qubit_operator.terms[keys]*QubitOperator(keys)
    return output

def eliminate_qubits(qubit_operator, target_qubits):
    """Eliminates target qubits from a qubit operator, and relabels the qubits

    Args:
        qubit_operator (openfermion.QubitOperator): operator to eliminate qubits from
        target_qubits (list): list of integers that correspond to the target qubits to eliminate

    Returns:
        openfermion.QubitOperator: the operator without the target qubits
    """
    output = QubitOperator()
    #creates a list of qubits that will not be eliminated
    new_qubits = list()
    for qubit in range(utils.count_qubits(qubit_operator)):
        if qubit not in target_qubits:
           new_qubits.append(qubit)
    #eliminates the target qubits from the input qubit operator, and returns 
    for keys in list(qubit_operator.terms.keys()):
        new_key = list()
        for key in keys:
            if key[0] not in target_qubits:
                new_key.append((new_qubits.index(key[0]), key[1]))
        try:
            output.terms[tuple(new_key)] += qubit_operator.terms[keys]
        except:
            output.terms[tuple(new_key)] = qubit_operator.terms[keys]
    return output

def get_hamiltonian(mol, mf):
    """Constructs the second-quantized molecular Hamiltonian

    Args:
        mol (PySCF Mole object): PySCF Mole object containing information on the input molecule
        mf (PySCF SCF object): PySCF SCF object containing information on self-consistent field (SCF) methods calculations for the Mole object

    Returns:
        openfermion.QubitOperator: second-quantized molecular Hamiltonian as a qubit operator in Jordan-Wigner basis 
        openfermion.FermionOperator: second-quantized molecular as a fermionic operator
    """
    hcore_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    one_electron_integrals = np.einsum('pi,pq,qj->ij', mf.mo_coeff, hcore_ao, mf.mo_coeff)

    eri_4fold_ao = mol.intor('int2e_sph')
    two_electron_integrals = ao2mo.incore.full(eri_4fold_ao, mf.mo_coeff)

    fermion_hamiltonian = FermionOperator()

    #gets the number of molecular orbitals
    number_of_MOs = len(mf.mo_coeff)

    for p in range(number_of_MOs):
        for q in range(number_of_MOs):
            fermion_hamiltonian += one_electron_integrals[p][q]*FermionOperator(f'{2*p}^ {2*q}')
            fermion_hamiltonian += one_electron_integrals[p][q]*FermionOperator(f'{2*p + 1}^ {2*q + 1}')

    for p in range(number_of_MOs):
        for q in range(number_of_MOs):
            for r in range(number_of_MOs):
                for s in range(number_of_MOs):
                    fermion_hamiltonian += 0.5*two_electron_integrals[p][s][q][r]*FermionOperator(f'{2*p}^ {2*q}^ {2*r} {2*s}')
                    fermion_hamiltonian += two_electron_integrals[p][s][q][r]*FermionOperator(f'{2*p}^ {2*q + 1}^ {2*r + 1} {2*s}')
                    fermion_hamiltonian += 0.5*two_electron_integrals[p][s][q][r]*FermionOperator(f'{2*p + 1}^ {2*q + 1}^ {2*r + 1} {2*s + 1}')

    fermion_hamiltonian += mf.energy_nuc()
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    return qubit_hamiltonian, fermion_hamiltonian

def print_character_tables():
    point_group_labels = ['C1', 'Cs', 'C2', 'Ci', 'C2v', 'C2h', 'D2', 'D2h']

    for point_group_name in point_group_labels:
        character_table, conj_labels, irrep_labels, conj_descriptions = get_character_table(point_group_name)
        print("Character table for point group " + point_group_name)
        display(HTML(tabulate(character_table, headers = conj_labels, showindex = irrep_labels, tablefmt='html')))

def get_MO_irreps_string(label_orb_symm, irrep_labels, html = False):
    """Gets the molecular orbitals irreps in the format:
    4b1u 2b3g 3b1u 4ag 2b2u 1b2g 1b3u 1b3g 3ag 1b2u 2b1u 2ag 1b1u 1ag

    Args:
        label_orb_symm (array): the orbital symmetry lables in the format: ['Ag' 'B1u' 'Ag' 'B1u' 'B2u' 'Ag' 'B3g' 'B3u' 'B2g' 'B2u' 'Ag' 'B1u' 'B3g' 'B1u']
        irrep_labels (array): the orbital symmetry labels in the format ['Ag', 'B1g', 'B2g', 'B3g', 'Au', 'B1u', 'B2u', 'B3u']
        HTML (bool): HTML output

    Returns:
        str: the molecular orbital irreps in the format: '4b1u 2b3g 3b1u 4ag 2b2u 1b2g 1b3u 1b3g 3ag 1b2u 2b1u 2ag 1b1u 1ag'
    """
    MO_irreps_string = str()
    orbital_names = list()
    rep_numbers = [0]*len(irrep_labels)
    for orbital_label in label_orb_symm:
        j = irrep_labels.index(orbital_label)
        rep_numbers[j] += 1
        if html == True:
            if 'A\'' not in irrep_labels:
                orbital_label = str(rep_numbers[j]) + orbital_label.lower()[:1] + '<sub>' + orbital_label[1:] + '</sub>' + ' '
            else:
                orbital_label = str(rep_numbers[j]) + orbital_label.lower()[:1] + orbital_label[1:] + ' '
        else:
            orbital_label = str(rep_numbers[j]) + orbital_label.lower() + ' '
        orbital_names.append(orbital_label)
        MO_irreps_string = orbital_label + ' ' + MO_irreps_string
    return MO_irreps_string, orbital_names

def qubit_operator_table(qubit_operator):
    """Reformats an openfermion.QubitOperator object as a string

    Args:
        qubit_operator (openfermion.QubitOperator): input operator

    Returns:
        str: the reformatted operator
        int: the number of terms
        int: the number of qubits the operator acts on
    """
    threshold = 25
    table = list()
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    ordered_keys = list(qubit_operator.terms.keys())
    ordered_keys.sort(reverse = True, key = lambda x: abs(qubit_operator.terms[x]))
    for keys in ordered_keys:
        Pauli_string = str()
        for key in keys:
            Pauli_string = str(key[1]) + str(key[0]).translate(subscripts)+ ' ' + Pauli_string
        if Pauli_string == str():
            Pauli_string = 'I'
        table.append(f'{qubit_operator.terms[keys].real:.5f} ' + Pauli_string + ' +')
    table[-1] = table[-1][:-2] + '.'
    terms = len(table)
    qubits = utils.count_qubits(qubit_operator)
    string = str()
    if len(table) < threshold:
        for i in table:
            string += i + ' '
        return 'H = ' + string, terms, qubits
    else:
        for i in table[:threshold - 3]:
            string += i + ' '
        string += ' ... + '
        for i in table[-1:]:
            string += i + ' '
        return 'H = ' + string, terms, qubits

def show_Clifford_kets_HTML(tableau, tableau_signs, full = True):
    """Outputs an HTML formatted string to show how the Clifford tableau acts on the computational basis

    Args:
        tableau (2D array): array in the Clifford tableau
        tableau_signs (1D array): signs in the Clifford tableau
        full (bool, optional): if True shows the whole computational basis is shown, if False only one state. Defaults to True.

    Returns:
        str: HTML formatted string
    """
    n = len(tableau)//2
    sqrt = round(np.sqrt(n))
    ZZ_block = (-tableau[:n, :n] + 1)//2
    sign_vector = (-tableau_signs[:n]+ 1)//2
    if full == False:
        b = randint(0, 2**n)
        string_b = f'{b:0{n}b}'
        b_list = list(string_b)[::-1]
        for i in range(len(b_list)):
            b_list[i] = int(b_list[i])
        c_list = np.matmul(ZZ_block, b_list + sign_vector)[::-1] % 2
        string_c = ''.join(str(x) for x in c_list)
        c = int(string_c, 2)
        return '<p>The change-of-basis transformation acts on the computational basis as a permutation.<p></p>For instance, it maps the computational basis state |' + string_b + '⟩' + ' to the basis state ' + '|' + string_c + '⟩.</p>'
    else:
        table = list()
        row = list()
        unvaried = str()
        count = 0
        for b in range(2**n):
            string_b = f'{b:0{n}b}'
            b_list = list(string_b)[::-1]
            for i in range(len(b_list)):
                b_list[i] = int(b_list[i])
            c_list = np.matmul(ZZ_block, b_list + sign_vector)[::-1] % 2
            string_c = ''.join(str(x) for x in c_list)
            c = int(string_c, 2)
            if b < c:
                if count % sqrt != 0:
                    row += ['|' + string_b + '⟩', ' ↔ ', '|' + string_c + '⟩', ' ']
                else:
                    table.append(row)
                    row= ['|' + string_b + '⟩', ' ↔ ', '|' + string_c + '⟩', ' ']
                count += 1
            if b == c:
                unvaried += '|' + string_b + '⟩, '
            unvaried_sentence = str()
            if unvaried != str():
                unvaried_sentence = '<p>And leaves unvaried the states ' + unvaried[:-2] + '.</p>'
        table.append(row)
        return '<p>The change-of-basis transformation acts on the computational basis as a permutation:</p>' + tabulate(table, tablefmt='html', stralign=None) + unvaried_sentence

def get_molecule_name(mol):
    """Gets the name of the molecule in the format H₂O

    Args:
        mol (PySCF Mole object): PySCF Mole object containing information on the input molecule

    Returns:
        str: the name of the molecule
    """
    atoms = list()
    numbers = list()
    atomlist = list()
    for x in mol._atom:
        atomlist.append(x[0])
    for i in range(len(atomlist)):
        if atomlist[i] == 'H':
            atomlist.insert(0, atomlist.pop(i))
    for i in range(len(atomlist)):
        if atomlist[i] == 'C':
            atomlist.insert(0, atomlist.pop(i))
    atomlist.sort()
    for x in atomlist:
        if x not in atoms:
            atoms.append(x[0])
            numbers.append(1)
        else:
            numbers[atoms.index(x[0])] += 1
    string = str()
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    for i in range(len(atoms)):
        if numbers[i] == 1:
            string += atoms[i]
        else:
            string += atoms[i] + str(numbers[i]).translate(subscripts)
    if mol.charge != 0:
        if mol.charge > 0:
            if mol.charge == 1:
                string += '<sup>+</sup>'
            else:
                string += '<sup>' + str(mol.charge) + '+</sup>'
        else:
            if mol.charge == -1:
                string += '<sup>-</sup>'
            else:
                string += '<sup>' + str(abs(mol.charge)) + '-</sup>'
    #special cases
    if string == 'H₃N':
        string = 'NH₃'
    return string

def QubitOperator_to_PauliSumOp(qubitoperator):
    """Converts a qubit operator stored as in an OpenFermion format (an openfermion.QubitOperator object) into the equivalent operator in Qiskit format (a qiskit.opflow.PauliSumOp object)

    Args:
        qubitoperator (openfermion.QubitOperator): the openfermion.QubitOperator input 

    Returns:
        opflow.PauliSumOp: the qiskit.opflow.PauliSumOp output
    """
    N = utils.count_qubits(qubitoperator)
    output = 0*quantum_info.SparsePauliOp('I'*N)
    for key in qubitoperator.terms:
        if not key:
            string = "I"*N
        else:
            string = str()
            count = 0
            for j in range(N):
                if(j > key[-1][0]):
                    while len(string) != N:
                        string += 'I'
                else:
                    if key[count][0] == j:
                        string += key[count][1]
                        count += 1
                    else:
                        string += 'I'
        output += qubitoperator.terms[key]*quantum_info.SparsePauliOp(string)
    return opflow.PauliSumOp(output).reduce()

def reduced_hamiltonian(atom, basis, charge = 0, spin = 0, irrep = None, verbose = True, show_lowest_eigenvalue = False, output_format = 'openfermion'):
    """Calculates the qubit representation of the second-quantized molecular Hamiltonian in an encoding that reduces its qubit count by using the point-group and parity of number of electron symmetries.

    Args:
        atom (str): molecular geometry (for example the hydrogen molecule in the optimized configuration is 'H 0 0 0; H 0.7414 0 0').
        basis (str): molecular chemistry basis (for example the minimal basis is 'sto-3g').
        charge (int, optional): total charge of the molecule. Defaults to 0.
        spin (int, optional): number of unpaired electrons 2S (the difference between the number of alpha and beta electrons). Defaults to 0.
        irrep (str, optional): irreducible representation of interest. Defaults to the totally symmetric irreducible representation.
        verbose (bool, optional): print level (if True prints a summary of the qubit reduction procedure in HTML format, if False does not print any input). Defaults to True.
        show_lowest_eigenvalue (bool, optional): if True shows lowest eigenvalues of the molecular Hamiltonians (when verbose is set to True). Defaults to False.
        output_format (str, optional): output format of qubit-reduced Hamiltonian, can be set to either 'openfermion' (returns an openfermion.QubitOperator object) or 'qiskit' (returns a qiskit.opflow.PauliSumOp object). Defaults to 'openfermion'.

    Returns:
        openfermion.QubitOperator or qiskit.opflow.PauliSumOp: hamiltonian in the qubit reduced encoding as an openfermion.QubitOperator object (qiskit.opflow.PauliSumOp object if output format is set to 'qiskit')
    """
    mol = gto.Mole()
    mol.atom = atom
    mol.symmetry = True
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.build()
    if mol.groupname == 'Dooh' or mol.groupname == 'Coov' or mol.groupname == 'SO3':
        mol.symmetry = 'D2h'
        mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    label_orb_symm = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
    character_table, conj_labels, irrep_labels, conj_descriptions = get_character_table(mol.groupname)
    molecule_name = get_molecule_name(mol)
    symmetry_generator_labels, symmetry_generators_strings, target_qubits, symmetry_generators, signs, descriptions = find_symmetry_generators(mol, mf, irrep)
    tableau, tableau_signs = make_clifford_tableau(symmetry_generators, signs, target_qubits)
    qubit_hamiltonian, fermion_hamiltonian = get_hamiltonian(mol, mf)
    transformed_hamiltonian = apply_Clifford_tableau(qubit_hamiltonian, tableau, tableau_signs)
    reduced_hamiltonian = simplify_QubitOperator(eliminate_qubits(transformed_hamiltonian, target_qubits))
    hamiltonian_table1, length1, qubits1 = qubit_operator_table(qubit_hamiltonian)
    hamiltonian_table2, length2, qubits2 = qubit_operator_table(transformed_hamiltonian)
    hamiltonian_table3, length3, qubits3 = qubit_operator_table(reduced_hamiltonian)
    MO_irreps_string, orbital_names = get_MO_irreps_string(label_orb_symm, irrep_labels, html=True)
    if show_lowest_eigenvalue == True: 
        ground_state1 = linalg.get_ground_state(linalg.get_sparse_operator(qubit_hamiltonian))
        ground_state2 = linalg.get_ground_state(linalg.get_sparse_operator(transformed_hamiltonian))
        if qubits3 != 1:
            ground_state3 = linalg.get_ground_state(linalg.get_sparse_operator(reduced_hamiltonian))
        else:
            ground_state3 = list(np.linalg.eigvalsh(linalg.get_sparse_operator(reduced_hamiltonian).toarray()))

    if irrep == None:
        irrep = irrep_labels[0]
    if qubits1 > 4:
        full = False
    else:
        full = True

    if verbose == True:
        eigensectors = list()
        clifford_on_symmetries = str()
        subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        html_groupname = mol.groupname[0] + '<sub>' + mol.groupname[1:] + '</sub>'
        for i in range(len(signs)):
            clifford_on_symmetries += '<p>' + symmetry_generators_strings[i].translate(subscripts).replace('+', '') + ' → ' + 'Z' + str(target_qubits[i]).translate(subscripts) + '</p>'
            eigensectors.append('{0:+d}'.format(signs[i]))
        for i in range(len(target_qubits)):
            symmetry_generators_strings[i] = symmetry_generators_strings[i][2:].translate(subscripts)
        to_print = ('<h2>Molecule</h2>' +
        '<p>The molecule is ' + molecule_name + ' in the ' + mol.basis + ' basis.</p>' +
        '<h2>Point group</h2>' +
        '<p>The <b>Boolean point group</b> for ' + molecule_name + ' is the group ' + html_groupname + '.</p>' +
        '<p>The character table of ' + html_groupname + ' is:</p>' + 
        tabulate(character_table, headers = conj_labels, showindex = irrep_labels, tablefmt='html') +
        '<p>Each row of the character table corresponds to an irreducible representation of ' + html_groupname + ', and each column to one of its elements.</p>' +
        '<p>(for a non-Abelian group each column would correspond to a conjugacy class of elements)</p>' +
        '<h2>Molecular orbitals</h2>' +
        '<p>The molecular orbitals (MOs) for ' + molecule_name + ' in the ' + mol.basis + ' basis are:</p>' +
        MO_irreps_string +
        '<p><i>(following a little-endian convention, lowest energy orbitals to the right)</i></p>' +
        '<p>Each molecular orbital corresponds to two spin-orbitals and to two qubits in the Jordan-Wigner basis (in the order β, α), for a total of ' + str(len(label_orb_symm)) + ' orbitals and ' + str(qubits1) + ' qubits before qubit number reduction.</p>' +
        '<h2>Full Boolean symmetry group</h1>' +
        '<p>The operators for the number of electrons in spin α (N<sub>α</sub>) and β (N<sub>β</sub>) also correspond to symmetries of the Hamiltonian,' + 
        'and therefore so do the parity operators P<sub>α</sub> = (-1)<sup>N<sub>α</sub></sup> and P<sub>β</sub> = (-1)<sup>N<sub>β</sub></sup> (which generate a symmetry group isomorphic to ℤ<sub>2</sub><sup>2</sup>).</p>' +
        '<p>The full Boolean symmetry group is isomorphic to ℤ <sub>2</sub><sup>' + str(len(target_qubits)) + '</sup> and its ' + str(len(target_qubits)) + ' generators and their representations in the qubit space are then:</p>' +
        tabulate(zip(symmetry_generator_labels, descriptions, symmetry_generators_strings, eigensectors, target_qubits), headers = ['Symmetry operator', 'Description', 'Qubit representation', 'Eigensector', 'Qubit to remove'], tablefmt='html') +
        '<p>The target eigensectors are the ones of states that are in the ' + irrep[0] + '<sub>' + irrep[1:] + '</sub> irreducible representation of ' + html_groupname + ', and for which N<sub>α</sub> is ' + str((mol.nelectron + mol.spin)//2 % 2).replace('0', 'even').replace('1', 'odd') + ' and N<sub>β</sub> is ' + str((mol.nelectron - mol.spin)//2 % 2).replace('0', 'even').replace('1', 'odd') + '.</p>' +
        '<h2>Change-of-basis transformation and Clifford tableau</h2>' +
        '<p>The change-of-basis transformation acts on the qubit operators corresponding to the generators of the Boolean symmetry group as:</p>' +
        clifford_on_symmetries +
        '<p>Its Clifford tableau is:</p>' +
        show_tableau(tableau, tableau_signs, html=True) +
        show_Clifford_kets_HTML(tableau, tableau_signs, full = full) +
        '<p>The states in the target eignesectors all have qubits ' + str(target_qubits)[1:-1] + ' equal to |0⟩.</p>' +
        '<h2>Jordan-Wigner Hamiltonian</h2>' +
        '<p>The <b>second-quantised qubit Hamiltonian</b> in the <b>Jordan-Wigner basis</b> is given by:</p>' +
        hamiltonian_table1 +
        '<p><i>(' + f'{length1:,}' + ' terms on ' + str(qubits1) + ' qubits)</i></p>')
        if show_lowest_eigenvalue == True:
            to_print += '<p>Its lowest eigenvalue is ' + f'{ground_state1[0]:.5f}' + ' Ha.</p>'
        to_print += ('<h2>Transformed Hamiltonian</h2>' +
        '<p>The <b>transformed qubit Hamiltonian</b> obtained by applying the change-of-basis transformation given by the Clifford tableau to the Jordan-Wigner Hamiltonian is:</p>' +
        hamiltonian_table2 +
        '<p><i>(also ' + f'{length2:,}' + ' terms on ' + str(qubits2) + ' qubits)</i></p>')
        if show_lowest_eigenvalue == True:
            to_print += '<p>Its lowest eigenvalue is ' + f'{ground_state2[0]:.5f}' + ' Ha.</p>'
        to_print += ('<h2>Reduced Hamiltonian</h2>' +
        '<p>The <b>reduced qubit Hamiltonian</b> (after eliminating qubits ' + str(target_qubits)[1:-1] + ' and relabelling the remaining qubits) is:</p>' +
        hamiltonian_table3 +
        '<p><i>(' + f'{length3:,}' + ' terms on ' + str(qubits3) + ' qubits)</i></p>')
        if show_lowest_eigenvalue == True:
            to_print += '<p>Its lowest eigenvalue is ' + f'{ground_state3[0]:.5f}' + ' Ha.</p>'

        display(HTML(to_print))
    
    output_format = output_format.lower().replace('_', '').replace(' ', '')
    if output_format == 'openfermion':
        return reduced_hamiltonian
    elif output_format == 'qiskit':
        reduced_hamiltonian = QubitOperator_to_PauliSumOp(reduced_hamiltonian)
        print(reduced_hamiltonian)
    else:
        raise ValueError('The output format must be either OpenFermion or Qiskit')