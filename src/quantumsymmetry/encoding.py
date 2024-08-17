from pyscf import gto, scf, mp, mcscf, symm
from .core import * 
from .qiskit_converter import UCC_SAE_circuit
from qiskit.circuit.quantumcircuit import QuantumCircuit
import numpy as np
from numpy import linalg
from tabulate import tabulate
from IPython.display import display, HTML
from openfermion import QubitOperator, FermionOperator, jordan_wigner, utils, linalg
from qiskit import quantum_info
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import QubitMapper, JordanWignerMapper, InterleavedQubitMapper
import time

class Encoding():
    def __init__(self, atom, basis, charge = 0, spin = 0, irrep = None, verbose = False, show_lowest_eigenvalue = False, CAS = None, natural_orbitals = False, active_mo = None, output_format = 'openfermion'):
        self.atom = atom
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.irrep = irrep
        self.verbose = verbose
        self.show_lowest_eigenvalue = show_lowest_eigenvalue
        self.CAS = CAS
        self.natural_orbitals = natural_orbitals
        self.active_mo = active_mo
        output_format = output_format.lower().replace('_', '').replace(' ', '')     
        self.output_format = output_format

        #Runs PySCF
        self.run_pyscf()
        
        self.character_table, self.conj_labels, self.irrep_labels, self.conj_descriptions = get_character_table(self.groupname)
        if self.irrep == None:
            self.irrep =  find_ground_state_irrep(self.label_orb_symm, self.mo_occ, self.character_table, self.irrep_labels)
        self.get_CAS_qubits()
        self.symmetry_generator_labels, self.symmetry_generators_strings, self.target_qubits, self.symmetry_generators, self.signs, self.descriptions = find_symmetry_generators(self.mol, self.irrep, self.label_orb_symm, self.CAS_qubits)
        self.tableau, self.tableau_signs = make_clifford_tableau(self.symmetry_generators, self.signs, self.target_qubits)    
        self.get_CAS_encoding()
        
    def _repr_html_(self):
        if self.verbose == True:
            return self.report(show_lowest_eigenvalue = self.show_lowest_eigenvalue)
        
    def run_pyscf(self):
        mol = gto.Mole()
        mol.atom = self.atom
        mol.symmetry = True
        mol.basis = self.basis
        mol.charge = self.charge
        mol.spin = self.spin
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

        if self.natural_orbitals == True:
            mymp = mp.UMP2(mf).run(verbose = 0)
            noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
            mf.mo_coeff = natorbs

        self.mol = mol
        self.mf = mf
        self.groupname = mol.groupname
        self.nelectron = self.mol.nelectron
        self.nelectron_up = (self.nelectron + self.spin)//2
        self.nelectron_down = (self.nelectron - self.spin)//2
        self.molecule_name = get_molecule_name(self.mol)
        self.label_orb_symm = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
        self.mo_occ = mf.mo_occ
        self.nspinorbital = 2*len(mf.mo_coeff)

    def get_CAS_qubits(self):
        if self.CAS != None:
            if self.active_mo == None:
                self.frozen_core_orbitals = list(range(self.nelectron - self.CAS[0]))
                self.active_space_orbitals = list(range(self.nelectron - self.CAS[0], self.nelectron - self.CAS[0] + 2*self.CAS[1]))
                self.virtual_orbitals = list(range(len(self.frozen_core_orbitals) + len(self.active_space_orbitals), self.nspinorbital))
            else:
                self.frozen_core_orbitals = []
                self.active_space_orbitals = []
                self.virtual_orbitals = []
                for mo in self.active_mo:
                    self.active_space_orbitals.append(2*mo - 2)
                    self.active_space_orbitals.append(2*mo - 1)
                count = 0
                for mo in range(self.nspinorbital):
                    if mo not in self.active_space_orbitals:
                        if count != self.nelectron - self.CAS[0]:
                            self.frozen_core_orbitals.append(mo)
                            count += 1
                        else:
                            self.virtual_orbitals.append(mo)
            self.CAS_qubits = [self.frozen_core_orbitals, self.active_space_orbitals, self.virtual_orbitals]
        else:
            self.CAS_qubits = None
    
    def get_CAS_encoding(self):
        if self.CAS != None:
            self.symmetry_target_qubits = self.target_qubits
            self.target_qubits = self.frozen_core_orbitals + self.target_qubits + self.virtual_orbitals
            
            #Update tableau signs for CAS
            CAS_tableau_signs = [1]*2*self.nspinorbital
            for x in self.frozen_core_orbitals:
                CAS_tableau_signs[x] = -1
            e = np.array(self.tableau)
            e = (1 - e) // 2
            e = np.linalg.inv(e)
            s = np.matmul(e, (1 -np.array(CAS_tableau_signs))//2) % 2
            s += (1-self.tableau_signs)//2
            s = s%2
            s = -2*s + 1
            self.tableau_signs = (s).astype(int).tolist()
        
    def apply(self, operator):
        if type(operator) == dict:
            for key in operator:
                operator.update({key: self.apply(operator[key])})
            return operator
        if type(operator) == list:
            for i in range(len(operator)):
                operator[i] = self.apply(operator[i])
            return operator
        if type(operator) == FermionOperator:
            operator = jordan_wigner(operator)
        if type(operator) == FermionicOp:
            operator = InterleavedQubitMapper(JordanWignerMapper()).map(operator)
        if type(operator) == quantum_info.SparsePauliOp:
            operator = SparsePauliOp_to_QubitOperator(operator)
        if type(operator) == QubitOperator:
            operator = apply_Clifford_tableau(operator, self.tableau, self.tableau_signs)
            operator = simplify_QubitOperator(project_operator(operator, self.target_qubits))
            if self.output_format == 'openfermion':
                return operator
            elif self.output_format == 'qiskit':
                operator = QubitOperator_to_PauliSumOp(operator, num_qubits= self.nspinorbital - len(self.target_qubits))
            return operator
        else:
            raise TypeError("Unsupported input type")
    
    def hamiltonian(self):
        self.jordan_wigner_hamiltonian, self.fermion_hamiltonian = get_hamiltonian(self.mol, self.mf)
        return self.apply(self.jordan_wigner_hamiltonian)

    def report(self, html = True, show_lowest_eigenvalue = False):
        MO_irreps_string, orbital_names = get_MO_irreps_string(self.label_orb_symm, self.irrep_labels, html = True)
        qubit_table_string = qubit_table(self.tableau, self.tableau_signs, self.symmetry_target_qubits, orbital_names, self.CAS_qubits)

        hamiltonian_table1, length1, qubits1 = qubit_operator_table(self.jordan_wigner_hamiltonian)
        hamiltonian_table3, length3, qubits3 = qubit_operator_table(self.hamiltonian)

        if self.show_lowest_eigenvalue == True: 
            ground_state1 = linalg.get_ground_state(linalg.get_sparse_operator(self.jordan_wigner_hamiltonian))
            if qubits3 != 1:
                ground_state3 = linalg.get_ground_state(linalg.get_sparse_operator(self.hamiltonian))
            else:
                ground_state3 = list(np.linalg.eigvalsh(linalg.get_sparse_operator(self.hamiltonian).toarray()))

        eigensectors = list()
        clifford_on_symmetries = str()
        subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        html_groupname = self.groupname[0] + '<sub>' + self.groupname[1:] + '</sub>'
        for i in range(len(self.signs)):
            clifford_on_symmetries += '<p>' + self.symmetry_generators_strings[i].translate(subscripts).replace('+', '') + ' → ' + 'Z' + str(self.symmetry_target_qubits[i]).translate(subscripts) + '</p>'
            eigensectors.append('{0:+d}'.format(self.signs[i]))
        for i in range(len(self.symmetry_target_qubits)):
            self.symmetry_generators_strings[i] = self.symmetry_generators_strings[i][2:].translate(subscripts)
        to_print = ('<h2>Molecule</h2>' +
        '<p>The molecule is ' + self.molecule_name + ' in the ' + self.basis + ' basis')
        if self.CAS != None:
            to_print += ' and CAS' + str(self.CAS)
        to_print += ('.</p>' +
        '<h2>Point group</h2>' +
        '<p>The <b>Boolean point group</b> for ' + self.molecule_name + ' is the group ' + html_groupname + '.</p>' +
        '<p>The character table of ' + html_groupname + ' is:</p>' + 
        tabulate(self.character_table, headers = self.conj_labels, showindex = self.irrep_labels, tablefmt='html') +
        '<p>Each row of the character table corresponds to an irreducible representation of ' + html_groupname + ', and each column to one of its elements.</p>' +
        '<p>(for a non-Abelian group each column would correspond to a conjugacy class of elements)</p>' +
        '<h2>Molecular orbitals</h2>' +
        '<p>The molecular orbitals (MOs) for ' + self.molecule_name + ' in the ' + self.basis + ' basis are:</p>' +
        MO_irreps_string +
        '<p>Each molecular orbital corresponds to two spin-orbitals and to two qubits in the Jordan-Wigner basis (in the order ↓, ↑), for a total of ' + str(len(self.label_orb_symm)) + ' orbitals and ' + str(qubits1) + ' in the Jordan-Wigner encoding.</p>' +
        '<h2>Full Boolean symmetry group</h1>' +
        '<p>The operators for the number of electrons in spin up (N<sub>↑</sub>) and down (N<sub>↓</sub>) also correspond to symmetries of the Hamiltonian,' + 
        'and therefore so do the parity operators P<sub>↑</sub> = (-1)<sup>N<sub>↑</sub></sup> and P<sub>↓</sub> = (-1)<sup>N<sub>↓</sub></sup> (which generate a symmetry group isomorphic to ℤ<sub>2</sub><sup>2</sup>).</p>' +
        '<p>The full Boolean symmetry group is isomorphic to ℤ <sub>2</sub><sup>' + str(len(self.symmetry_target_qubits)) + '</sup> and its ' + str(len(self.symmetry_target_qubits)) + ' generators and their representations in the qubit space are then:</p>' +
        tabulate(zip(self.symmetry_generator_labels, self.descriptions, self.symmetry_generators_strings, eigensectors, self.symmetry_target_qubits), headers = ['Symmetry operator', 'Description', 'Qubit representation', 'Eigensector', 'Qubit to remove'], tablefmt='html') +
        '<p>The target eigensectors are the ones of states that are in the ' + self.irrep[0] + '<sub>' + self.irrep[1:] + '</sub> irreducible representation of ' + html_groupname + ', and for which N<sub>↑</sub> is ' + str((self.nelectron + self.spin)//2 % 2).replace('0', 'even').replace('1', 'odd') + ' and N<sub>↓</sub> is ' + str((self.nelectron - self.spin)//2 % 2).replace('0', 'even').replace('1', 'odd') + '.</p>' +
        '<h2>Change-of-basis transformation and Clifford tableau</h2>' +
        '<p>The change-of-basis transformation acts on the qubit operators corresponding to the generators of the Boolean symmetry group as:</p>' +
        clifford_on_symmetries +
        '<p>Its Clifford tableau is:</p>' +
        show_tableau(self.tableau, self.tableau_signs, html=True) +
        #show_Clifford_kets_HTML(tableau, tableau_signs, mf.mo_occ, target_qubits, full = full) +
        '<p>The states in the target eigensectors all have qubits ' + str(self.symmetry_target_qubits)[1:-1] + ' equal to |0⟩, and these qubits are removed to obtain the <b>symmetry-adapted encoding</b>.</p>')
        if self.CAS != None:
            to_print += '<p>The complete-active-space (CAS) approximation corresponds to also setting the occupancy of the qubits corresponding'
            if len(self.frozen_core_orbitals) != 0:
                to_print += ' to the frozen-core spin-orbitals (qubits ' + str(self.frozen_core_orbitals)[1:-1] + ')'
            if len(self.frozen_core_orbitals) != 0 and len(self.virtual_orbitals) != 0:
                to_print += ' and the ones corresponding'
            if len(self.virtual_orbitals) != 0:
                to_print += ' to the virtual spin-orbitals (qubits ' + str(self.virtual_orbitals)[1:-1] + ')'
            to_print += ' in the Jordan-Wigner encdoing) as equal to |0⟩. These qubits are also removed.</p>'
        to_print += ('<h2>Qubits in the symmetry-adapted encoding</h2>' +
        '<p>In the <b>symmetry-adapted encoding</b> information about the occupancy of the ' + str(qubits1) + ' spin-orbitals is encoded in ' + str(qubits3) + ' qubits as:</p>' +
        qubit_table_string +
        '<p><i>(where addition is taken modulo 2)</i></p>' +
        '<h2>Jordan-Wigner Hamiltonian</h2>' +
        '<p>The <b>second-quantised qubit Hamiltonian</b> in the <b>Jordan-Wigner basis</b> is given by:</p>' +
        hamiltonian_table1 +
        '<p><i>(' + f'{length1:,}' + ' terms on ' + str(qubits1) + ' qubits)</i></p>')
        if show_lowest_eigenvalue == True:
            to_print += '<p>Its lowest eigenvalue is ' + f'{ground_state1[0]:.15f}' + ' Ha.</p>'
        to_print += ('<h2>Qubit-reduced Hamiltonian</h2>' +
        '<p>The <b>qubit-reduced Hamiltonian</b> in the symmetry-adapted encoding (obtained by changing basis by applying the Clifford tableau above, eliminating the redundant qubits and relabelling the remaining qubits) is:</p>' +
        hamiltonian_table3 +
        '<p><i>(' + f'{length3:,}' + ' terms on ' + str(qubits3) + ' qubits)</i></p>')
        if show_lowest_eigenvalue == True:
            to_print += '<p>Its lowest eigenvalue is ' + f'{ground_state3[0]:.15f}' + ' Ha.</p>'
        display(HTML(to_print))

    def qiskit_mapper(self):
        #QubitMapper object
        mapper = QubitMapper()
        mapper.map = self.apply
        return mapper
    
    def HF_circuit(self):
        #Compute Jordan-Wigner Hartree-Fock ket as an int (e.g. 3 for 0011)
        b = HartreeFock_ket(self.mo_occ)

        #Recover ZZ-block of Cliffort tableau and implement v' = C_ZZ v + s mod2
        tableau = np.array(self.tableau)
        tableau_signs = np.array(self.tableau_signs)
        n = self.nspinorbital
        ZZ_block = (-tableau[:n, :n] + 1)//2
        sign_vector = (-tableau_signs[:n]+ 1)//2
        string_b = f'{b:0{n}b}'
        b_list = list(string_b)[::-1]
        for i in range(len(b_list)):
            b_list[i] = int(b_list[i])
        c_list = np.matmul(ZZ_block, b_list + sign_vector)[::-1] % 2
        string_c = ''.join(str(int(x)) for x in c_list)
        
        #Remove target qubits
        target_qubits = self.target_qubits
        target_qubits.sort(reverse = True)
        for qubit in target_qubits:
            l = len(string_c)
            string_c = string_c[:l - qubit - 1] + string_c[l - qubit:]

        #Build Qiskit QuantumCircuit object to prepare the SAE HF state
        output = QuantumCircuit(len(string_c))
        for i, bit in enumerate(string_c[::-1]):
            if bit == '1':
                output.x(i)
        
        return output
    
    qiskit_mapper = property(qiskit_mapper)
    hamiltonian = property(hamiltonian)
    HF_circuit = property(HF_circuit)

#Legacy functions

def reduced_hamiltonian(atom, basis, charge = 0, spin = 0, irrep = None, verbose = True, show_lowest_eigenvalue = False, CAS = None, active_mo = None, natural_orbitals = False, output_format = 'openfermion'):
    encoding = Encoding(atom = atom, basis = basis, charge = charge, spin = spin, irrep = irrep, verbose = verbose, show_lowest_eigenvalue = show_lowest_eigenvalue, CAS = CAS, natural_orbitals = natural_orbitals, active_mo = active_mo, output_format = output_format)
    return encoding.hamiltonian

def make_encoding(atom, basis, charge = 0, spin = 0, irrep = None, CAS = None, natural_orbitals = False, active_mo = None):
    encoding = Encoding(atom = atom, basis = basis, charge = charge, spin = spin, irrep = irrep, CAS = CAS, natural_orbitals = natural_orbitals, active_mo = active_mo)
    return (encoding.tableau, encoding.tableau_signs, encoding.target_qubits)