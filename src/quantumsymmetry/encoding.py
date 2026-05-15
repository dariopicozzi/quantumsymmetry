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
    def __init__(self, atom, basis, charge = 0, spin = 0, irrep = None, symmetry = True, verbose = False, show_lowest_eigenvalue = False, CAS = None, quick_CAS=False, natural_orbitals = False, active_mo = None, bravyi_kitaev = False, output_format = 'openfermion'):
        self.atom = atom
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.irrep = irrep
        self.symmetry = symmetry
        self.verbose = verbose
        self.show_lowest_eigenvalue = show_lowest_eigenvalue
        self.CAS = CAS
        self.quick_CAS = quick_CAS
        self.natural_orbitals = natural_orbitals
        self.active_mo = active_mo
        self.bravyi_kitaev = bool(bravyi_kitaev)
        self._bk_tableau_cache = {}
        output_format = output_format.lower().replace('_', '').replace(' ', '')     
        self.output_format = output_format

        #Runs PySCF
        self.run_pyscf()
        
        self.character_table, self.conj_labels, self.irrep_labels, self.conj_descriptions = get_character_table(self.groupname)
        if self.irrep is None:
            if self.symmetry:
                self.irrep = find_ground_state_irrep(
                    self.label_orb_symm,
                    self.mo_occ,
                    self.character_table,
                    self.irrep_labels
                )
            else:
                self.irrep = self.irrep_labels[0]
        self.get_CAS_qubits()


        if self.symmetry:
            (self.symmetry_generator_labels,
            self.symmetry_generators_strings,
            self.target_qubits,
            self.symmetry_generators,
            self.signs,
            self.descriptions) = find_symmetry_generators(
                self.mol,
                self.irrep,
                self.label_orb_symm,
                self.CAS_qubits
            )
            self.tableau, self.tableau_signs = make_clifford_tableau(
                self.symmetry_generators,
                self.signs,
                self.target_qubits
            )

        else:
            self.symmetry_generator_labels = []
            self.symmetry_generators_strings = []
            self.target_qubits = []
            self.symmetry_generators = []
            self.signs = []
            self.descriptions = []
            n = 2 * len(self.mf.mo_coeff)
            self.nspinorbital = n
            self.tableau = [
                [ -1 if i == j else 1
                for j in range(2*n) ]
                for i in range(2*n)
            ]
            self.tableau_signs = [1] * (2*n)

        self.get_CAS_encoding()     

    def _get_bk_tableau(self, n_qubits):
        n_qubits = int(n_qubits)
        if n_qubits < 0:
            raise ValueError("n_qubits must be non-negative")
        if n_qubits not in self._bk_tableau_cache:
            T = make_BK_T(n_qubits).astype(int) % 2
            T_inv = gf2_inv(T).astype(int) % 2
            M_ZZ = T_inv % 2
            M_XX = T.T % 2
            M = np.block(
                [
                    [M_ZZ, np.zeros((n_qubits, n_qubits), dtype=int)],
                    [np.zeros((n_qubits, n_qubits), dtype=int), M_XX],
                ]
            )
            tableau = [list(1 - 2 * x) for x in M]
            tableau_signs = [1] * (2 * n_qubits)
            self._bk_tableau_cache[n_qubits] = (tableau, tableau_signs)
        return self._bk_tableau_cache[n_qubits]

    def _repr_html_(self):
        if self.verbose == True:
            return self.report(show_lowest_eigenvalue = self.show_lowest_eigenvalue)
        
    def run_pyscf(self):
        mol = gto.Mole()
        mol.atom = self.atom
        mol.symmetry = self.symmetry
        mol.basis = self.basis
        mol.charge = self.charge
        mol.spin = self.spin
        mol.verbose = 0
        mol.build()
        if (self.symmetry == True) and (mol.groupname == 'Dooh' or mol.groupname == 'SO3'):
            mol.symmetry = 'D2h'
            mol.build()
        elif (self.symmetry == True) and (mol.groupname == 'Coov'):
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
        if self.symmetry:
            self.label_orb_symm = symm.label_orb_symm(
                mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff
            )
        else:
            self.label_orb_symm = [None] * mf.mo_coeff.shape[1]
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
        self.symmetry_target_qubits = self.target_qubits
        if self.CAS != None:
            self.target_qubits = self.frozen_core_orbitals + self.target_qubits + self.virtual_orbitals
            
            #Update tableau signs for CAS
            CAS_tableau_signs = [1]*2*self.nspinorbital
            for x in self.frozen_core_orbitals:
                CAS_tableau_signs[x] = -1
            if self.symmetry:
                s = (1 -np.array(CAS_tableau_signs))//2
                e = np.array(self.tableau)
                e = (1 - e) // 2
                e = np.linalg.inv(e)
                s = np.matmul(e, (1 -np.array(CAS_tableau_signs))//2) % 2
                s += (1-self.tableau_signs)//2
                s = s%2
                s = -2*s + 1
                self.tableau_signs = (s).astype(int).tolist()
            else:
                self.tableau_signs = CAS_tableau_signs

    def apply(self, operator):
        if type(operator) == dict:
            for key in operator:
                operator.update({key: self.apply(operator[key])})
            return operator
        if type(operator) == list:
            output = []
            for item in operator:
                mapped = self.apply(item)
                # Drop mapped zero operators to keep ansatz compact.
                # This is especially important when a CAS/projected encoding makes
                # many UCC excitations identically zero.
                if type(mapped) == quantum_info.SparsePauliOp:
                    try:
                        if mapped.size == 0:
                            continue
                        # Only drop *exactly* zero operators.
                        # Using allclose() can incorrectly remove small-but-nonzero terms
                        # and change the ansatz/energy.
                        if mapped.coeffs.size == 0:
                            continue
                        if np.all(mapped.coeffs == 0):
                            continue
                    except Exception:
                        pass
                output.append(mapped)
            return output
        if type(operator) == FermionOperator:
            operator = jordan_wigner(operator)
        if type(operator) == FermionicOp:
            operator = InterleavedQubitMapper(JordanWignerMapper()).map(operator)
        if type(operator) == quantum_info.SparsePauliOp:
            operator = SparsePauliOp_to_QubitOperator(operator)
        if type(operator) == QubitOperator:
            operator = apply_Clifford_tableau(operator, self.tableau, self.tableau_signs)
            operator = simplify_QubitOperator(project_operator(operator, self.target_qubits))
            if self.bravyi_kitaev:
                n_qubits = self.nspinorbital - len(self.target_qubits)
                bk_tableau, bk_tableau_signs = self._get_bk_tableau(n_qubits)
                operator = apply_Clifford_tableau(operator, bk_tableau, bk_tableau_signs)
                operator = simplify_QubitOperator(operator)
            if self.output_format == 'openfermion':
                return operator
            elif self.output_format == 'qiskit':
                num_qubits = self.nspinorbital - len(self.target_qubits)
                operator = QubitOperator_to_PauliSumOp(operator, num_qubits=num_qubits)
            return operator
        else:
            raise TypeError("Unsupported input type")
    
    def hamiltonian(self):
        def _remap_qubit_operator_indices(qubit_operator, index_map):
            remapped = QubitOperator()
            for term, coeff in qubit_operator.terms.items():
                if not term:
                    remapped += coeff * QubitOperator(())
                    continue
                new_term = tuple((index_map[q], p) for (q, p) in term)
                remapped += coeff * QubitOperator(new_term)
            return remapped

        if self.quick_CAS and self.CAS is not None:
            cas_qubit_hamiltonian, self.fermion_hamiltonian = (
                get_CAS_hamiltonian(self.mol, self.mf, self.CAS, active_mo=self.active_mo)
            )
            # Embed the CAS Hamiltonian into the full spin-orbital indexing so that
            # it can be consistently transformed/projected by the full-space tableau.
            index_map = {i: self.active_space_orbitals[i] for i in range(len(self.active_space_orbitals))}
            self.jordan_wigner_hamiltonian = _remap_qubit_operator_indices(cas_qubit_hamiltonian, index_map)
        else:
            self.jordan_wigner_hamiltonian, self.fermion_hamiltonian = (
                get_hamiltonian(self.mol, self.mf)
            )

        return self.apply(self.jordan_wigner_hamiltonian)

    def report(self, html = True, show_lowest_eigenvalue = False):
        MO_irreps_string, orbital_names = get_MO_irreps_string(self.label_orb_symm, self.irrep_labels, html = True)
        qubit_table_string = qubit_table(self.tableau, self.tableau_signs, self.symmetry_target_qubits, orbital_names, self.CAS_qubits)

        hamiltonian_table3, length3, qubits3 = qubit_operator_table(self.hamiltonian)
        hamiltonian_table1, length1, qubits1 = qubit_operator_table(self.jordan_wigner_hamiltonian)

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

        # Optional final JW->BK affine map (with b=0) on the remaining qubits.
        # The BK Clifford maps |f> → |Tf>, so the HF bitstring transforms as
        # b = T @ f.
        if self.bravyi_kitaev:
            n = len(string_c)
            T = make_BK_T(n).astype(int) % 2
            bits = np.array([int(x) for x in string_c[::-1]], dtype=int)  # qubit 0 is LSB
            bk_bits = (T @ bits) % 2
            string_c = ''.join(str(int(x)) for x in bk_bits[::-1])

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
    encoding = Encoding(atom = atom, basis = basis, charge = charge, spin = spin, irrep = irrep, show_lowest_eigenvalue = show_lowest_eigenvalue, CAS = CAS, natural_orbitals = natural_orbitals, active_mo = active_mo, output_format = output_format)
    if verbose == True:
        encoding.report(show_lowest_eigenvalue = show_lowest_eigenvalue)
    return encoding.hamiltonian

def make_encoding(atom, basis, charge = 0, spin = 0, irrep = None, CAS = None, natural_orbitals = False, active_mo = None):
    encoding = Encoding(atom = atom, basis = basis, charge = charge, spin = spin, irrep = irrep, CAS = CAS, natural_orbitals = natural_orbitals, active_mo = active_mo)
    return (encoding.tableau, encoding.tableau_signs, encoding.target_qubits)


class PeriodicEncoding():
    """Symmetry-adapted encoding for periodic systems (crystals).

    Two construction paths are supported:

    1. **From PySCF** (atom + lattice vectors + k-mesh).  The supercell
       Hamiltonian is built via the Strategy A pipeline in
       :mod:`quantumsymmetry.periodic`, the supercell point group is
       detected, and the maximal Boolean Z2^k subgroup is extracted
       automatically.

    2. **BYO Hamiltonian.**  The caller supplies a pre-built
       :class:`openfermion.FermionOperator` together with +/-1 generator
       arrays.  Used by the model-Hamiltonian benchmarks (Hubbard,
       two-band).
    """

    def __init__(self,
                 # PySCF-driven path
                 atom=None, a=None, basis=None, kpts=None, pseudo=None,
                 charge=0, spin=0, df='auto', exxdiv='ewald',
                 active_bands=None, symmetry=True, verbose=False,
                 symm_energy_tol=5e-3, symm_purity_tol=0.95,
                 active_mos=None, integral_backend='kpts',
                 # BYO-Hamiltonian path
                 fermion_hamiltonian=None, nspinorbital=None,
                 nelectron_up=None, nelectron_down=None,
                 symmetry_generators=None, signs=None,
                 symmetry_generator_labels=None,
                 # Common
                 name='crystal', bravyi_kitaev=False,
                 output_format='openfermion'):
        self.name = name
        self.bravyi_kitaev = bool(bravyi_kitaev)
        self._bk_tableau_cache = {}
        self.output_format = output_format.lower().replace('_', '').replace(' ', '')
        self.molecule_name = name

        if fermion_hamiltonian is None and atom is not None:
            # ---- PySCF-driven path ----
            if a is None:
                raise ValueError(
                    "PeriodicEncoding from atom requires lattice vectors `a` (3x3)"
                )
            if kpts is None:
                raise ValueError(
                    "PeriodicEncoding from atom requires `kpts=(n1, n2, n3)`"
                )
            from .periodic import build_periodic_inputs
            pyscf_inputs = build_periodic_inputs(
                atom=atom, a=a, basis=basis, kpts_mesh=kpts, pseudo=pseudo,
                spin=spin, charge=charge, df=df, exxdiv=exxdiv,
                active_bands=active_bands,
                verbose=int(verbose) if verbose else 0,
                name=name,
                symmetry=bool(symmetry),
                symm_energy_tol=symm_energy_tol,
                symm_purity_tol=symm_purity_tol,
                active_mos=active_mos,
                integral_backend=integral_backend,
            )
            self._pyscf_periodic = pyscf_inputs
            kw = {k: v for k, v in pyscf_inputs.items() if not k.startswith('_')}
            self._setup(
                fermion_hamiltonian=kw['fermion_hamiltonian'],
                nspinorbital=kw['nspinorbital'],
                nelectron_up=kw['nelectron_up'],
                nelectron_down=kw['nelectron_down'],
                symmetry_generators=kw['symmetry_generators'],
                signs=kw['signs'],
                symmetry_generator_labels=kw['symmetry_generator_labels'],
            )
        else:
            # ---- BYO-Hamiltonian path ----
            self._setup(
                fermion_hamiltonian=fermion_hamiltonian,
                nspinorbital=nspinorbital,
                nelectron_up=nelectron_up,
                nelectron_down=nelectron_down,
                symmetry_generators=symmetry_generators,
                signs=signs,
                symmetry_generator_labels=symmetry_generator_labels,
            )

    def _setup(self, fermion_hamiltonian, nspinorbital, nelectron_up, nelectron_down,
               symmetry_generators, signs, symmetry_generator_labels):
        if fermion_hamiltonian is None:
            raise ValueError("PeriodicEncoding requires a fermion_hamiltonian (openfermion.FermionOperator)")
        if nspinorbital is None:
            raise ValueError("PeriodicEncoding requires nspinorbital (int)")

        self.fermion_hamiltonian = fermion_hamiltonian
        self.jordan_wigner_hamiltonian = jordan_wigner(fermion_hamiltonian)

        self.nspinorbital = int(nspinorbital)
        self.nelectron_up = int(nelectron_up) if nelectron_up is not None else None
        self.nelectron_down = int(nelectron_down) if nelectron_down is not None else None
        if self.nelectron_up is not None and self.nelectron_down is not None:
            self.nelectron = self.nelectron_up + self.nelectron_down
            self.spin = self.nelectron_up - self.nelectron_down
        else:
            self.nelectron = None
            self.spin = 0

        if symmetry_generators is None or len(symmetry_generators) == 0:
            # No symmetry: identity tableau
            self.symmetry = False
            self.symmetry_generator_labels = []
            self.symmetry_generators_strings = []
            self.target_qubits = []
            self.symmetry_target_qubits = []
            self.symmetry_generators = []
            self.signs = []
            self.descriptions = []
            n = self.nspinorbital
            self.tableau = [[-1 if i == j else 1 for j in range(2*n)] for i in range(2*n)]
            self.tableau_signs = [1] * (2*n)
            return

        # Validate generators: each is an array of length nspinorbital with entries +/-1
        gens = []
        for g in symmetry_generators:
            arr = np.asarray(g, dtype=int)
            if arr.shape != (self.nspinorbital,):
                raise ValueError(
                    f"Each symmetry generator must have length {self.nspinorbital}; got {arr.shape}"
                )
            if not set(np.unique(arr).tolist()).issubset({-1, 1}):
                raise ValueError("symmetry_generators entries must be +1 or -1 only")
            gens.append(arr)

        if signs is None:
            signs_list = [1] * len(gens)
        else:
            signs_list = [int(s) for s in signs]
            if not all(s in (-1, 1) for s in signs_list):
                raise ValueError("signs entries must be +1 or -1 only")
        if len(signs_list) != len(gens):
            raise ValueError("signs must have the same length as symmetry_generators")

        # Row-reduce over GF(2) to canonical pivots and target qubits
        sg, sgn, tq = reduced_row_echelon_generators(gens, signs_list)
        self.symmetry = True
        self.symmetry_generators = sg
        self.signs = sgn
        self.target_qubits = list(tq)
        self.symmetry_target_qubits = list(tq)

        if symmetry_generator_labels is None:
            self.symmetry_generator_labels = [f'g{i}' for i in range(len(sg))]
        else:
            labels = [str(x) for x in symmetry_generator_labels]
            if len(labels) >= len(sg):
                self.symmetry_generator_labels = labels[:len(sg)]
            else:
                self.symmetry_generator_labels = labels + [
                    f'g{i}' for i in range(len(labels), len(sg))
                ]
        self.descriptions = [''] * len(sg)

        # Human-readable Z-string representations
        self.symmetry_generators_strings = []
        for i, gen in enumerate(sg):
            tail = ''
            for j in range(self.nspinorbital):
                if gen[j] == -1:
                    tail = f'Z{j} ' + tail
            prefix = '- ' if sgn[i] == -1 else '+ '
            self.symmetry_generators_strings.append(prefix + tail)

        self.tableau, self.tableau_signs = make_clifford_tableau(sg, sgn, self.target_qubits)

    def _get_bk_tableau(self, n_qubits):
        n_qubits = int(n_qubits)
        if n_qubits < 0:
            raise ValueError("n_qubits must be non-negative")
        if n_qubits not in self._bk_tableau_cache:
            T = make_BK_T(n_qubits).astype(int) % 2
            T_inv = gf2_inv(T).astype(int) % 2
            M_ZZ = T_inv % 2
            M_XX = T.T % 2
            M = np.block(
                [
                    [M_ZZ, np.zeros((n_qubits, n_qubits), dtype=int)],
                    [np.zeros((n_qubits, n_qubits), dtype=int), M_XX],
                ]
            )
            tableau = [list(1 - 2 * x) for x in M]
            tableau_signs = [1] * (2 * n_qubits)
            self._bk_tableau_cache[n_qubits] = (tableau, tableau_signs)
        return self._bk_tableau_cache[n_qubits]

    def apply(self, operator):
        if type(operator) == dict:
            for key in operator:
                operator.update({key: self.apply(operator[key])})
            return operator
        if type(operator) == list:
            output = []
            for item in operator:
                mapped = self.apply(item)
                if type(mapped) == quantum_info.SparsePauliOp:
                    try:
                        if mapped.size == 0:
                            continue
                        if mapped.coeffs.size == 0:
                            continue
                        if np.all(mapped.coeffs == 0):
                            continue
                    except Exception:
                        pass
                output.append(mapped)
            return output
        if type(operator) == FermionOperator:
            operator = jordan_wigner(operator)
        if type(operator) == FermionicOp:
            operator = InterleavedQubitMapper(JordanWignerMapper()).map(operator)
        if type(operator) == quantum_info.SparsePauliOp:
            operator = SparsePauliOp_to_QubitOperator(operator)
        if type(operator) == QubitOperator:
            operator = apply_Clifford_tableau(operator, self.tableau, self.tableau_signs)
            operator = simplify_QubitOperator(project_operator(operator, self.target_qubits))
            if self.bravyi_kitaev:
                n_qubits = self.nspinorbital - len(self.target_qubits)
                bk_tableau, bk_tableau_signs = self._get_bk_tableau(n_qubits)
                operator = apply_Clifford_tableau(operator, bk_tableau, bk_tableau_signs)
                operator = simplify_QubitOperator(operator)
            if self.output_format == 'openfermion':
                return operator
            elif self.output_format == 'qiskit':
                num_qubits = self.nspinorbital - len(self.target_qubits)
                operator = QubitOperator_to_PauliSumOp(operator, num_qubits=num_qubits)
            return operator
        raise TypeError("Unsupported input type")

    def hamiltonian(self):
        return self.apply(self.jordan_wigner_hamiltonian)

    def qiskit_mapper(self):
        mapper = QubitMapper()
        mapper.map = self.apply
        return mapper

    qiskit_mapper = property(qiskit_mapper)
    hamiltonian = property(hamiltonian)

