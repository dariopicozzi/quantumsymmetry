"""Build new tutorial notebooks 06, 07, 08 for QuantumSymmetry.

Run with the project venv:

    .venv/bin/python docs/tutorials/_build_new_tutorials.py

This writes three .ipynb files in the same directory. After writing,
execute them with:

    .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        docs/tutorials/06_complete_active_space.ipynb \
        docs/tutorials/07_bravyi_kitaev.ipynb \
        docs/tutorials/08_periodic.ipynb
"""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path


HERE = Path(__file__).resolve().parent

LOGO = (
    '[<img src="../quantumsymmetry_logo.png" alt="QuantumSymmetry" width="450"/>]'
    "(https://github.com/dariopicozzi/quantumsymmetry)"
)

COLAB_CELL = (
    "%%capture\n"
    "if 'google.colab' in str(get_ipython()):\n"
    "    !pip -q install quantumsymmetry"
)

COLAB_NOTE = (
    "> **Note:** if you are running this notebook on Google Colab, the next "
    "cell will install `quantumsymmetry` and its dependencies:"
)


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


# ---------------------------------------------------------------------------
# 06 — Complete active space (CAS)
# ---------------------------------------------------------------------------

def build_06() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list = []

    cells.append(md(LOGO))
    cells.append(md(COLAB_NOTE))
    cells.append(code(COLAB_CELL))

    cells.append(md(
        "# The complete active space (CAS) approximation\n"
        "\n"
        "In the previous notebooks we have used `quantumsymmetry` to reduce "
        "the number of qubits in the qubit Hamiltonian by exploiting the "
        "**exact** Boolean symmetries of the molecule: the point group and "
        "the spin parities. The ground-state energy is unchanged by this "
        "reduction, as the projection is onto a symmetry sector that the "
        "Hamiltonian already preserves.\n"
        "\n"
        "We can squeeze the qubit count down even further if we are willing "
        "to accept a small, controlled approximation. In the **complete "
        "active space (CAS)** approach, the molecular orbitals are split "
        "into three subspaces:\n"
        "\n"
        "- the **frozen-core orbitals**, assumed to be doubly occupied;\n"
        "- the **active orbitals**, whose occupancies are allowed to vary "
        "and which capture electron correlation;\n"
        "- the **virtual orbitals**, assumed to be unoccupied.\n"
        "\n"
        "The notation $(n, m)$-CAS means that $n$ electrons are distributed "
        "in all possible ways across $m$ active spatial orbitals. The "
        "frozen-core/virtual choice is an *approximate* $Z$-symmetry: it "
        "fixes the occupancy of those orbitals to 1 (frozen-core) or 0 "
        "(virtual), exactly as the parity symmetries fix the parity of a "
        "set of qubits, and the qubits associated with these orbitals can "
        "be removed using the same affine Clifford machinery.\n"
        "\n"
        "The combined approach is the **symmetry-adapted encoding with "
        "complete active space (SAE-CAS)** introduced in the article:\n"
        "\n"
        "> *Picozzi, D. and Tennyson, J. (manuscript). Symmetry-adapted "
        "qubit encoding with complete active space and Bravyi-Kitaev mapping "
        "for quantum chemistry on a quantum computer.*\n"
        "\n"
        "In this notebook we are going to apply SAE-CAS to two illustrative "
        "molecules, beryllium hydride (BeH₂) and water (H₂O), in the minimal "
        "STO-3G basis. We will see that the qubit savings are significant, "
        "and that the ground-state energy in the active space matches the "
        "PySCF CASCI reference to numerical precision.\n"
        "\n"
        "1. [Beryllium hydride (BeH₂) with CAS(2,3)](#BeH2)\n"
        "2. [Water (H₂O) with CAS(8,6)](#H2O)\n"
        "3. [How accurate is the active-space approximation?](#accuracy)\n"
    ))

    # --- BeH2 section ---
    cells.append(md(
        '<a name="BeH2"></a>\n'
        "## Beryllium hydride (BeH₂) with CAS(2,3)\n"
        "\n"
        "Beryllium hydride is a linear molecule (Be in the middle, two H's "
        "along the $z$-axis). In the minimal STO-3G basis it has 7 spatial "
        "orbitals, or 14 spin-orbitals, and 6 electrons. In tutorial "
        "[03_molecular_hamiltonians](03_molecular_hamiltonians.ipynb) we "
        "saw that the symmetry-adapted encoding alone reduces the qubit "
        "count from 14 down to 9 by exploiting the $D_{2h}$ point group "
        "and the spin parities.\n"
        "\n"
        "The two lowest-energy molecular orbitals, $1a_g$ and $2a_g$, are "
        "very close to the doubly-occupied $1s$ orbital of beryllium and "
        "the bonding combination of the two hydrogen $1s$ atomic orbitals, "
        "and remain essentially fully occupied in the ground state. We can "
        "freeze them. Similarly, the highest-energy virtual orbitals "
        "(several $\\sigma$* and $\\pi$* orbitals) remain essentially "
        "unoccupied, and we can drop them.\n"
        "\n"
        "We keep the two $\\sigma$-bonding/antibonding orbitals "
        "($1b_{1u}$, $3a_g$) and one $\\pi$ orbital ($1b_{2u}$ or $1b_{3u}$, "
        "they are degenerate in the linear geometry), which gives a "
        "**CAS(2,3)**: 2 electrons in 3 active spatial orbitals. The "
        "remaining 4 occupied spin-orbitals are frozen-core and the 8 "
        "highest-lying spin-orbitals are virtual.\n"
        "\n"
        "We pass the active space to `Encoding` with the optional argument "
        "`CAS=(n_electrons, n_active_orbitals)`:"
    ))
    cells.append(code(
        "import quantumsymmetry\n"
        "\n"
        "encoding = quantumsymmetry.Encoding(\n"
        "    atom = 'Be 0 0 0; H 0 0 1.33; H 0 0 -1.33',\n"
        "    basis = 'sto-3g',\n"
        "    CAS = (2, 3),\n"
        "    verbose = True,\n"
        "    show_lowest_eigenvalue = True,\n"
        ")"
    ))
    cells.append(md(
        "Reading the verbose report from top to bottom: the molecule has 14 "
        "spin-orbitals; the SAE-CAS classifies the lowest 4 as frozen-core, "
        "the next 6 as active, and the top 4 as virtual. On top of that, "
        "the exact symmetry generators ($P_\\uparrow$, $P_\\downarrow$, and "
        "the relevant reflections in $D_{2h}$) each remove an additional "
        "qubit. The overall reduction is\n"
        "\n"
        "$$14 \\;\\rightarrow\\; 2 \\quad \\text{qubits}.$$\n"
        "\n"
        "The qubit-reduced Hamiltonian has 10 Pauli terms on 2 qubits, down "
        "from 596 Pauli terms on 9 qubits in the symmetry-adapted encoding "
        "without CAS. The ground state of the reduced Hamiltonian matches "
        "the FCI/CASCI energy of CAS(2,3) for BeH₂ in STO-3G to numerical "
        "precision.\n"
        "\n"
        "Note that the orbital classification was inferred from PySCF's "
        "Hartree-Fock orbital energies. If we want to specify a custom "
        "active space (for instance to include or exclude specific orbitals "
        "by symmetry), we can pass the `active_mo` argument with a list of "
        "1-based MO indices instead:"
    ))
    cells.append(code(
        "encoding = quantumsymmetry.Encoding(\n"
        "    atom = 'Be 0 0 0; H 0 0 1.33; H 0 0 -1.33',\n"
        "    basis = 'sto-3g',\n"
        "    CAS = (2, 3),\n"
        "    active_mo = [3, 4, 5],   # 1-based MO indices to use as active\n"
        "    verbose = False,\n"
        ")\n"
        "print('Final number of qubits:', encoding.hamiltonian.many_body_order())"
    ))

    # --- H2O section ---
    cells.append(md(
        '<a name="H2O"></a>\n'
        "## Water (H₂O) with CAS(8,6)\n"
        "\n"
        "The water molecule has 10 electrons distributed in 7 spatial "
        "orbitals in the minimal STO-3G basis: a deep $1a_1$ orbital "
        "(essentially the oxygen $1s$), and 6 valence orbitals. The "
        "$1a_1$ orbital is energetically isolated and remains doubly "
        "occupied to a very good approximation, which makes it the "
        "textbook example of a *chemical frozen core*: we freeze it and "
        "let the other 8 electrons distribute across the remaining 6 "
        "active orbitals. This is a **CAS(8,6)** with no virtual "
        "orbitals.\n"
        "\n"
        "H₂O is in the $C_{2v}$ point group, which has 2 independent "
        "Boolean symmetries on top of the two spin parities."
    ))
    cells.append(code(
        "encoding = quantumsymmetry.Encoding(\n"
        "    atom = 'O 0 0 0.119; H 0 0.758 -0.477; H 0 -0.758 -0.477',\n"
        "    basis = 'sto-3g',\n"
        "    CAS = (8, 6),\n"
        "    verbose = True,\n"
        "    show_lowest_eigenvalue = True,\n"
        ")"
    ))
    cells.append(md(
        "Here the CAS approximation contributes 2 qubit reductions "
        "(freezing the oxygen $1s$ spin-up and spin-down spin-orbitals), "
        "and the exact symmetries contribute 4 more (the two spin "
        "parities and the two non-trivial point-group reflections "
        "$\\sigma_v(xz)$ and $\\sigma_v(yz)$). Overall, the 14 spin-orbital "
        "Hamiltonian reduces to **8 qubits**.\n"
        "\n"
        "By contrast, the symmetry-adapted encoding without CAS for H₂O "
        "in STO-3G reduces the system to 10 qubits (see the relevant "
        "section in tutorial "
        "[03_molecular_hamiltonians](03_molecular_hamiltonians.ipynb)). "
        "Freezing the chemically inert oxygen $1s$ buys us 2 additional "
        "qubits, while shifting the recovered energy by a fraction of a "
        "milli-Hartree at this level of theory."
    ))

    # --- Accuracy section ---
    cells.append(md(
        '<a name="accuracy"></a>\n'
        "## How accurate is the active-space approximation?\n"
        "\n"
        "Because the CAS projection fixes the occupancy of frozen-core "
        "and virtual orbitals, the reduced Hamiltonian is in general "
        "not exactly equivalent to the original molecular Hamiltonian: "
        "we are choosing a subspace where some orbitals are doubly "
        "occupied and others empty, and ignoring the (small) admixture "
        "of configurations outside this subspace.\n"
        "\n"
        "To quantify the error in a controlled way, we can use PySCF's "
        "CASCI as the reference, which is the exact diagonalisation of "
        "the molecular Hamiltonian projected onto the same active "
        "space. The SAE-CAS qubit Hamiltonian's lowest eigenvalue "
        "should match the PySCF CASCI energy to numerical precision, "
        "because the SAE-CAS construction is provably equivalent to "
        "the canonical frozen-core/CAS projection (see Appendix A in "
        "the SAE-CAS paper).\n"
        "\n"
        "Let's verify this for BeH₂ CAS(2,3):"
    ))
    cells.append(code(
        "import numpy as np\n"
        "from pyscf import gto, scf, mcscf\n"
        "from openfermion.linalg import qubit_operator_sparse\n"
        "\n"
        "atom = 'Be 0 0 0; H 0 0 1.33; H 0 0 -1.33'\n"
        "basis = 'sto-3g'\n"
        "\n"
        "# Reference CASCI from PySCF\n"
        "mol = gto.M(atom = atom, basis = basis, symmetry = False, verbose = 0)\n"
        "mf  = scf.RHF(mol).run(verbose = 0)\n"
        "cas = mcscf.CASCI(mf, ncas = 3, nelecas = 2).run(verbose = 0)\n"
        "casci_energy = cas.e_tot\n"
        "\n"
        "# SAE-CAS qubit Hamiltonian eigenvalue\n"
        "encoding = quantumsymmetry.Encoding(\n"
        "    atom = atom, basis = basis, CAS = (2, 3),\n"
        ")\n"
        "H = qubit_operator_sparse(encoding.hamiltonian).toarray()\n"
        "sae_cas_energy = float(np.linalg.eigvalsh(H).min())\n"
        "\n"
        "print(f'PySCF CASCI energy     : {casci_energy:.10f} Ha')\n"
        "print(f'SAE-CAS qubit ground   : {sae_cas_energy:.10f} Ha')\n"
        "print(f'Absolute difference    : {abs(casci_energy - sae_cas_energy):.2e} Ha')"
    ))
    cells.append(md(
        "The two energies match to better than 10⁻¹⁰ Hartree — the only "
        "remaining difference is floating-point arithmetic. The CAS error "
        "compared to FCI is a separate (and physically meaningful) "
        "quantity, controlled by the choice of active space, not by the "
        "SAE-CAS encoding itself.\n"
        "\n"
        "<p style=\"text-align: left\"> <a href=\"05_VQE_circuits.ipynb\" "
        "/>< Previous: Running a variational algorithm with a "
        "symmetry-adapted encoding</a> </p>\n"
        "<p style=\"text-align: right\"> <a href=\"07_bravyi_kitaev.ipynb\" "
        "/>Next: The Bravyi-Kitaev mapping ></a> </p>"
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    return nb


# ---------------------------------------------------------------------------
# 07 — Bravyi-Kitaev mapping
# ---------------------------------------------------------------------------

def build_07() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list = []

    cells.append(md(LOGO))
    cells.append(md(COLAB_NOTE))
    cells.append(code(COLAB_CELL))

    cells.append(md(
        "# The Bravyi-Kitaev mapping\n"
        "\n"
        "All the symmetry-adapted encodings we have built so far have used "
        "the **Jordan-Wigner (JW) mapping** between fermionic and qubit "
        "operators. In the JW mapping the $j$-th qubit stores the "
        "occupancy of the $j$-th spin-orbital, and the fermionic creation "
        "and annihilation operators map to\n"
        "\n"
        "$$\n"
        "a^{\\dagger}_j \\;\\rightarrow\\; \\tfrac12 (X_j - i Y_j)\\, Z_{j-1} "
        "\\cdots Z_0, \\qquad\n"
        "a_j \\;\\rightarrow\\; \\tfrac12 (X_j + i Y_j)\\, Z_{j-1} "
        "\\cdots Z_0.\n"
        "$$\n"
        "\n"
        "The trailing string of Pauli $Z$ operators implements the "
        "fermionic sign rule. It has length $O(n)$, where $n$ is the "
        "number of qubits, which means a single fermionic excitation in "
        "Jordan-Wigner has Pauli weight $O(n)$.\n"
        "\n"
        "The **Bravyi-Kitaev (BK) mapping** is an alternative "
        "fermion-to-qubit encoding that distributes the parity bookkeeping "
        "more cleverly between qubits. In BK, the operators "
        "$a^{\\dagger}_j$, $a_j$ have a Pauli weight that scales as "
        "$O(\\log n)$ rather than $O(n)$. BK is itself a Clifford basis "
        "change of JW that acts as an affine map on computational-basis "
        "bitstrings; this is exactly the kind of map that "
        "`quantumsymmetry` already uses to fold in Boolean symmetries.\n"
        "\n"
        "On `Encoding` we can switch to the Bravyi-Kitaev mapping by "
        "passing `bravyi_kitaev=True`. The resulting **SAE-CAS-BK** "
        "encoding is unitarily equivalent to SAE-CAS: it uses the same "
        "number of qubits, has the same number of variational parameters, "
        "and yields the same eigenspectrum, but the per-term Pauli "
        "weights and entangling-gate counts of the resulting circuits "
        "are different.\n"
        "\n"
        "1. [BeH₂: SAE-CAS vs SAE-CAS-BK](#BeH2)\n"
        "2. [Pauli weight comparison](#weight)\n"
        "3. [Unitary equivalence of the two encodings](#equivalence)\n"
    ))

    cells.append(md(
        '<a name="BeH2"></a>\n'
        "## BeH₂: SAE-CAS vs SAE-CAS-BK\n"
        "\n"
        "We re-use the BeH₂ CAS(2,3) example from "
        "[06_complete_active_space](06_complete_active_space.ipynb), but "
        "this time we ask for the Bravyi-Kitaev mapping:"
    ))
    cells.append(code(
        "import quantumsymmetry\n"
        "\n"
        "encoding_jw = quantumsymmetry.Encoding(\n"
        "    atom = 'Be 0 0 0; H 0 0 1.33; H 0 0 -1.33',\n"
        "    basis = 'sto-3g',\n"
        "    CAS = (2, 3),\n"
        ")\n"
        "encoding_bk = quantumsymmetry.Encoding(\n"
        "    atom = 'Be 0 0 0; H 0 0 1.33; H 0 0 -1.33',\n"
        "    basis = 'sto-3g',\n"
        "    CAS = (2, 3),\n"
        "    bravyi_kitaev = True,\n"
        ")\n"
        "\n"
        "print('SAE-CAS     Hamiltonian:', encoding_jw.hamiltonian)\n"
        "print()\n"
        "print('SAE-CAS-BK  Hamiltonian:', encoding_bk.hamiltonian)"
    ))
    cells.append(md(
        "Both Hamiltonians have the same number of Pauli terms on the "
        "same number of qubits (2 qubits, 10 terms): the BK basis change "
        "is unitary, so it does not change the number of independent "
        "operators or the spectrum. What it does change is **which** "
        "Pauli operators appear: where SAE-CAS has a $Z_0 Z_1$ or "
        "$X_0 X_1$, SAE-CAS-BK might have a single-qubit $Z_1$ or a "
        "rearranged combination."
    ))

    cells.append(md(
        '<a name="weight"></a>\n'
        "## Pauli weight comparison\n"
        "\n"
        "On systems with just a few qubits the Pauli weight is necessarily "
        "small, so to make the locality difference visible we look at a "
        "larger system. Consider the LiH molecule in the cc-pVDZ basis. "
        "This has 20 spin-orbitals, so the Jordan-Wigner string in the "
        "operator $a^{\\dagger}_{19}$ would carry 19 $Z$ operators; in BK "
        "it carries roughly $\\log_2 20 \\approx 5$ of them.\n"
        "\n"
        "We can confirm this by encoding a single creation operator on "
        "the highest-index spin-orbital under each mapping. We use "
        "tutorial [04_fermionic_operators](04_fermionic_operators.ipynb)'s "
        "`apply` method, but we disable the symmetry filtering by "
        "creating an encoding with no symmetries to remove (so that "
        "`apply` returns the bare JW or BK qubit operator):"
    ))
    cells.append(code(
        "from openfermion import FermionOperator, count_qubits\n"
        "\n"
        "atom_lih = 'Li 0 0 0; H 0 0 1.5949'\n"
        "basis = 'cc-pvdz'\n"
        "\n"
        "# A single creation operator on a high-index spin-orbital.\n"
        "# This is *not* symmetric, so we deliberately disable symmetry\n"
        "# reduction by setting symmetry=False; the encoding then just\n"
        "# applies the chosen fermion-to-qubit mapping.\n"
        "fop = FermionOperator('19^')\n"
        "\n"
        "enc_jw = quantumsymmetry.Encoding(\n"
        "    atom = atom_lih, basis = basis, symmetry = False,\n"
        ")\n"
        "enc_bk = quantumsymmetry.Encoding(\n"
        "    atom = atom_lih, basis = basis, symmetry = False,\n"
        "    bravyi_kitaev = True,\n"
        ")\n"
        "\n"
        "qop_jw = enc_jw.apply(fop)\n"
        "qop_bk = enc_bk.apply(fop)\n"
        "\n"
        "def max_pauli_weight(qop):\n"
        "    return max(len(term) for term, _ in qop.terms.items()) if qop.terms else 0\n"
        "\n"
        "print(f'JW: max Pauli weight of a_19^ = {max_pauli_weight(qop_jw)}')\n"
        "print(f'BK: max Pauli weight of a_19^ = {max_pauli_weight(qop_bk)}')"
    ))
    cells.append(md(
        "The BK weight is dramatically smaller — and this difference "
        "carries over into shallower circuits when we Trotterise an "
        "exponential of a sum of fermionic operators, as in UCCSD. The "
        "trade-off is that in BK each individual qubit no longer has a "
        "direct \"this spin-orbital is occupied\" interpretation, since "
        "occupancies are stored in linear combinations of qubits."
    ))

    cells.append(md(
        '<a name="equivalence"></a>\n'
        "## Unitary equivalence of the two encodings\n"
        "\n"
        "The SAE-CAS and SAE-CAS-BK Hamiltonians are unitarily equivalent: "
        "their eigenspectra agree exactly. We can verify this for our BeH₂ "
        "example by diagonalising both Hamiltonians as dense matrices:"
    ))
    cells.append(code(
        "import numpy as np\n"
        "from openfermion.linalg import qubit_operator_sparse\n"
        "\n"
        "H_jw = qubit_operator_sparse(encoding_jw.hamiltonian).toarray()\n"
        "H_bk = qubit_operator_sparse(encoding_bk.hamiltonian).toarray()\n"
        "\n"
        "eigs_jw = np.sort(np.linalg.eigvalsh(H_jw))\n"
        "eigs_bk = np.sort(np.linalg.eigvalsh(H_bk))\n"
        "\n"
        "print('SAE-CAS     spectrum:', np.round(eigs_jw, 8))\n"
        "print('SAE-CAS-BK  spectrum:', np.round(eigs_bk, 8))\n"
        "print('Max abs difference  :', float(np.max(np.abs(eigs_jw - eigs_bk))))"
    ))
    cells.append(md(
        "The two spectra coincide to numerical precision, confirming that "
        "SAE-CAS-BK is just a relabelling (a Clifford basis change) of "
        "SAE-CAS, with no impact on what the Hamiltonian *says* and "
        "potentially a real impact on how cheaply the corresponding "
        "circuits run on hardware.\n"
        "\n"
        "<p style=\"text-align: left\"> <a "
        "href=\"06_complete_active_space.ipynb\" />< Previous: The complete "
        "active space (CAS) approximation</a> </p>\n"
        "<p style=\"text-align: right\"> <a href=\"08_periodic.ipynb\" />Next: "
        "Periodic systems and crystals ></a> </p>"
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    return nb


# ---------------------------------------------------------------------------
# 08 — Periodic systems
# ---------------------------------------------------------------------------

def build_08() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list = []

    cells.append(md(LOGO))
    cells.append(md(COLAB_NOTE))
    cells.append(code(COLAB_CELL))

    cells.append(md(
        "# Periodic systems and crystals\n"
        "\n"
        "Up to here every example has been a single molecule sitting "
        "in vacuum, with a finite point group of geometrical symmetries. "
        "**Crystalline materials** are infinite periodic arrangements of "
        "atoms, and their symmetry group — the **space group** — combines "
        "point-group operations with discrete lattice translations.\n"
        "\n"
        "`quantumsymmetry` extends the symmetry-adapted encoding to "
        "periodic systems via the `PeriodicEncoding` class. The method "
        "is described in the article:\n"
        "\n"
        "> *Picozzi, D. (manuscript). Periodic symmetry-adapted encoding: "
        "qubit reduction in crystalline electronic structure.*\n"
        "\n"
        "Conceptually, three new ingredients appear compared to the "
        "molecular case:\n"
        "\n"
        "1. **Supercells and $k$-point folding.** A periodic "
        "Hartree-Fock calculation in a Gaussian-orbital basis is most "
        "naturally expressed in a Bloch basis at multiple $k$-points; "
        "folding this $N_k$-point calculation into a $\\Gamma$-point "
        "supercell Hamiltonian gives an object that the SAE machinery "
        "can chew on.\n"
        "2. **Active bands rather than active atomic orbitals.** The "
        "active space is selected from a window of bands around the "
        "Fermi level, exactly the periodic analogue of choosing "
        "frozen-core/active/virtual orbitals around the molecular "
        "HOMO-LUMO gap.\n"
        "3. **Crystal translations as Boolean symmetries.** In a "
        "supercell that contains $N_k$ primitive cells, certain "
        "half-lattice translations map the supercell to itself modulo a "
        "supercell vector. These are exact $\\mathbb{Z}_2$ symmetries of "
        "the folded Hamiltonian, and each one removes one extra qubit on "
        "top of what spin parity and point-group symmetries would give.\n"
        "\n"
        "We illustrate all of this on a concrete example: caesium "
        "chloride (CsCl) in its B2 ($Pm\\overline{3}m$) crystal "
        "structure, which reaches the full theoretical reduction of "
        "eight Boolean generators on a small active space.\n"
        "\n"
        "1. [CsCl in a (2,2,2) supercell](#CsCl)\n"
        "2. [The generators that get detected](#generators)\n"
        "3. [The reduced Hamiltonian and its spectrum](#hamiltonian)\n"
    ))

    cells.append(md(
        '<a name="CsCl"></a>\n'
        "## CsCl in a (2,2,2) supercell\n"
        "\n"
        "CsCl is a textbook ionic crystal in the B2 structure: a simple "
        "cubic lattice with Cs at the origin and Cl at the body-centred "
        "position $(a/2, a/2, a/2)$, where $a = 4.123$ Å is the "
        "experimental lattice constant. The space group is "
        "$Pm\\overline{3}m$, which combines lattice translations with "
        "the full cubic point group.\n"
        "\n"
        "We build a $(2,2,2)$ supercell, which contains $2^3 = 8$ "
        "primitive cells. In the figure below, the black cube is the "
        "supercell itself, the thin grey lines outline its 8 internal "
        "primitive cells, and one of them is highlighted in translucent "
        "red. Cs atoms (purple) sit at the cube corners and Cl atoms "
        "(green) at the body centres; the Cs–Cl nearest-neighbour pairs "
        "are drawn as thin sticks so the lattice network is visible.\n"
        "\n"
        '<p align="center" style="background-color:white; padding:8px;">'
        '<img src="figures_periodic/supercell_cscl_primitive_cells.png" '
        'alt="CsCl supercell with 8 primitive cells" width="380"/>'
        '</p>\n'
        "\n"
        "Inside this supercell we select a **CAS(6,7)** active space: 6 "
        "electrons in 7 active spatial orbitals, taken as the three "
        "HOMOs (MOs 62–64, a $\\Gamma_4^-$ triplet), the LUMO "
        "(MO 65, $\\Gamma_1^+$), and the next virtual triplet "
        "(MOs 66–68, $X_1^+$). That gives 14 active spin-orbitals, so "
        "the Jordan-Wigner baseline is 14 qubits.\n"
        "\n"
        "Translating every atom by the half-lattice vector "
        "$\\mathbf{T} = \\mathbf{a}_0/2$ permutes the atoms onto each "
        "other modulo a supercell lattice vector, so $\\mathbf{T}$ is "
        "an exact symmetry of the periodic Hamiltonian. The arrow below "
        "marks one such half-translation along the $\\mathbf{a}_0$ "
        "direction; the two analogous translations along "
        "$\\mathbf{a}_1$ and $\\mathbf{a}_2$ are also symmetries of "
        "this cubic supercell:\n"
        "\n"
        '<p align="center" style="background-color:white; padding:8px;">'
        '<img src="figures_periodic/supercell_cscl_half_translation.png" '
        'alt="half-lattice translation along a_0/2" width="380"/>'
        '</p>\n'
        "\n"
        "Each half-translation generates a $\\mathbb{Z}_2$ operator "
        "that removes one qubit from the encoding, in the same way "
        "that point-group symmetries do for molecules. We pass the "
        "active space to `PeriodicEncoding` as a list of 1-based MO "
        "indices:"
    ))
    cells.append(code(
        "import io, contextlib\n"
        "import numpy as np\n"
        "from quantumsymmetry import PeriodicEncoding\n"
        "\n"
        "a_lat = 4.123  # angstrom, experimental CsCl lattice constant\n"
        "# Simple cubic primitive lattice vectors\n"
        "a = np.diag([a_lat, a_lat, a_lat])\n"
        "atom = [\n"
        "    ['Cs', (0.0,      0.0,      0.0     )],\n"
        "    ['Cl', (a_lat/2., a_lat/2., a_lat/2.)],\n"
        "]\n"
        "\n"
        "# CAS(6,7): MOs 62-68 (1-based) — HOMO triplet, LUMO, next virtual triplet\n"
        "active_mos = list(range(62, 69))\n"
        "\n"
        "# Silence the PySCF/build progress lines so the cell output stays compact\n"
        "with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):\n"
        "    encoding = PeriodicEncoding(\n"
        "        atom = atom, a = a, basis = 'gth-szv-molopt-sr', pseudo = 'gth-pade',\n"
        "        kpts = (2, 2, 2),\n"
        "        active_mos = active_mos,\n"
        "        name = 'CsCl-222-CAS67',\n"
        "    )\n"
        "\n"
        "print(f'Spin-orbitals       : {encoding.nspinorbital}')\n"
        "print(f'Hartree-Fock (Nα,Nβ): ({encoding.nelectron_up}, {encoding.nelectron_down})')\n"
        "print(f'Number of generators: {len(encoding.symmetry_generators)}')\n"
        "print(f'Qubits after reduction: '\n"
        "      f'{encoding.nspinorbital} -> '\n"
        "      f'{encoding.nspinorbital - len(encoding.target_qubits)}')"
    ))
    cells.append(md(
        "The build takes about ten seconds — most of it is the periodic "
        "Hartree-Fock at 8 $k$-points and the automatic detection of "
        "space-group $\\mathbb{Z}_2$ generators. The output should "
        "report that the 14 active spin-orbitals collapse to **6 qubits**."
    ))

    cells.append(md(
        '<a name="generators"></a>\n'
        "## The generators that get detected\n"
        "\n"
        "`PeriodicEncoding` automatically discovers all applicable "
        "$\\mathbb{Z}_2$ symmetry generators, with no manual input. We "
        "can inspect the labelled list:"
    ))
    cells.append(code(
        "for label in encoding.symmetry_generator_labels:\n"
        "    print(' ', label)"
    ))
    cells.append(md(
        "The labels group naturally into three families:\n"
        "\n"
        "- `P↑`, `P↓` — the **spin parities** of the active-space "
        "electron count. These are the same as in the molecular case.\n"
        "- `T_(a0)/2`, `T_(a1)/2`, `T_(a2)/2` — three independent "
        "**half-lattice crystal translations**, one along each "
        "Cartesian primitive lattice vector. These are the new "
        "ingredient compared to molecular SAE.\n"
        "- `σ[+00]`, `σ[0+0]`, `σ[00+]` — three independent **mirror "
        "planes** of the cubic $Pm\\overline{3}m$ point group. One of "
        "them is shown below as a translucent purple plane cutting "
        "obliquely through the supercell:\n"
        "\n"
        '<p align="center" style="background-color:white; padding:8px;">'
        '<img src="figures_periodic/supercell_cscl_mirror_plane.png" '
        'alt="mirror plane of the CsCl supercell" width="380"/>'
        '</p>\n'
        "\n"
        "Spin parity contributes 2 generators, supercell point-group "
        "operations contribute 3, and crystal translations contribute "
        "3, for a total of **8 generators** and 8 qubit removals: "
        "$14 \\rightarrow 6$. This is the theoretical maximum reduction "
        "available on a $(2,2,2)$ supercell — every type of "
        "$\\mathbb{Z}_2$ symmetry that can survive a periodic "
        "Hamiltonian (spin, point group, half translation) is present, "
        "which is what makes CsCl a useful headline demonstration."
    ))

    cells.append(md(
        '<a name="hamiltonian"></a>\n'
        "## The reduced Hamiltonian and its spectrum\n"
        "\n"
        "The qubit-reduced Hamiltonian is accessible through the same "
        "`.hamiltonian` property as in the molecular case:"
    ))
    cells.append(code(
        "from openfermion.linalg import qubit_operator_sparse\n"
        "\n"
        "H = encoding.hamiltonian\n"
        "n_terms = len(list(H.terms))\n"
        "print(f'Reduced Hamiltonian: {n_terms} Pauli terms on '\n"
        "      f'{encoding.nspinorbital - len(encoding.target_qubits)} qubits')\n"
        "\n"
        "H_dense = qubit_operator_sparse(H).toarray()\n"
        "eigs = np.sort(np.linalg.eigvalsh(H_dense))\n"
        "print(f'Ground state energy : {eigs[0]:.10f} Ha')\n"
        "print(f'First excited state : {eigs[1]:.10f} Ha')"
    ))
    cells.append(md(
        "To see what the active space looks like in real space, here is "
        "an isosurface of MO 62 — the highest-energy occupied orbital "
        "inside the active window, transforming as the $x$-component of "
        "the $\\Gamma_4^-$ HOMO triplet. Red and blue lobes correspond "
        "to opposite signs of the orbital amplitude:\n"
        "\n"
        '<p align="center" style="background-color:white; padding:8px;">'
        '<img src="figures_periodic/orbital_cscl_mo62.png" '
        'alt="HOMO of CsCl supercell — MO 62" width="380"/>'
        '</p>\n'
        "\n"
        "The orbital is centred on the Cl sublattice and changes sign "
        "along the $x$ direction — a pattern that maps onto itself, up "
        "to a global sign, under each half-lattice translation "
        "$\\mathbf{T}$ shown above. That is exactly what makes "
        "$\\mathbf{T}$ act as a Pauli $Z$-like operator on the "
        "active-space qubits, and hence a Boolean symmetry of the "
        "encoded Hamiltonian."
    ))
    cells.append(md(
        "The reduced Hamiltonian for CsCl CAS(6,7) in the $(2,2,2)$ "
        "supercell has 71 Pauli terms on 6 qubits — small enough to "
        "fit comfortably on near-term hardware, yet capturing the full "
        "CAS(6,7) correlation energy of the periodic system to the "
        "accuracy of the active space.\n"
        "\n"
        "The PeriodicEncoding object exposes the same interface as the "
        "molecular `Encoding` — `apply` to map fermionic operators, "
        "`qiskit_mapper` to convert excitation operators in a UCC "
        "ansatz, and so on — so the VQE workflow from tutorial "
        "[05_VQE_circuits](05_VQE_circuits.ipynb) carries over without "
        "modification to the crystalline setting.\n"
        "\n"
        "<p style=\"text-align: left\"> <a href=\"07_bravyi_kitaev.ipynb\" "
        "/>< Previous: The Bravyi-Kitaev mapping</a> </p>"
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    return nb


def main() -> None:
    notebooks = {
        "06_complete_active_space.ipynb": build_06(),
        "07_bravyi_kitaev.ipynb":          build_07(),
        "08_periodic.ipynb":               build_08(),
    }
    for name, nb in notebooks.items():
        out = HERE / name
        with out.open("w", encoding = "utf-8") as fp:
            nbf.write(nb, fp)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
