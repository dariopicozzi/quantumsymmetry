"""Regression test for the symmetry-adapted excitation-operator builder.

``make_excitation_ops`` builds the single/double excitation generators about a
Jordan-Wigner reference determinant and pushes them through the symmetry-adapted
encoding.  It used to call the removed qiskit-nature ``FermionicOp`` dense-label
constructor (``register_length=`` / ``display_format=``) and raised ``TypeError``
on qiskit-nature >= 0.6; it now builds OpenFermion ``FermionOperator`` generators
instead.  This guards that path.

The module only needs the optional qiskit-nature interop to *import*
``make_excitation_ops`` (it lives in ``qiskit_converter``); the builder itself no
longer depends on qiskit-nature.  Skip gracefully when the extra is absent.
"""

import pytest
import numpy as np

try:
    from quantumsymmetry import make_excitation_ops, make_encoding
    from quantumsymmetry.core import HartreeFock_ket
    from quantumsymmetry.qiskit_converter import get_num_particles_spin_orbitals
except ImportError:
    pytest.skip(
        "qiskit-nature interop unavailable (optional 'quantumsymmetry[qiskit]' "
        "extra not installed)",
        allow_module_level=True,
    )

from pyscf import gto, scf


def _hf_reference(atom, basis, charge=0, spin=0):
    """Full Jordan-Wigner Hartree-Fock determinant as a bitstring."""
    mol = gto.M(atom=atom, basis=basis, symmetry=True,
                charge=charge, spin=spin, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    _, num_spin_orbitals = get_num_particles_spin_orbitals(atom, basis, charge, spin)
    return format(HartreeFock_ket(mf.mo_occ), f'0{num_spin_orbitals}b')


def test_make_excitation_ops_h2():
    atom, basis = 'H 0 0 0; H 0 0 0.7414', 'sto3g'
    encoding = make_encoding(atom, basis)
    reference = _hf_reference(atom, basis)        # full JW HF determinant '0011'

    ops = make_excitation_ops(reference, encoding)

    # 4 singles + 1 double on the 2-electron / 4-spin-orbital reference.
    assert len(ops) == 5

    # In the 1-qubit symmetry-adapted encoding only one excitation survives the
    # symmetry projection -- exactly UCC_SAE_circuit's single variational
    # parameter -- and it is the off-diagonal Y generator.
    nonzero = [o for o in ops
               if o is not None and o.coeffs.size
               and np.any(np.abs(o.coeffs) > 1e-12)]
    assert len(nonzero) == 1
    assert nonzero[0].num_qubits == 1
    assert nonzero[0].paulis.to_labels() == ['Y']
