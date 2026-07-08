# The symmetry-adapted-encoding core (.core / .encoding) is built on pyscf +
# openfermion and does NOT require qiskit-nature.  The tree-circuit ansatz in
# .treecircuit only needs numpy/scipy/qiskit.  Both quantum-chemistry packages
# (pyscf) and the qiskit-nature interop (.qiskit_converter) are kept optional so
# a lean deployment can still expose the parts it has.
import importlib.util as _importlib_util
import warnings as _warnings

# --- symmetry-adapted encoding (needs pyscf; not qiskit-nature) ---------------
try:
    from .core import apply_encoding
    from .encoding import Encoding, PeriodicEncoding, reduced_hamiltonian, make_encoding
except ImportError as _exc:  # pragma: no cover
    # Silent when pyscf is genuinely absent (a deliberate lean install); warn if
    # pyscf is present but the encoding still failed to import (a real problem).
    if _importlib_util.find_spec("pyscf") is not None:
        _warnings.warn(
            "quantumsymmetry: the symmetry-adapted-encoding API (Encoding, "
            "make_encoding, reduced_hamiltonian) could not be imported, so only "
            "the tree-circuit API is available. Underlying error: "
            f"{_exc!r}",
            stacklevel=2,
        )

# --- optional qiskit-nature interop (UCC ansatz, qiskit QubitMapper) -----------
try:
    from .qiskit_converter import UCC_SAE_circuit, SymmetryAdaptedEncodingQubitConverter, HartreeFockCircuit, make_excitation_ops
except ImportError as _exc:  # pragma: no cover
    # Silent when qiskit-nature is simply not installed (an optional extra); warn
    # if it is present but the interop failed (e.g. a qiskit version mismatch).
    if _importlib_util.find_spec("qiskit_nature") is not None:
        _warnings.warn(
            "quantumsymmetry: the qiskit-nature interop (UCC_SAE_circuit, "
            "SymmetryAdaptedEncodingQubitConverter, ...) could not be imported; "
            "the core encoding API is unaffected. Underlying error: "
            f"{_exc!r}",
            stacklevel=2,
        )
from .treecircuit import (
    MinimalCircuit,
    minimize_energy,
    VQEResult,
    WindowedPatienceStopper,
    evolve_realtime,
    RealTimeResult,
    project_vqd,
    ProjectionResult,
    sample_sector_haar,
    sieve_states_by_symmetry,
    get_excitations_from_states,
    dressing_generators,
    make_dressing_pool,
    apply_dressing,
    dressing_diagnostics,
    decoupling_surrogate,
    total_spin_expectation,
)