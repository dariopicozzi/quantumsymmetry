"""Pruned binary-tree variational quantum circuits.

This subpackage provides the hardware-efficient variational circuit ansatz
based on a binary-tree decomposition of projective Hilbert space, together
with its exact Fubini-Study metric, coordinate maps, singular-initialisation
optimizers, and chemistry symmetry helpers.

The implementation is organised into thematic modules:

* :mod:`~quantumsymmetry.treecircuit.tree` -- pure tree combinatorics.
* :mod:`~quantumsymmetry.treecircuit.affine` -- affine-block construction.
* :mod:`~quantumsymmetry.treecircuit.circuit` -- pruned circuit + fast binder.
* :mod:`~quantumsymmetry.treecircuit.metric` -- metric and coordinate maps.
* :mod:`~quantumsymmetry.treecircuit.optimize` -- gradients and singular init.
* :mod:`~quantumsymmetry.treecircuit.minimal_circuit` -- the ``MinimalCircuit`` facade and ``minimize_energy``.
* :mod:`~quantumsymmetry.treecircuit.dynamics` -- real- and imaginary-time evolution.
* :mod:`~quantumsymmetry.treecircuit.sampling` -- sector-Haar (Fubini-Study) sampling.
* :mod:`~quantumsymmetry.treecircuit.dressing` -- Schrieffer-Wolff dressing layer.
* :mod:`~quantumsymmetry.treecircuit.chemistry` -- symmetry sieving helpers.
* :mod:`~quantumsymmetry.treecircuit.spin` -- exact total-spin adaptation.

(The submodules above are importable directly, e.g.
``from quantumsymmetry.treecircuit import tree``.)
"""

# Tree combinatorics
from .tree import (
    classify_params,
    classify_phase_params,
    best_bit_reordering_bnb,
    discard_constant_bits,
    chart_topology,
    node_angles,
    node_amplitudes,
)

# Affine blocks
from .affine import (
    get_affine_blocks,
)

# Circuit construction and compilation
from .circuit import (
    circuit,
    get_qiskit_circuit,
    triangular_gate_list,
    triangular_circuit_list,
    get_cnot_indices_to_remove,
    get_pruned_circuit_list,
    AffineCompiledCircuit,
    compile_affine,
    bind_fast,
)

# Coordinate and metric helpers
from .metric import (
    make_inverse_metric,
    make_cartesian_to_polyspherical,
    make_polyspherical_to_cartesian,
    polyspherical_metric_pruned,
    polyspherical_inv_metric_pruned,
    polyspherical_to_cartesian_pruned,
    cartesian_to_polyspherical_pruned,
)

# Optimizers / singular initialisation
from .optimize import (
    shift_rule_gradient,
    finite_difference_gradient,
    singular_initialise,
    singular_initialise_complex,
    singular_initialise_realtime,
    singular_initialise_realtime_shift_rule,
)

# Chemistry symmetry helpers
from .chemistry import (
    sieve_states_by_symmetry,
    get_excitations_from_states,
    dressing_generators,
)

# Sector-Haar (Fubini--Study) sampling
from .sampling import (
    tree_beta_parameters,
    sample_sector_haar,
    sample_sector_haar_oracle,
)

# Exact total-spin adaptation (gate-free).  The user-facing entry point is
# MinimalCircuit.from_particle_number(..., total_spin=S); the names below are
# the sector CSF machinery and the determinant-chart (gate-angle) utilities.
from .spin import (
    sector_csf_unitary,
    spin_block_columns,
    ballot_leaders,
    ballot_embedding,
    ballot_free_tied,
    spin_constraint_matrix,
    sector_amplitude_map,
    constraint_values,
    constraint_jacobian,
    solve_tied,
    tie_jacobian,
    woodbury_inverse_metric,
    total_spin_expectation,
)

# High-level public API: the MinimalCircuit facade and its drivers.
from .minimal_circuit import (
    MinimalCircuit,
    minimize_energy,
    VQEResult,
    WindowedPatienceStopper,
)
from .dynamics import evolve_realtime, RealTimeResult, project_vqd, ProjectionResult

# Schrieffer--Wolff dressing layer + FS-sampled surrogate oracle
from .dressing import (
    make_dressing_pool,
    apply_dressing,
    dressing_diagnostics,
    decoupling_surrogate,
)
