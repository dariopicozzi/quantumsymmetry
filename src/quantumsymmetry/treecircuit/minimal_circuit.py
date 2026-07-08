"""User-facing interface to the pruned binary-tree ("minimal") ansatz.

This module exposes a single declarative object, :class:`MinimalCircuit`, that
wraps the low-level :mod:`quantumsymmetry.treecircuit` primitives and gives
direct access to the physically meaningful objects of the method:

* the **support set** :math:`S` (the active computational-basis states) and its
  classified binary **tree**,
* the pruned **circuit** :math:`U(\\theta, \\omega)` as a Qiskit
  :class:`~qiskit.circuit.QuantumCircuit`,
* the **state** :math:`|\\psi(\\theta, \\omega)\\rangle` (fast :math:`O(|S|)`
  evaluation, no dense circuit simulation),
* the polyspherical / Bloch **chart coordinates** :math:`(\\theta, \\omega)` and
  the maps :math:`\\psi \\leftrightarrow (\\theta, \\omega)`,
* the Fubini--Study **metric** (its inverse :math:`G^{-1}`),
* gradients and the analytic **singular initialisation**.

The object mirrors the declarative style of :class:`quantumsymmetry.Encoding`::

    mc = MinimalCircuit.from_particle_number(num_spatial_orbitals=2,
                                             num_particles=(1, 1))
    mc.circuit            # parametric Qiskit circuit
    mc.num_parameters
    psi = mc.statevector(theta)

Static ground-state optimisation lives in the separate driver
:func:`minimize_energy`, which consumes a :class:`MinimalCircuit` and a qubit
Hamiltonian (a Qiskit :class:`~qiskit.quantum_info.SparsePauliOp` *or* an
OpenFermion ``QubitOperator`` in the Jordan--Wigner convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .circuit import circuit as _build_circuit
from .tree import chart_topology
from .metric import (
    make_cartesian_to_polyspherical,
    make_inverse_metric,
    make_polyspherical_to_cartesian,
)
from .optimize import (
    finite_difference_gradient,
    shift_rule_gradient,
    singular_initialise,
)

__all__ = [
    "MinimalCircuit",
    "minimize_energy",
    "VQEResult",
    "WindowedPatienceStopper",
]


# ---------------------------------------------------------------------------
# Unified windowed-patience stopping rule (measured in evaluation units).
#
# Shared by the natural-gradient optimiser below and by the external UCCSD
# baselines (gradient descent / COBYLA) so that all benchmarked methods stop on
# *exactly* the same criterion, expressed in cumulative cost-function (energy)
# evaluations rather than optimiser iterations.  This makes the three methods
# directly comparable even though one GD/NG step consumes many evaluations
# while one COBYLA simplex probe consumes a single evaluation.
#
# The rule maintains a Polyak trailing mean ``Ebar`` over the energies recorded
# within the last ``window`` *evaluations*, and declares convergence when
# ``Ebar`` has not improved by more than ``eps`` over the last ``patience``
# *evaluations*.  It uses only the running mean -- never a best-so-far iterate
# -- so it cannot exploit a noise-driven dip below the true expectation value.
# Setting ``window = patience = 1`` with a tiny ``eps`` recovers the plain
# per-step ``|dE| < eps`` rule used in the noiseless regime.
# ---------------------------------------------------------------------------

class WindowedPatienceStopper:
    """Polyak-mean windowed-patience early-stop, measured in evaluation units.

    Parameters
    ----------
    window:
        Number of trailing *evaluations* whose recorded energies are averaged
        into the Polyak mean ``Ebar`` (``1`` = report the latest iterate).
    patience:
        Number of *evaluations* over which the trailing Polyak mean ``Ebar``
        must fail to set a new best (lowest) value -- by more than ``eps`` --
        before convergence is declared.
    eps:
        Minimum ``Ebar`` improvement (in Hartree) that counts as progress.

    Notes
    -----
    The stop *decision* uses a patience-on-best-trailing-mean criterion: we
    track the lowest trailing Polyak mean seen so far and stop once no newer
    trailing mean has beaten it by more than ``eps`` within the last
    ``patience`` evaluations.  This is robust to *non-monotone* optimisers
    (e.g. COBYLA's simplex probes, or noisy stochastic steps), which would
    otherwise trip a naive "improved since the last sample" test the instant a
    single probe goes uphill.  The *reported* energy remains the Polyak mean at
    the stopping point (:attr:`reported_energy`) -- never a best raw energy.
    """

    def __init__(self, window: int, patience: int, eps: float):
        self.window = max(1, int(window))
        self.patience = max(1, int(patience))
        self.eps = float(eps)
        # Parallel arrays of cumulative-evaluation counts, the raw iterate
        # energy recorded at that point, and the Polyak mean ``Ebar`` there.
        self._evals: List[int] = []
        self._energies: List[float] = []
        self._polyak: List[float] = []
        # Best (lowest) trailing Polyak mean seen so far and the cumulative
        # evaluation count at which that best was (last) achieved.
        self._best_polyak: float = float("inf")
        self._best_evals: int = -1

    def update(self, cum_evals: int, energy: float) -> float:
        """Record one ``(cumulative-evaluations, energy)`` sample; return ``Ebar``."""
        self._evals.append(int(cum_evals))
        self._energies.append(float(energy))
        # Polyak mean over every recorded sample within the trailing
        # ``window`` evaluations of the current point.
        lo = int(cum_evals) - self.window
        vals = [e for c, e in zip(self._evals, self._energies) if c > lo]
        ebar = float(np.mean(vals)) if vals else float(energy)
        self._polyak.append(ebar)
        # Update the best trailing mean: a new best requires an improvement of
        # more than ``eps`` over the incumbent so tiny wiggles do not keep
        # resetting the patience clock.
        if ebar < self._best_polyak - self.eps:
            self._best_polyak = ebar
            self._best_evals = int(cum_evals)
        elif not self._evals[:-1]:
            # First sample seeds the incumbent even if it is not an improvement.
            self._best_polyak = ebar
            self._best_evals = int(cum_evals)
        return ebar

    def should_stop(self) -> bool:
        """True once the best trailing ``Ebar`` has stalled for ``patience`` evals."""
        if not self._polyak:
            return False
        now = self._evals[-1]
        # Require at least ``patience`` evaluations of history before stopping.
        if now - self._evals[0] < self.patience:
            return False
        # Stop when no new best trailing mean has been set within the last
        # ``patience`` evaluations.
        return (now - self._best_evals) >= self.patience

    @property
    def reported_energy(self) -> float:
        """Polyak mean at the current (stop) point."""
        return float(self._polyak[-1]) if self._polyak else float("nan")


# ---------------------------------------------------------------------------
# Support-set builders (active leaves).
# ---------------------------------------------------------------------------

def _particle_number_support(
    num_spatial_orbitals: int,
    num_particles: Tuple[int, int],
) -> List[int]:
    """Active leaves for a fixed (n_alpha, n_beta) particle-number sector.

    Bitstring layout ``[alpha_0..alpha_{m-1}, beta_0..beta_{m-1}]`` read
    MSB->LSB, matching the Qiskit-Nature Jordan--Wigner convention.
    """
    n_alpha, n_beta = num_particles
    m = int(num_spatial_orbitals)
    states: List[int] = []
    for occ_a in combinations(range(m), n_alpha):
        a = [0] * m
        for i in occ_a:
            a[i] = 1
        for occ_b in combinations(range(m), n_beta):
            b = [0] * m
            for j in occ_b:
                b[j] = 1
            states.append(int("".join(str(x) for x in (a + b)), 2))
    return states


def _hartree_fock_index(
    num_spatial_orbitals: int,
    num_particles: Tuple[int, int],
) -> int:
    """Integer index of the Hartree--Fock determinant in the same convention."""
    n_alpha, n_beta = num_particles
    m = int(num_spatial_orbitals)
    a = [1 if i < n_alpha else 0 for i in range(m)]
    b = [1 if i < n_beta else 0 for i in range(m)]
    return int("".join(str(x) for x in (a + b)), 2)


def _hamming_weight_support(num_qubits: int, weight: int) -> List[int]:
    """Active leaves = all bitstrings of length ``num_qubits`` with ``weight`` ones."""
    states: List[int] = []
    for ones in combinations(range(num_qubits), weight):
        value = 0
        for bit in ones:
            value |= 1 << bit
        states.append(value)
    return sorted(states)


# ---------------------------------------------------------------------------
# MinimalCircuit.
# ---------------------------------------------------------------------------

class MinimalCircuit:
    """Pruned binary-tree ansatz on a fixed computational-basis support set.

    Parameters
    ----------
    num_qubits:
        Number of qubits the circuit acts on.
    support:
        Iterable of integers -- the active computational-basis states (the
        "active leaves") that span the ansatz support.
    complex:
        If ``True`` the ansatz carries per-leaf phases (``R_z`` blocks) and the
        chart has both amplitude (``theta``) and phase (``omega``) coordinates.
        If ``False`` (default) the ansatz is real-amplitude only.
    reorder:
        Whether to apply the exhaustive branch-and-bound bit reordering before
        pruning. Disabled by default (``False``): on structured active sets
        (symmetry sectors, fixed-Hamming-weight subspaces) it yields no CNOT
        reduction over the cheap constant-bit pass and can be exponentially
        slow, so it is retained only as an opt-in.
    """

    def __init__(
        self,
        num_qubits: int,
        support: Sequence[int],
        *,
        complex: bool = False,
        reorder: bool = False,
    ) -> None:
        self.num_qubits = int(num_qubits)
        self.support = [int(s) for s in support]
        self.complex = bool(complex)
        self.reorder = bool(reorder)

        if not self.support:
            raise ValueError("`support` must contain at least one active leaf.")
        max_state = 1 << self.num_qubits
        for s in self.support:
            if s < 0 or s >= max_state:
                raise ValueError(
                    f"support entry {s} out of range for {self.num_qubits} qubits."
                )

        self._circuit = _build_circuit(
            self.num_qubits,
            active_leaves=self.support,
            reorder=self.reorder,
            complex=self.complex,
        )

        # Lazily-constructed treecircuit callables / structures.
        self._inv_metric_fn = None
        self._c2p_fn = None
        self._p2c_fn = None
        self._topology = None

        # Optional chemistry metadata (set by `from_particle_number`).
        self._hf_meta: Optional[Tuple[int, Tuple[int, int]]] = None

    # -- alternative constructors -------------------------------------------

    @classmethod
    def from_particle_number(
        cls,
        num_spatial_orbitals: int,
        num_particles: Tuple[int, int],
        *,
        complex: bool = False,
        reorder: bool = False,
        total_spin: Optional[float] = None,
        support: Optional[Sequence[int]] = None,
        encoding=None,
    ) -> "MinimalCircuit":
        """Build a circuit spanning a fixed ``(n_alpha, n_beta)`` particle-number sector.

        Uses ``2 * num_spatial_orbitals`` qubits in the Jordan--Wigner ordering
        ``[alpha_0..alpha_{m-1}, beta_0..beta_{m-1}]``.

        ``support`` optionally restricts the sector to a subset of determinant
        leaves (e.g. a point-group irrep from
        :func:`~quantumsymmetry.sieve_states_by_symmetry`) while keeping the
        Hartree--Fock conveniences.

        ``total_spin`` additionally enforces total spin ``S`` *exactly*: the
        returned ansatz varies a binary tree over the spin-``S`` configuration
        state functions (an exactly diagonal metric), realised on the plain
        pruned determinant tree with no change-of-basis gate, so
        ``<S^2> = S (S + 1)`` at every parameter value.  The construction
        targets the highest weight ``Sz = S`` and therefore requires
        ``n_alpha - n_beta = 2 * total_spin`` (real-amplitude only).  See
        :mod:`quantumsymmetry.treecircuit.spin`.

        ``encoding`` optionally lands the ansatz in a symmetry-adapted
        :class:`~quantumsymmetry.Encoding` instead of plain Jordan--Wigner, so
        the circuit acts on ``encoding.encoded_qubits`` and its leaves are the
        encoded determinants.  This composes with ``total_spin`` -- the
        spin-adapted CSF tree is built in the intermediate Jordan--Wigner basis
        and each determinant mapped to its encoded correspondent -- giving an
        exactly spin-adapted ansatz in any encoding.  When ``encoding`` is set,
        ``support`` (if given) is a list of *encoded* computational-basis leaves
        -- e.g. a CAS/model-space block -- and defaults to the encoding's whole
        symmetry-adapted sector.  See :meth:`from_encoding` for the convenience
        form.
        """
        if total_spin is not None:
            if complex:
                raise ValueError(
                    "total_spin is only supported for the real-amplitude "
                    "ansatz (complex=False).")
            from .spin import build_spin_adapted

            return build_spin_adapted(
                num_spatial_orbitals, num_particles, total_spin,
                support=support, reorder=reorder, encoding=encoding)
        m = int(num_spatial_orbitals)
        na, nb = int(num_particles[0]), int(num_particles[1])
        if encoding is not None:
            num_qubits = int(encoding.encoded_qubits)
            if support is None:
                support = [int(s) for s in encoding.symmetry_adapted_support((na, nb))]
        else:
            num_qubits = 2 * m
            if support is None:
                support = _particle_number_support(num_spatial_orbitals, num_particles)
        mc = cls(num_qubits, support, complex=complex, reorder=reorder)
        mc._hf_meta = (m, (na, nb))
        return mc

    @classmethod
    def from_encoding(
        cls,
        encoding,
        *,
        total_spin: Optional[float] = None,
        complex: bool = False,
        reorder: bool = False,
        support: Optional[Sequence[int]] = None,
    ) -> "MinimalCircuit":
        """Build a tree ansatz directly in a symmetry-adapted ``Encoding``.

        Convenience wrapper over :meth:`from_particle_number` that reads the
        spatial-orbital count and ``(n_alpha, n_beta)`` from ``encoding``.  The
        returned circuit acts on ``encoding.encoded_qubits`` and, with
        ``total_spin=S``, is exactly spin-adapted (``<S^2> = S(S+1)`` at every
        parameter value).  Mirrors the plain
        ``MinimalCircuit(encoding.encoded_qubits, encoding.symmetry_adapted_support())``
        idiom with the spin option folded in.
        """
        m = int(encoding.nspinorbital) // 2
        num_particles = (int(encoding.nelectron_up), int(encoding.nelectron_down))
        return cls.from_particle_number(
            m, num_particles, complex=complex, reorder=reorder,
            total_spin=total_spin, support=support, encoding=encoding)

    @classmethod
    def from_hamming_weight(
        cls,
        num_qubits: int,
        weight: int,
        *,
        complex: bool = False,
        reorder: bool = False,
    ) -> "MinimalCircuit":
        """Build a circuit spanning the fixed-Hamming-weight subspace (e.g. QAOA)."""
        support = _hamming_weight_support(int(num_qubits), int(weight))
        return cls(num_qubits, support, complex=complex, reorder=reorder)

    # -- core objects -------------------------------------------------------

    @property
    def circuit(self):
        """The parametric pruned Qiskit :class:`QuantumCircuit`."""
        return self._circuit

    @property
    def num_parameters(self) -> int:
        """Total number of variational parameters in the circuit."""
        return int(self._circuit.num_parameters)

    def bound_circuit(self, theta: Sequence[float]):
        """The Qiskit circuit bound at chart coordinates ``theta``.

        For the plain ansatz the chart coordinates *are* the gate angles;
        spin-adapted variants override this with the classical lowering from
        their chart to the determinant-tree gate angles."""
        return self._circuit.assign_parameters(np.asarray(theta, dtype=float))

    @property
    def tree(self):
        """The classified binary-tree topology of the support set."""
        if self._topology is None:
            self._topology = chart_topology(self.num_qubits, self.support, reorder=self.reorder)
        return self._topology

    # -- chart coordinate maps ---------------------------------------------

    def statevector(
        self,
        theta: Sequence[float],
        omega: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        r"""Reconstruct :math:`|\psi\rangle` from chart coordinates.

        ``theta`` are the amplitude (``R_y``) parameters.  ``omega`` are the
        per-leaf phases and is only used when ``complex=True``.
        """
        if self._p2c_fn is None:
            self._p2c_fn = make_polyspherical_to_cartesian(
                self.num_qubits, self.support, complex=self.complex, reorder=self.reorder
            )
        theta = np.asarray(theta, dtype=float)
        if self.complex:
            if omega is None:
                omega = np.zeros(len(self.support), dtype=float)
            return np.asarray(
                self._p2c_fn(active_vals=theta, phase_active_vals=np.asarray(omega, dtype=float))
            )
        return np.asarray(self._p2c_fn(active_vals=theta))

    def parameters(
        self, statevector: Sequence[complex]
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""Map a state vector to chart coordinates (inverse of :meth:`statevector`).

        Returns ``theta`` when ``complex=False`` and ``(theta, omega)`` when
        ``complex=True``.
        """
        if self._c2p_fn is None:
            self._c2p_fn = make_cartesian_to_polyspherical(
                self.num_qubits, self.support, complex=self.complex, reorder=self.reorder
            )
        result = self._c2p_fn(np.asarray(statevector))
        if self.complex:
            theta, omega = result
            return np.asarray(theta, dtype=float), np.asarray(omega, dtype=float)
        return np.asarray(result, dtype=float)

    # -- geometry -----------------------------------------------------------

    def inverse_metric(self, theta: Sequence[float]) -> np.ndarray:
        r"""Inverse Fubini--Study metric :math:`G^{-1}(\theta)` at ``theta``."""
        if self._inv_metric_fn is None:
            self._inv_metric_fn = make_inverse_metric(
                self.num_qubits, self.support, reorder=self.reorder
            )
        return np.asarray(self._inv_metric_fn(active_vals=np.asarray(theta, dtype=float)), dtype=float)

    # -- gradients & initialisation ----------------------------------------

    def gradient(
        self,
        energy_fn: Callable[[np.ndarray], float],
        theta: Sequence[float],
        *,
        method: str = "shift",
        eps: float = 1e-3,
    ) -> np.ndarray:
        """Gradient of ``energy_fn`` at ``theta``.

        ``method="shift"`` uses the exact four-term parameter-shift rule;
        ``method="fd"`` uses central finite differences with step ``eps``.
        """
        theta = np.asarray(theta, dtype=float)
        if method == "shift":
            return np.asarray(shift_rule_gradient(energy_fn, theta), dtype=float)
        if method == "fd":
            return np.asarray(
                finite_difference_gradient(energy_fn, theta, eps=eps), dtype=float
            )
        raise ValueError(f"Unknown gradient method {method!r} (expected 'shift' or 'fd').")

    def singular_initialise(
        self,
        theta: Sequence[float],
        energy_fn: Callable[[np.ndarray], float],
        *,
        lr: float = 0.15,
        tau: float = 1e-10,
    ) -> np.ndarray:
        """Analytic corner-escape: move off a chart corner along steepest descent."""
        return np.asarray(
            singular_initialise(
                np.asarray(theta, dtype=float), energy_fn, self.num_qubits, self.support, lr,
                tau=tau, reorder=self.reorder
            ),
            dtype=float,
        )

    # -- chemistry conveniences --------------------------------------------

    def hartree_fock_statevector(self) -> np.ndarray:
        """One-hot Hartree--Fock state (requires a particle-number construction)."""
        if self._hf_meta is None:
            raise ValueError(
                "Hartree--Fock state is only defined for circuits built with "
                "`MinimalCircuit.from_particle_number(...)`."
            )
        num_spatial_orbitals, num_particles = self._hf_meta
        idx = _hartree_fock_index(num_spatial_orbitals, num_particles)
        psi = np.zeros(2 ** self.num_qubits)
        psi[idx] = 1.0
        return psi

    def hartree_fock_parameters(self) -> np.ndarray:
        """Chart coordinates of the Hartree--Fock state (amplitude ``theta``)."""
        params = self.parameters(self.hartree_fock_statevector())
        if self.complex:
            return params[0]
        return params

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MinimalCircuit(num_qubits={self.num_qubits}, "
            f"support_size={len(self.support)}, num_parameters={self.num_parameters}, "
            f"complex={self.complex})"
        )


# ---------------------------------------------------------------------------
# Ground-state optimisation driver.
# ---------------------------------------------------------------------------

@dataclass
class VQEResult:
    """Outcome of :func:`minimize_energy`."""

    energy: float
    optimal_parameters: np.ndarray
    history: List[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    num_parameters: int = 0
    num_energy_evaluations: int = 0
    # Optional convergence trace recorded when ``record_trace=True``: one entry
    # ``{"evals": cumulative_energy_evaluations, "energy": raw_iterate_energy,
    # "polyak": Polyak_mean}`` per optimiser step (plus the initial point).
    trace: List[Dict[str, float]] = field(default_factory=list)
    # Polyak trailing mean at the stop point (== ``energy`` when the
    # windowed-patience stopper is active; otherwise ``None``).
    polyak_energy: Optional[float] = None
    polyak_window: int = 0


def _to_sparse_pauli_op(hamiltonian: Any, num_qubits: int):
    """Coerce ``hamiltonian`` to a Qiskit ``SparsePauliOp``.

    Accepts a ``SparsePauliOp`` directly or an OpenFermion ``QubitOperator``
    (Jordan--Wigner).  The OpenFermion qubit index ``q`` maps to Qiskit qubit
    ``q`` (label position ``num_qubits - 1 - q``).
    """
    from qiskit.quantum_info import SparsePauliOp

    if isinstance(hamiltonian, SparsePauliOp):
        return hamiltonian

    # Duck-type the OpenFermion QubitOperator (has a `.terms` dict).
    terms = getattr(hamiltonian, "terms", None)
    if terms is not None:
        labels = []
        for term, coeff in terms.items():
            chars = ["I"] * num_qubits
            for qubit, pauli in term:
                chars[num_qubits - 1 - int(qubit)] = pauli
            labels.append(("".join(chars), complex(coeff)))
        if not labels:
            labels = [("I" * num_qubits, 0.0)]
        return SparsePauliOp.from_list(labels, num_qubits=num_qubits)

    raise TypeError(
        "hamiltonian must be a qiskit SparsePauliOp or an OpenFermion QubitOperator; "
        f"got {type(hamiltonian)!r}."
    )


def _default_max_iterations(num_qubits: int) -> int:
    if num_qubits <= 6:
        return 10000
    if num_qubits <= 10:
        return 15000
    return 20000


def minimize_energy(
    mc: MinimalCircuit,
    hamiltonian: Any,
    *,
    initial_parameters: Optional[Sequence[float]] = None,
    learning_rate: float = 0.15,
    max_iterations: Optional[int] = None,
    tol: float = 1e-8,
    gradient: str = "shift",
    fd_eps: float = 1e-3,
    optimizer: str = "natural-gradient",
    cobyla_rhobeg: float = 0.5,
    singular_init: bool = True,
    reference_basis_index: Optional[int] = None,
    reference_state: Optional[Sequence[complex]] = None,
    estimator: Any = None,
    energy_fn: Optional[Callable[[np.ndarray], float]] = None,
    callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
    stop_window: Optional[int] = None,
    stop_patience: Optional[int] = None,
    stop_eps: Optional[float] = None,
    grad_tol: Optional[float] = None,
    grad_shifts_per_param: Optional[int] = None,
    max_evaluations: Optional[int] = None,
    record_trace: bool = False,
    record_initial_corner: bool = False,
    early_stop: bool = True,
) -> VQEResult:
    """Natural-gradient ground-state search on a :class:`MinimalCircuit`.

    Minimises ``<psi(theta)| H |psi(theta)>`` using the metric-preconditioned
    (natural) gradient: at each step ``theta <- theta - lr * G^{-1} grad``.

    Parameters
    ----------
    mc:
        The ansatz.
    hamiltonian:
        A Qiskit ``SparsePauliOp`` or an OpenFermion ``QubitOperator`` (JW).
    initial_parameters:
        Starting ``theta``.  If given, it is used directly and overrides
        ``reference_basis_index`` / ``reference_state``.  Otherwise the
        starting point is the chart pre-image of ``reference_state`` (an
        arbitrary statevector, e.g. a uniform Dicke superposition) or
        ``reference_basis_index`` (a single basis state) when supplied;
        failing that, the default (Hartree--Fock chart point if ``mc`` was
        built via :meth:`MinimalCircuit.from_particle_number`, else the chart
        origin).
    learning_rate, max_iterations, tol:
        Optimiser controls.  ``max_iterations`` defaults to a qubit-count
        heuristic; convergence is declared when ``|dE| < tol``.
    gradient:
        ``"shift"`` (exact four-term parameter shift) or ``"fd"`` (finite diff).
    optimizer:
        ``"natural-gradient"`` (default) runs the metric-preconditioned
        gradient descent ``theta <- theta - lr * G^{-1} grad``.
        ``"metric-cobyla"`` runs a derivative-free, metric-preconditioned COBYLA
        instead: at each anchor the Fubini--Study trust region
        ``sum_i g_ii delta_i^2 <= rho^2`` is whitened to an isotropic ball via
        the closed-form diagonal metric (so COBYLA optimises in tangent
        coordinates ``v_i = sqrt(g_ii) delta_i``), the step is retracted
        identity-in-chart (a first-order retraction: the polyspherical angle is
        the great-circle arc length on each nested sphere), and the anchor is
        rebuilt each COBYLA pass (first-order Riemannian trust region).
        Degenerate tangent directions (``g^{-1}_ii = 0`` at a chart corner) are
        frozen at that anchor.  Uses no gradient circuits.
        ``"geodesic-cg"`` runs a Riemannian conjugate gradient on the support
        sphere :math:`S^{M-1}` (the Rayleigh quotient ``E = <psi|H|psi>`` over
        the active determinants).  The search direction is the
        metric-preconditioned (natural) gradient ``G^{-1} grad`` built from the
        closed-form diagonal Fubini--Study metric and the parameter-shift
        Euclidean gradient; consecutive directions are made conjugate by the
        Polak--Ribiere rule with parallel transport of the previous gradient
        and direction along the connecting geodesic.  Each step takes an
        *exact* geodesic (great-circle) line search: along a great circle the
        energy is a single sinusoid ``a0 + a1 cos 2t + b1 sin 2t``, so three
        energy evaluations at ``t in {0, pi/3, 2 pi/3}`` determine the line
        minimum ``t* = (atan2(b1, a1) + pi) / 2`` in closed form.  Geodesics are
        realised by the classical (device-free) polyspherical chart maps
        ``mc.statevector`` / ``mc.parameters``; the only device cost is the
        parameter-shift gradient (``2|theta|`` in the real chart, charged via
        ``grad_shifts_per_param``) plus three line energies per iteration.
        Real chart only (``complex=False``).
        ``"rotosolve"`` runs a derivative-free exact coordinate descent that
        exploits the closed-form trigonometric structure of each chart angle:
        with the others fixed, the energy along one polyspherical angle is an
        exact two-frequency trigonometric polynomial
        ``a0 + a1 cos x + b1 sin x + a2 cos 2x + b2 sin 2x`` (the root angle
        sweeps a great circle -- frequency two only -- and each inner angle a
        small circle -- frequencies one and two).  Five energy evaluations at
        equispaced shifts ``2 pi k / 5`` reconstruct the five coefficients
        exactly, and the per-axis global minimum is the lowest real root of the
        derivative (a quartic in ``exp(i x)``), so each coordinate jumps
        straight to its exact optimum with no step size or line search.  The
        current-iterate energy is reused as the ``x = 0`` sample, so each axis
        costs only four *new* evaluations; the parameters are swept cyclically
        until a sweep no longer improves the energy by more than ``tol``.  Uses
        no gradient circuits and only the chart angles (no statevector).  Real
        chart only (``complex=False``).
    cobyla_rhobeg:
        Initial COBYLA trust radius (in the whitened tangent coordinates) used
        when ``optimizer="metric-cobyla"``.  Defaults to ``0.5``.
    singular_init:
        Apply the analytic corner-escape before iterating (default ``True``).
        Only invoked when the starting point is the Hartree--Fock / origin
        default; an explicit reference already sits in the chart interior.
    reference_basis_index, reference_state:
        Structured starting point.  QAOA / Max-Bisection use a uniform-Dicke
        ``reference_state``; a basis-state ``reference_basis_index`` is a
        convenience for single-determinant references.
    estimator:
        A Qiskit V2 estimator used to evaluate energies.  Defaults to a
        noiseless :class:`~qiskit.primitives.StatevectorEstimator`.
    energy_fn:
        Custom energy callable ``theta -> float`` overriding the estimator path.
    callback:
        Optional ``callback(iteration, energy, theta)`` invoked each step.
    stop_window, stop_patience, stop_eps:
        Enable the unified :class:`WindowedPatienceStopper` early-stopping rule
        (used by the convergence-study benchmark so all methods stop on the
        same criterion, measured in cumulative energy-evaluation units).  When
        ``stop_window`` is given, convergence is declared once the Polyak
        trailing mean over the energies within the last ``stop_window``
        *evaluations* fails to improve by more than ``stop_eps`` across the
        last ``stop_patience`` *evaluations*, and the reported energy is that
        Polyak mean rather than the best raw iterate.  When ``None`` (default)
        the legacy trailing-window-std plateau rule and best-iterate readout
        are used.
    grad_tol:
        When given, the natural-gradient loop terminates as soon as the
        sup-norm of the *raw* energy gradient ``max_i |dE/dtheta_i|`` falls
        below ``grad_tol`` (a first-order stationarity test).  This is the
        noiseless stopping criterion and is independent of the cost-evaluation
        accounting (it observes the gradient that is computed each step
        regardless).  Takes precedence over the windowed-patience / legacy
        plateau rules when set.
    grad_shifts_per_param:
        When given, the reported cost-evaluation count (and the unit fed to the
        stopper / trace / ``max_evaluations`` cap) charges
        ``grad_shifts_per_param`` energy evaluations per parameter-derivative
        instead of the four the exact shift rule physically issues, plus one
        per line/energy evaluation and one for the initial energy.  Use ``2``
        for the real-wavefunction two-term rule (the generator spectrum is
        ``{-1,0,+1}``, identical to a fermionic excitation, so for a real
        target state the four-term rule reduces to two terms).  When ``None``
        (default) the raw oracle-call count is reported.
    max_evaluations:
        Optional hard cap on the cumulative number of energy (cost-function)
        evaluations; the loop stops once this many circuit energy evaluations
        have been consumed, regardless of ``max_iterations``.
    record_trace:
        When ``True``, populate :attr:`VQEResult.trace` with one
        ``{"evals", "energy", "polyak"}`` entry per optimiser step for the
        convergence-versus-cost plots.
    early_stop:
        When ``True`` (default) the active stopping rule (windowed-patience if
        configured, else the legacy plateau test) may terminate the loop before
        ``max_iterations`` / ``max_evaluations``.  Set ``False`` to disable all
        early termination and always run to the evaluation/iteration cap.
    """
    if gradient not in ("shift", "fd"):
        raise ValueError(f"gradient must be 'shift' or 'fd'; got {gradient!r}.")

    if optimizer not in (
        "natural-gradient", "metric-cobyla", "geodesic-cg", "rotosolve"
    ):
        raise ValueError(
            "optimizer must be 'natural-gradient', 'metric-cobyla', "
            f"'geodesic-cg' or 'rotosolve'; got {optimizer!r}."
        )

    if optimizer in ("geodesic-cg", "rotosolve") and mc.complex:
        raise NotImplementedError(
            f"optimizer={optimizer!r} currently supports the real chart only "
            "(omega phases not yet handled)."
        )

    if getattr(mc, "_spin_adapted", False) and optimizer in ("geodesic-cg", "rotosolve"):
        raise NotImplementedError(
            f"optimizer={optimizer!r} optimises on the determinant-support "
            "sphere and would break exact spin adaptation; use the default "
            "'natural-gradient' (or 'metric-cobyla') optimizer."
        )

    if energy_fn is None:
        from qiskit.primitives import StatevectorEstimator

        ham = _to_sparse_pauli_op(hamiltonian, mc.num_qubits)
        est = estimator if estimator is not None else StatevectorEstimator()

        def energy_fn(theta: np.ndarray) -> float:  # type: ignore[misc]
            bound = mc.bound_circuit(theta)
            value = est.run([(bound, ham)]).result()[0].data.evs
            return float(np.real(value))

    # Wrap the energy oracle in a counter so the result can report the number
    # of cost-function evaluations to convergence (gradient shifts + line
    # energies + the initial evaluation), a key resource metric alongside the
    # CNOT count.
    _eval_count = {"n": 0}
    _energy_oracle = energy_fn

    def energy_fn(theta: np.ndarray) -> float:  # type: ignore[misc]
        _eval_count["n"] += 1
        return _energy_oracle(theta)

    # Optional analytic cost accounting.  The exact shift rule physically issues
    # four energy evaluations per parameter-derivative, but for a *real* target
    # wavefunction the generator's ``{-1,0,+1}`` spectrum admits a two-term
    # rule, so the comparison charges ``grad_shifts_per_param`` (e.g. ``2``) per
    # derivative.  ``_charged`` mirrors ``_eval_count`` in this convention and,
    # when active, is the unit reported and fed to the stopper / trace / cap.
    _charge_active = grad_shifts_per_param is not None
    _charged = {"n": 0}

    def _units() -> int:
        return int(_charged["n"]) if _charge_active else int(_eval_count["n"])

    # ----------------------------------------------------------------- Initial point.
    n_params = int(mc.num_parameters)

    def _default_candidate() -> np.ndarray:
        if mc._hf_meta is not None:
            try:
                return mc.hartree_fock_parameters()
            except Exception:
                pass
        return np.zeros(n_params, dtype=float)

    def _reference_candidate(idx: int) -> np.ndarray:
        sv = np.zeros(2 ** mc.num_qubits, dtype=complex)
        sv[int(idx)] = 1.0
        return _statevector_candidate(sv)

    def _statevector_candidate(sv: Sequence[complex]) -> np.ndarray:
        out = mc.parameters(np.asarray(sv, dtype=complex))
        if isinstance(out, tuple):
            theta, omega = out
            return np.concatenate([np.asarray(theta, dtype=float),
                                   np.asarray(omega, dtype=float)])
        return np.asarray(out, dtype=float)

    # ``at_chart_corner`` flags initial points that sit exactly on a chart
    # corner (a single computational-basis state), where the diagonal
    # Fubini--Study metric freezes the deeper tree directions and unescaped
    # natural-gradient descent can only open the tree one layer per iteration.
    # Both the default HF / zeros candidate *and* an explicit single-basis-state
    # ``reference_basis_index`` are such corners and need the analytic escape;
    # a superposition ``reference_state`` and user-supplied ``initial_parameters``
    # already lie in the chart interior.
    at_chart_corner = False
    if initial_parameters is not None:
        params = np.asarray(initial_parameters, dtype=float)
        chose_default = False
    elif n_params > 0 and reference_state is not None:
        # Explicit superposition reference (e.g. a uniform Dicke state).  Start
        # the descent here directly: the chart origin is frequently a
        # stationary single-basis-state local optimum with *lower* initial
        # energy, so it must not be allowed to override the reference.
        params = _statevector_candidate(reference_state)
        chose_default = False
    elif n_params > 0 and reference_basis_index is not None:
        params = _reference_candidate(int(reference_basis_index))
        chose_default = False
        at_chart_corner = True
    else:
        params = _default_candidate()
        chose_default = True

    # The singular-coordinate corner-escape is well-defined at the chart corners
    # reached by the HF / zeros candidate and by a single-basis-state
    # ``reference_basis_index``; an explicit superposition reference already sits
    # in the chart interior, where unprojected NG-descent is appropriate.
    # When ``record_initial_corner`` is set we evaluate (and later record) the
    # bare corner energy *before* the analytic escape, so the reported
    # trajectory honestly begins at the initialisation point (e.g. the
    # Hartree-Fock determinant) rather than at the post-escape interior state;
    # the escape itself remains an uncharged analytic first step.
    corner_energy: Optional[float] = None
    if (record_initial_corner and singular_init and n_params > 0
            and (chose_default or at_chart_corner)):
        corner_energy = float(energy_fn(params))
    if singular_init and (chose_default or at_chart_corner) and n_params > 0:
        params = mc.singular_initialise(params, energy_fn, lr=learning_rate)

    if max_iterations is None:
        max_iterations = _default_max_iterations(mc.num_qubits)

    # Unified windowed-patience stopping rule (opt-in).  When enabled it
    # replaces both the legacy trailing-std plateau test and the best-iterate
    # readout: convergence is the Polyak-mean stall and the reported energy is
    # that Polyak mean.  Disabled (``None``) preserves the legacy behaviour.
    use_patience = stop_window is not None
    stopper = (
        WindowedPatienceStopper(
            window=int(stop_window),
            patience=int(stop_patience if stop_patience is not None else stop_window),
            eps=float(stop_eps if stop_eps is not None else tol),
        )
        if use_patience
        else None
    )

    e_prev = energy_fn(params)
    # Initial-energy charge (one evaluation) in the analytic convention; the
    # one-off singular-init corner-escape probes are not charged, matching the
    # UCCSD baseline's "1 initial energy" accounting.
    _charged["n"] = 1
    history = [float(e_prev)]
    iterations = 0
    converged = False
    # Track best iterate seen so far. Under non-trivial landscapes the final
    # iterate is not necessarily the right thing to report; we always return
    # the best plateau iterate instead.
    best_e = float(e_prev)
    best_params = params.copy()
    plateau_window = 10

    trace: List[Dict[str, float]] = []
    if stopper is not None:
        ebar0 = stopper.update(_units(), float(e_prev))
    else:
        ebar0 = float(e_prev)
    if record_trace:
        if corner_energy is not None:
            trace.append({"evals": _units(),
                          "energy": float(corner_energy),
                          "polyak": float(corner_energy)})
        trace.append({"evals": _units(),
                      "energy": float(e_prev), "polyak": float(ebar0)})

    if optimizer == "metric-cobyla":
        # ------------------------------------------------------------------
        # Metric-preconditioned, derivative-free COBYLA (first-order
        # Riemannian trust region in the polyspherical chart).
        #
        # At an anchor ``a`` the diagonal Fubini--Study metric whitens the
        # trust region: with ``v_i = sqrt(g_ii(a)) * delta_i`` COBYLA sees an
        # isotropic ball.  We expose ``theta = a + diag(sqrt(g^{-1}_ii(a))) v``
        # to COBYLA (``sqrt(g^{-1}_ii) = 1/sqrt(g_ii)``), reusing the package's
        # inverse-metric diagonal so the existing corner ``epsilon``-freeze
        # carries over for free: directions with ``g^{-1}_ii = 0`` get scale 0
        # and are dropped from the COBYLA variable at that anchor.  The step is
        # retracted identity-in-chart (first-order); the anchor and the
        # whitening are rebuilt after each COBYLA pass.
        # ------------------------------------------------------------------
        from scipy.optimize import minimize as _scipy_minimize

        class _StopMetricCobyla(Exception):
            pass

        freeze_eps = 1e-6  # matches the inverse-metric corner threshold
        anchor = params.copy()
        e_anchor = float(e_prev)

        def _observe(theta: np.ndarray) -> float:
            """Evaluate energy and update history / best / stopper / trace."""
            nonlocal best_e, best_params, iterations
            e = energy_fn(theta)
            iterations += 1
            history.append(float(e))
            if np.isfinite(e) and float(e) < best_e:
                best_e = float(e)
                best_params = theta.copy()
            ebar = (stopper.update(int(_eval_count["n"]), float(e))
                    if stopper is not None else float(e))
            if record_trace:
                trace.append({"evals": int(_eval_count["n"]),
                              "energy": float(e), "polyak": float(ebar)})
            if early_stop and stopper is not None and stopper.should_stop():
                raise _StopMetricCobyla("converged")
            if max_evaluations is not None and _eval_count["n"] >= int(max_evaluations):
                raise _StopMetricCobyla("budget")
            return float(e)

        # Trust-region restart schedule.  Each re-anchor rebuilds the whitening
        # from the (position-dependent) metric and shrinks the trust radius
        # geometrically, so refinement comes from re-anchoring -- not from a
        # single inner COBYLA grinding rhobeg down to machine precision and then
        # being re-inflated on the next pass (which wastes evaluations and
        # oscillates).  The inner rhoend is kept loose (a fraction of rhobeg);
        # the outer loop tightens.
        rhobeg = float(cobyla_rhobeg)
        rho_shrink = 0.35
        rho_floor = 1e-4

        try:
            while True:
                # Open any true-corner degenerate subtrees analytically before
                # whitening (subtree-scoped: for a single active leaf the
                # "subtree" is the whole tree).  The metric whitening freezes a
                # direction whenever an ancestor sits exactly on a chart corner
                # (g_ii = product of cos^2/sin^2 of ancestors vanishes); rather
                # than let COBYLA rediscover those directions one tree layer per
                # restart (a slow wavefront), the analytic corner-escape opens
                # each such subtree in a single top-down sweep.  At an interior
                # anchor the corner detector (tau) no-ops with no extra evals, so
                # this mainly fires at the cold start -- in particular the SAE
                # Hartree-Fock leaf, where an explicit ``reference_basis_index``
                # skipped the shared cold-start opener.  Its shift probes go
                # through ``energy_fn`` and are charged to the budget like any
                # other evaluation.
                inv_diag = np.diag(np.asarray(mc.inverse_metric(anchor), dtype=float))
                scale = np.sqrt(np.clip(inv_diag, 0.0, None))  # sqrt(g^{-1}_ii)
                if np.count_nonzero(scale > freeze_eps) < scale.size:
                    opened = np.asarray(
                        mc.singular_initialise(anchor, energy_fn, lr=learning_rate),
                        dtype=float)
                    if not np.allclose(opened, anchor):
                        anchor = opened
                        e_anchor = _observe(anchor)
                        inv_diag = np.diag(np.asarray(mc.inverse_metric(anchor), dtype=float))
                        scale = np.sqrt(np.clip(inv_diag, 0.0, None))
                active = np.flatnonzero(scale > freeze_eps)
                if active.size == 0:
                    break  # fully degenerate anchor: nothing left to open or move

                base = anchor          # captured per pass (used synchronously)
                sc = scale

                def _obj(v: np.ndarray) -> float:
                    theta = base.copy()
                    theta[active] = base[active] + sc[active] * np.asarray(v, dtype=float)
                    return _observe(theta)

                opt = _scipy_minimize(
                    _obj, np.zeros(active.size, dtype=float), method="COBYLA",
                    options={"maxiter": int(max_iterations),
                             "rhobeg": float(rhobeg),
                             "tol": float(max(rhobeg * 1e-2, rho_floor))},
                )

                new_anchor = anchor.copy()
                new_anchor[active] = anchor[active] + scale[active] * np.asarray(opt.x, dtype=float)
                e_new_anchor = float(opt.fun)  # already counted inside _obj
                improved = (e_anchor - e_new_anchor) > float(tol)
                anchor = new_anchor
                e_anchor = e_new_anchor
                if early_stop and stopper is not None and stopper.should_stop():
                    converged = True
                    break
                if max_evaluations is not None and _eval_count["n"] >= int(max_evaluations):
                    break
                # Refinement comes from re-anchoring with a geometrically
                # shrinking trust region.  Once a whole pass (a fresh
                # metric-whitened trust region of the current radius) fails to
                # improve the energy by ``tol``, neither a smaller radius nor a
                # re-anchor at the same point can help, so we are converged --
                # stop immediately rather than grinding ``rhobeg`` to the floor
                # (which only burns evaluations confirming a plateau already
                # reached).
                if not improved:
                    converged = True
                    break
                rhobeg = max(rhobeg * rho_shrink, rho_floor)
        except _StopMetricCobyla as exc:
            converged = (str(exc) == "converged")

        if stopper is not None:
            reported_energy = stopper.reported_energy
            polyak_energy = float(reported_energy)
            polyak_window_used = int(stopper.window)
        else:
            reported_energy = float(best_e)
            polyak_energy = None
            polyak_window_used = 0

        return VQEResult(
            energy=float(reported_energy),
            optimal_parameters=best_params,
            history=history,
            iterations=iterations,
            converged=converged,
            num_parameters=mc.num_parameters,
            num_energy_evaluations=int(_eval_count["n"]),
            trace=trace,
            polyak_energy=polyak_energy,
            polyak_window=polyak_window_used,
        )

    if optimizer == "geodesic-cg":
        # ------------------------------------------------------------------
        # Riemannian conjugate gradient on the support sphere S^{M-1}.
        #
        # The objective is the Rayleigh quotient E = <psi|H|psi> over the unit
        # sphere of real amplitudes on the active determinants (``mc.support``).
        # The search direction is the metric-preconditioned (natural) gradient
        # ``G^{-1} grad`` -- here ``G`` is the closed-form diagonal Fubini-Study
        # metric and ``grad`` the parameter-shift Euclidean gradient -- pushed
        # forward through the chart to the tangent space at ``psi``; on the
        # round sphere this coincides with the Riemannian gradient
        # ``2 (H psi - E psi)``.  Successive directions are made conjugate by the
        # Polak-Ribiere rule with parallel transport along the connecting
        # geodesic.  Each step is an *exact* great-circle line search: along
        # ``psi(t) = cos t psi + sin t v`` the energy is a single sinusoid in
        # ``2t``, so three energy evaluations pin down the closed-form minimum.
        # Geodesics are mapped to/from chart angles by the device-free
        # ``mc.statevector`` / ``mc.parameters`` maps; the only device cost is
        # the gradient (``2|theta|`` real-chart shifts, charged via
        # ``grad_shifts_per_param``) plus three line energies per iteration.
        # ------------------------------------------------------------------
        class _StopGeoCG(Exception):
            pass

        support_idx = np.asarray(mc.support, dtype=int)
        full_dim = 2 ** mc.num_qubits

        def _psi_support(theta: np.ndarray) -> np.ndarray:
            """Unit real support amplitudes of |psi(theta)> (device-free)."""
            return np.real(np.asarray(mc.statevector(theta)))[support_idx]

        def _theta_of(psi_vec: np.ndarray) -> np.ndarray:
            """Chart angles for a support amplitude vector (device-free)."""
            full = np.zeros(full_dim, dtype=complex)
            full[support_idx] = psi_vec
            out = mc.parameters(full)
            return np.asarray(out, dtype=float)

        def _chart_jacobian(theta: np.ndarray, base: np.ndarray,
                            h: float = 1e-6) -> np.ndarray:
            """Forward-difference d psi_support / d theta (device-free)."""
            J = np.empty((support_idx.size, theta.size), dtype=float)
            for i in range(theta.size):
                tp = theta.copy()
                tp[i] += h
                J[:, i] = (_psi_support(tp) - base) / h
            return J

        def _natural_tangent(theta: np.ndarray):
            """Natural-gradient direction pushed to the tangent at psi."""
            grad = mc.gradient(energy_fn, theta, method=gradient, eps=fd_eps)
            if _charge_active:
                shifts = int(grad_shifts_per_param) if gradient == "shift" else 2
                _charged["n"] += shifts * int(n_params)
            inv_metric = np.asarray(mc.inverse_metric(theta), dtype=float)
            if inv_metric.ndim == 1:
                inv_metric = np.diag(inv_metric)
            inv_metric = np.nan_to_num(inv_metric, nan=0.0, posinf=0.0, neginf=0.0)
            base = _psi_support(theta)
            jac = _chart_jacobian(theta, base)
            u = jac @ (inv_metric @ grad)
            u -= (u @ base) * base  # project onto the tangent space at psi
            return u, base

        def _transport(w: np.ndarray, psi: np.ndarray, v: np.ndarray,
                       t: float) -> np.ndarray:
            """Parallel transport of tangent ``w`` along the geodesic ``v`` by ``t``."""
            return w + (w @ v) * ((np.cos(t) - 1.0) * v - np.sin(t) * psi)

        def _line_min(psi: np.ndarray, direction: np.ndarray):
            """Exact great-circle line search; returns (psi_new, t*, v, E_min)."""
            v = direction / np.linalg.norm(direction)
            ts = (0.0, np.pi / 3.0, 2.0 * np.pi / 3.0)
            energies = np.empty(3, dtype=float)
            for k, t in enumerate(ts):
                gamma = np.cos(t) * psi + np.sin(t) * v
                e = energy_fn(_theta_of(gamma))
                if _charge_active:
                    _charged["n"] += 1
                energies[k] = e
            design = np.array([[1.0, np.cos(2 * t), np.sin(2 * t)] for t in ts])
            a0, a1, b1 = np.linalg.solve(design, energies)
            phi = np.arctan2(b1, a1)
            t_star = (phi + np.pi) / 2.0
            psi_new = np.cos(t_star) * psi + np.sin(t_star) * v
            psi_new /= np.linalg.norm(psi_new)
            return psi_new, float(t_star), v, float(a0 - np.hypot(a1, b1))

        def _observe(theta: np.ndarray, e: float) -> None:
            """Record an accepted iterate (history / best / stopper / trace)."""
            nonlocal best_e, best_params, iterations
            iterations += 1
            history.append(float(e))
            if np.isfinite(e) and float(e) < best_e:
                best_e = float(e)
                best_params = theta.copy()
            ebar = (stopper.update(_units(), float(e))
                    if stopper is not None else float(e))
            if record_trace:
                trace.append({"evals": _units(),
                              "energy": float(e), "polyak": float(ebar)})
            if early_stop and stopper is not None and stopper.should_stop():
                raise _StopGeoCG("converged")
            if max_evaluations is not None and _units() >= int(max_evaluations):
                raise _StopGeoCG("budget")

        # First-order stationarity floor: the Riemannian-gradient norm below
        # which the descent is declared converged when no windowed-patience
        # stopper / gradient tolerance is configured.
        gtol = float(grad_tol) if grad_tol is not None else 1e-10

        if n_params > 0:
            theta = params.copy()
            grad_dir, psi = _natural_tangent(theta)
            search = -grad_dir
            try:
                for _ in range(max_iterations):
                    if float(np.linalg.norm(grad_dir)) < gtol:
                        converged = True
                        break
                    psi_new, t_star, v, e_min = _line_min(psi, search)
                    theta = _theta_of(psi_new)
                    # The sinusoid minimum is exact, so report it as the iterate
                    # energy rather than spending a fourth circuit evaluation.
                    _observe(theta, e_min)
                    grad_new, _ = _natural_tangent(theta)
                    grad_T = _transport(grad_dir, psi, v, t_star)
                    dir_T = _transport(search, psi, v, t_star)
                    denom = float(grad_dir @ grad_dir)
                    beta = (max(0.0, float(grad_new @ (grad_new - grad_T)) / denom)
                            if denom > 0.0 else 0.0)
                    search = -grad_new + beta * dir_T
                    if float(search @ grad_new) > 0.0:  # not a descent dir: reset
                        search = -grad_new
                    grad_dir, psi = grad_new, psi_new
                    if max_evaluations is not None and _units() >= int(max_evaluations):
                        break
            except _StopGeoCG as exc:
                converged = (str(exc) == "converged")

        if stopper is not None:
            reported_energy = stopper.reported_energy
            polyak_energy = float(reported_energy)
            polyak_window_used = int(stopper.window)
        else:
            reported_energy = float(best_e)
            polyak_energy = None
            polyak_window_used = 0

        return VQEResult(
            energy=float(reported_energy),
            optimal_parameters=best_params,
            history=history,
            iterations=iterations,
            converged=converged,
            num_parameters=mc.num_parameters,
            num_energy_evaluations=_units(),
            trace=trace,
            polyak_energy=polyak_energy,
            polyak_window=polyak_window_used,
        )

    if optimizer == "rotosolve":
        # ------------------------------------------------------------------
        # Derivative-free exact coordinate descent (Rotosolve) in the chart.
        #
        # Along a single polyspherical angle ``x = theta_i`` (others fixed) the
        # energy is an exact two-frequency trigonometric polynomial
        #     E(x) = a0 + a1 cos x + b1 sin x + a2 cos 2x + b2 sin 2x ,
        # because varying ``theta_i`` rotates |psi> within a fixed 2-plane (a
        # small circle on the support sphere; a great circle for the root
        # angle, where the frequency-one terms vanish).  Five energy samples at
        # equispaced shifts ``t_k = 2 pi k / 5`` determine the five
        # coefficients exactly; the per-axis minimiser is the lowest real root
        # of ``E'(x) = 0`` -- a quartic in ``z = exp(i x)`` -- so each
        # coordinate jumps to its exact optimum.  The current energy is reused
        # as the ``x = 0`` sample, so each axis costs four *new* evaluations.
        # Pure chart method: no gradient circuits, no statevector.
        # ------------------------------------------------------------------
        class _StopRoto(Exception):
            pass

        # Five equispaced reconstruction nodes (relative angle shifts).  The
        # +-pi/2 two-term rule is singular for the frequency-two component, so
        # five distinct nodes are required.
        _nodes = (np.arange(5) * (2.0 * np.pi / 5.0))
        _design5 = np.array(
            [[1.0, np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)]
             for t in _nodes]
        )
        _design5_inv = np.linalg.inv(_design5)

        def _axis_min(coef: np.ndarray, e_center: float):
            """Exact (offset, value) minimiser of the 2-frequency axis model."""
            a0, a1, b1, a2, b2 = coef
            # E'(x) = b cos x + c sin x + d cos 2x + e sin 2x, with
            b, c, d, e = b1, -a1, 2.0 * b2, -2.0 * a2
            lead = 0.5 * d - 0.5j * e
            cand = [0.0]  # x = 0 (the current point) is always a candidate
            if abs(lead) > 1e-14:
                # z^2 E'(x) = lead z^4 + (b/2 - i c/2) z^3 + (b/2 + i c/2) z
                #            + (d/2 + i e/2)
                poly = np.array([
                    lead,
                    0.5 * b - 0.5j * c,
                    0.0 + 0.0j,
                    0.5 * b + 0.5j * c,
                    0.5 * d + 0.5j * e,
                ])
                roots = np.roots(poly)
                on_circle = roots[np.abs(np.abs(roots) - 1.0) < 1e-6]
                cand.extend(np.angle(on_circle).tolist())
            def _model(x: float) -> float:
                return float(a0 + a1 * np.cos(x) + b1 * np.sin(x)
                             + a2 * np.cos(2 * x) + b2 * np.sin(2 * x))
            best_x, best_v = 0.0, e_center
            for x in cand:
                v = _model(x)
                if v < best_v:
                    best_x, best_v = float(x), v
            return best_x, best_v

        def _observe(theta: np.ndarray, e: float) -> None:
            """Record an accepted iterate (history / best / stopper / trace)."""
            nonlocal best_e, best_params, iterations
            iterations += 1
            history.append(float(e))
            if np.isfinite(e) and float(e) < best_e:
                best_e = float(e)
                best_params = theta.copy()
            ebar = (stopper.update(_units(), float(e))
                    if stopper is not None else float(e))
            if record_trace:
                trace.append({"evals": _units(),
                              "energy": float(e), "polyak": float(ebar)})
            if early_stop and stopper is not None and stopper.should_stop():
                raise _StopRoto("converged")
            if max_evaluations is not None and _units() >= int(max_evaluations):
                raise _StopRoto("budget")

        if n_params > 0:
            theta = params.copy()
            e_cur = float(e_prev)  # reused as the x = 0 sample of the first axis
            try:
                for _ in range(max_iterations):
                    e_sweep_start = e_cur
                    for i in range(n_params):
                        samples = np.empty(5, dtype=float)
                        samples[0] = e_cur  # x = 0 reuse (no new evaluation)
                        probe = theta.copy()
                        for k in range(1, 5):
                            probe[i] = theta[i] + _nodes[k]
                            samples[k] = energy_fn(probe)
                            if _charge_active:
                                _charged["n"] += 1
                        coef = _design5_inv @ samples
                        offset, e_new = _axis_min(coef, e_cur)
                        if offset != 0.0:
                            theta[i] += offset
                            e_cur = e_new
                        _observe(theta, e_cur)
                    if (e_sweep_start - e_cur) < tol:
                        converged = True
                        break
            except _StopRoto as exc:
                converged = (str(exc) == "converged")

        if stopper is not None:
            reported_energy = stopper.reported_energy
            polyak_energy = float(reported_energy)
            polyak_window_used = int(stopper.window)
        else:
            reported_energy = float(best_e)
            polyak_energy = None
            polyak_window_used = 0

        return VQEResult(
            energy=float(reported_energy),
            optimal_parameters=best_params,
            history=history,
            iterations=iterations,
            converged=converged,
            num_parameters=mc.num_parameters,
            num_energy_evaluations=_units(),
            trace=trace,
            polyak_energy=polyak_energy,
            polyak_window=polyak_window_used,
        )

    for it in range(max_iterations):
        iterations = it + 1
        grad = mc.gradient(energy_fn, params, method=gradient, eps=fd_eps)
        # Analytic gradient charge: `grad_shifts_per_param` per differentiated
        # parameter (real-wavefunction two-term rule), decoupled from the four
        # the exact shift rule physically issues.
        if _charge_active:
            shifts = int(grad_shifts_per_param) if gradient == "shift" else 2
            _charged["n"] += shifts * int(n_params)
        # First-order stationarity stop (noiseless): the raw-gradient sup-norm.
        if grad_tol is not None and early_stop:
            if float(np.max(np.abs(grad))) < float(grad_tol):
                converged = True
                break
        inv_metric = mc.inverse_metric(params)
        # Chart corners drive 1/w_i to ill-defined values; sanitise the
        # inverse metric and clip the NG step so a single bad iterate cannot
        # propel us across the chart.
        inv_metric = np.nan_to_num(inv_metric, nan=0.0, posinf=1e6, neginf=-1e6)
        nat_grad = np.clip(inv_metric @ grad, -5.0, 5.0)
        params = params - learning_rate * nat_grad

        e_new = energy_fn(params)
        if _charge_active:
            _charged["n"] += 1
        history.append(float(e_new))
        if callback is not None:
            callback(iterations, float(e_new), params)
        if not np.isfinite(e_new):
            break
        if float(e_new) < best_e:
            best_e = float(e_new)
            best_params = params.copy()

        ebar = (stopper.update(_units(), float(e_new))
                if stopper is not None else float(e_new))
        if record_trace:
            trace.append({"evals": _units(),
                          "energy": float(e_new), "polyak": float(ebar)})

        if stopper is not None:
            # Unified windowed-patience stop; the optimal parameters are the
            # current iterate (the Polyak mean is a readout, not a point).
            # When ``grad_tol`` is set it is authoritative (checked at the top
            # of the loop), so the windowed-patience test is suppressed and the
            # stopper is used only for its Polyak-mean energy readout.
            if early_stop and grad_tol is None and stopper.should_stop():
                converged = True
                best_e = float(e_new)
                best_params = params.copy()
                e_prev = e_new
                break
        elif grad_tol is not None:
            # Gradient-norm stopping is authoritative (checked at the top of the
            # loop on the freshly computed gradient); suppress the legacy
            # plateau test so the stop is purely the first-order criterion.
            pass
        elif early_stop:
            # Legacy plateau stopping: declare convergence only when the
            # trailing window has stabilised below `tol`.
            if len(history) >= plateau_window:
                window = np.asarray(history[-plateau_window:], dtype=float)
                if float(np.std(window, ddof=1)) < tol:
                    converged = True
                    break
        e_prev = e_new

        # Hard cap on cumulative cost-function (energy) evaluations.
        if max_evaluations is not None and _units() >= int(max_evaluations):
            break

    if stopper is not None:
        reported_energy = stopper.reported_energy
        polyak_energy: Optional[float] = float(reported_energy)
        polyak_window_used = int(stopper.window)
    else:
        reported_energy = float(best_e)
        polyak_energy = None
        polyak_window_used = 0

    return VQEResult(
        energy=float(reported_energy),
        optimal_parameters=best_params,
        history=history,
        iterations=iterations,
        converged=converged,
        num_parameters=mc.num_parameters,
        num_energy_evaluations=_units(),
        trace=trace,
        polyak_energy=polyak_energy,
        polyak_window=polyak_window_used,
    )
