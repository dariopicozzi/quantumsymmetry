"""Exact total-spin adaptation of the pruned tree, with no change-of-basis gate.

Particle number, spin projection and point-group symmetry act diagonally on the
computational basis, so :class:`~quantumsymmetry.treecircuit.MinimalCircuit`
enforces them exactly by choosing the support set.  Total spin is the
exception: its eigenstates, the configuration state functions (CSFs), are
entangled combinations of determinants, so a determinant support is generally
reducible and the plain tree reaches a definite-spin state only variationally.

This module provides the gate-free spin adaptation of the tree ansatz: a state
of definite total spin ``S`` is prepared *exactly*, on the unchanged pruned
determinant-tree circuit, with the closed-form diagonal metric intact.  The
user-facing entry point is the ``total_spin`` keyword of
:meth:`MinimalCircuit.from_particle_number`::

    mc = MinimalCircuit.from_particle_number(3, (2, 2), total_spin=0)
    result = minimize_energy(mc, hamiltonian)      # <S^2> = 0 at every step

The returned object varies a binary tree over the spin-``S`` CSFs (the *CSF
chart*, whose Fubini--Study metric is exactly diagonal because the
determinant-to-CSF change of basis is a fixed unitary), and realises every
state on the plain determinant tree; the change of basis is used only
classically and never appears as a gate.  The construction targets the highest
weight ``Sz = S``, so ``num_particles`` must satisfy ``n_alpha - n_beta =
2*total_spin``; the lower-weight members of the multiplet follow by the fixed
lowering operator and need no separate ansatz.

The same spin manifold admits a dual parametrisation directly in the
determinant-tree angles: the *ballot leaders* (determinants whose open-shell
spin word has all partial sums nonnegative) span a backbone of free nodes,
and the remaining angles are *tied* to closed-form functions of them, with the
metric acquiring a rank-``r`` Woodbury correction.  The utilities for that
route -- the free/tied node split, the tie solver, the tie Jacobian and the
Woodbury inverse metric -- are exposed here as functions for users who want
the optimisation loop to live natively in the gate angles.

Basis convention: leaf integer = Qiskit statevector index, little-endian over
blocked Jordan--Wigner modes (``alpha_p`` = bit ``p``, ``beta_p`` = bit
``m + p``), matching both the circuit's leaf indexing and Qiskit Nature's
blocked ``JordanWignerMapper``.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import numpy.linalg as la

from .metric import (
    make_cartesian_to_polyspherical,
    make_inverse_metric,
    make_polyspherical_to_cartesian,
)
from .optimize import singular_initialise as _singular_initialise
from .tree import chart_topology
from .minimal_circuit import MinimalCircuit

__all__ = [
    "sector_csf_unitary",
    "spin_block_columns",
    "ballot_leaders",
    "ballot_embedding",
    "ballot_free_tied",
    "spin_constraint_matrix",
    "sector_amplitude_map",
    "constraint_values",
    "constraint_jacobian",
    "solve_tied",
    "tie_jacobian",
    "woodbury_inverse_metric",
    "total_spin_expectation",
]

_half = Fraction(1, 2)


# ---------------------------------------------------------------------------
# Genealogical spin-coupling kernel (sequential Clebsch-Gordan).
#
# Conventions: single spin |0> = up (m = +1/2), |1> = down; spin k is appended
# as the least significant bit at coupling step k, so spin 1 is the MSB of the
# t-bit uncoupled index.  A coupling path is (S_1, ..., S_t) with S_1 = 1/2 and
# steps of +/- 1/2 staying nonnegative.
# ---------------------------------------------------------------------------

def _enumerate_paths(t: int) -> List[tuple]:
    paths = [(_half,)]
    for _ in range(t - 1):
        nxt = []
        for p in paths:
            for step in (_half, -_half):
                s = p[-1] + step
                if s >= 0:
                    nxt.append(p + (s,))
        paths = nxt
    return paths


def _cg_step_up(j: Fraction, M: Fraction) -> Tuple[float, float]:
    denom = float(2 * j + 1)
    return (np.sqrt(float(j + M + _half) / denom),
            np.sqrt(float(j - M + _half) / denom))


def _cg_step_down(j: Fraction, M: Fraction) -> Tuple[float, float]:
    denom = float(2 * j + 1)
    return (np.sqrt(float(j - M + _half) / denom),
            np.sqrt(float(j + M + _half) / denom))


def _coupled_vectors_for_path(path: tuple) -> dict:
    """{M: vector in the 2^t uncoupled basis} for the top spin S = path[-1]."""
    t = len(path)
    cur = {_half: np.array([1.0, 0.0]), -_half: np.array([0.0, 1.0])}
    for k in range(2, t + 1):
        j, S = path[k - 2], path[k - 1]
        nxt = {}
        M = -S
        while M <= S:
            vec = np.zeros(2 ** k)
            up_src = cur.get(M - _half)
            dn_src = cur.get(M + _half)
            if S == j + _half:
                c, s = _cg_step_up(j, M)
                if up_src is not None:
                    vec[0::2] += c * up_src
                if dn_src is not None:
                    vec[1::2] += s * dn_src
            else:
                c, s = _cg_step_down(j, M)
                if up_src is not None:
                    vec[0::2] += -c * up_src
                if dn_src is not None:
                    vec[1::2] += s * dn_src
            nxt[M] = vec
            M += 1
        cur = nxt
    return cur


def _csfs_for_block(t: int, M: Fraction) -> List[tuple]:
    """CSFs of ``t`` spins with projection ``M`` as (S, path, vector), ordered
    by total spin S ascending then by coupling path."""
    out = []
    for path in sorted(_enumerate_paths(t), key=lambda p: (p[-1], p)):
        S = path[-1]
        if abs(M) > S:
            continue
        out.append((S, path, _coupled_vectors_for_path(path)[M]))
    return out


# ---------------------------------------------------------------------------
# Sector determinants and the sector CSF unitary.
# ---------------------------------------------------------------------------

def leaf_int(m: int, occ_a: Sequence[int], occ_b: Sequence[int]) -> int:
    """Leaf integer of the determinant with alpha orbitals ``occ_a`` and beta
    orbitals ``occ_b`` (blocked Jordan--Wigner, little-endian)."""
    return sum(1 << p for p in occ_a) + sum(1 << (m + p) for p in occ_b)


def sector_leaves(m: int, n_alpha: int, n_beta: int) -> List[int]:
    """All ``{N, Sz}`` sector determinant leaf integers (unsorted)."""
    return [leaf_int(m, oa, ob)
            for oa in combinations(range(m), n_alpha)
            for ob in combinations(range(m), n_beta)]


def _pattern_of(m: int, occ_a, occ_b):
    """(docc, open_orbitals, open_spin_bits): open-shell spins in ascending
    orbital order, bit 0 = alpha (up), 1 = beta (down)."""
    sa, sb = set(occ_a), set(occ_b)
    docc = tuple(sorted(sa & sb))
    open_orbs = tuple(sorted(sa ^ sb))
    spins = tuple(0 if p in sa else 1 for p in open_orbs)
    return docc, open_orbs, spins


def _blocked_to_interleaved_parity(occ_a, occ_b) -> float:
    """Sign relating the blocked-JW determinant to the interleaved normal order
    the spin-coupling kernel assumes (same-orbital spin flips carry no JW
    string there, so the naive Clebsch--Gordan amplitudes are exact)."""
    blocked = [(0, p) for p in sorted(occ_a)] + [(1, p) for p in sorted(occ_b)]
    target = sorted(blocked, key=lambda sp: (sp[1], sp[0]))
    perm = [blocked.index(x) for x in target]
    sign, seen = 1.0, [False] * len(perm)
    for i in range(len(perm)):
        if seen[i]:
            continue
        j, clen = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            clen += 1
        if clen % 2 == 0:
            sign = -sign
    return sign


def _blocked_leaf_occ(leaf: int, m: int):
    """``(occ_a, occ_b)`` spatial-orbital tuples of a blocked-JW determinant."""
    occ_a = tuple(p for p in range(m) if (leaf >> p) & 1)
    occ_b = tuple(p for p in range(m) if (leaf >> (m + p)) & 1)
    return occ_a, occ_b


def _interleaved_leaf_occ(leaf: int, m: int):
    """``(occ_a, occ_b)`` of an interleaved-JW determinant (alpha=2p, beta=2p+1),
    the convention of the symmetry-adapted encoding's ``sae_to_jw``."""
    occ_a = tuple(p for p in range(m) if (leaf >> (2 * p)) & 1)
    occ_b = tuple(p for p in range(m) if (leaf >> (2 * p + 1)) & 1)
    return occ_a, occ_b


def encode_blocked_leaves(encoding, leaves, m):
    """Map blocked-JW determinant leaves to their symmetry-adapted-encoding
    leaves, with the sign that keeps a spin eigenstate a spin eigenstate.

    The encoding's Jordan--Wigner<->encoded map (``jw_to_sae``) is a sign-free
    GF(2) bijection on the sector; the only sign is the blocked->interleaved
    fermionic reordering parity (``_blocked_to_interleaved_parity``), since the
    encoding works in the interleaved (alpha=2p, beta=2p+1) convention while the
    CSF machinery here is blocked.  Folding that parity into the encoded
    amplitude reproduces the state the encoding assigns to the determinant.
    Returns ``(encoded_leaves, signs)``.
    """
    enc_leaves, signs = [], []
    for l in leaves:
        occ_a, occ_b = _blocked_leaf_occ(int(l), m)
        a_inter = (sum(1 << (2 * p) for p in occ_a)
                   + sum(1 << (2 * p + 1) for p in occ_b))
        enc_leaves.append(int(encoding.jw_to_sae(a_inter)))
        signs.append(_blocked_to_interleaved_parity(occ_a, occ_b))
    return enc_leaves, np.asarray(signs, dtype=float)


def sector_csf_unitary(m: int, n_alpha: int, n_beta: int,
                       allowed_leaves: Optional[Sequence[int]] = None):
    """Build the fixed determinant-to-CSF change of basis of a molecular sector.

    The unitary is block-diagonal over occupation patterns: a determinant
    factorises into doubly occupied / empty spectator orbitals and ``t`` open
    shells carrying spin-1/2's, and within one pattern the CSFs are the
    genealogically coupled states of those ``t`` spins, times the per-
    determinant Jordan--Wigner reordering parity.

    Parameters
    ----------
    m, n_alpha, n_beta:
        Spatial-orbital count and the ``(n_alpha, n_beta)`` sector.
    allowed_leaves:
        Optional restriction of the determinant basis, e.g. a point-group
        irrep from :func:`sieve_states_by_symmetry`.  Point-group screening
        keeps whole occupation-pattern blocks (the spatial irrep depends only
        on the pattern); a partially covered block raises.

    Returns
    -------
    (leaves, U, labels):
        ``leaves`` -- the sorted ``k`` sector determinant leaf integers;
        ``U`` -- ``(k, k)`` real orthogonal, column ``j`` the CSF expressed
        over ``leaves``; ``labels[j] = (S, docc, open_orbs, path)``.
    """
    Mz = Fraction(n_alpha - n_beta, 2)
    allowed = None if allowed_leaves is None else {int(x) for x in allowed_leaves}

    blocks: dict = {}
    parities: dict = {}
    for oa in combinations(range(m), n_alpha):
        for ob in combinations(range(m), n_beta):
            docc, open_orbs, spins = _pattern_of(m, oa, ob)
            l = leaf_int(m, oa, ob)
            parities[l] = _blocked_to_interleaved_parity(oa, ob)
            blocks.setdefault((docc, open_orbs), []).append((l, spins))

    if allowed is not None:
        kept = {}
        for key, dets in blocks.items():
            in_set = [d for d in dets if d[0] in allowed]
            if not in_set:
                continue
            if len(in_set) != len(dets):
                raise ValueError(
                    "allowed_leaves splits an occupation-pattern block "
                    f"{key}: {len(in_set)}/{len(dets)} determinants kept; "
                    "point-group screening should keep whole blocks.")
            kept[key] = dets
        blocks = kept

    leaves = sorted(l for blk in blocks.values() for l, _ in blk)
    pos = {l: j for j, l in enumerate(leaves)}
    k = len(leaves)

    U = np.zeros((k, k))
    labels: List[tuple] = [None] * k

    for (docc, open_orbs), dets in blocks.items():
        dets.sort()
        t = len(open_orbs)
        if t == 0:
            (l, _), = dets           # closed shell: its own CSF (S = 0)
            U[pos[l], pos[l]] = parities[l]
            labels[pos[l]] = (Fraction(0), docc, open_orbs, ())
            continue
        det_sidx = {sum(s << (t - 1 - i) for i, s in enumerate(spins)): l
                    for l, spins in dets}
        csfs = _csfs_for_block(t, Mz)
        assert len(csfs) == len(dets), (docc, open_orbs, len(csfs), len(dets))
        for (l_pre, _), (S, path, vec) in zip(dets, csfs):
            col = pos[l_pre]
            for sidx, amp in enumerate(vec):
                if amp != 0.0:
                    l_dst = det_sidx[sidx]
                    U[pos[l_dst], col] = amp * parities[l_dst]
            labels[col] = (S, docc, open_orbs, path)

    return leaves, U, labels


def spin_block_columns(labels: Sequence[tuple], total_spin) -> List[int]:
    """Column indices of the CSFs with total spin ``total_spin``."""
    S = Fraction(total_spin)
    return [j for j, lab in enumerate(labels) if lab[0] == S]


# ---------------------------------------------------------------------------
# Ballot (Yamanouchi) leaders and the backbone.
# ---------------------------------------------------------------------------

def is_ballot_leader(leaf: int, m: int) -> bool:
    """True iff the determinant's open-shell spin word (ascending orbital
    order, alpha = +1, beta = -1) has all partial sums nonnegative.
    Closed-shell determinants are vacuously ballot leaders."""
    occ_a = tuple(p for p in range(m) if (leaf >> p) & 1)
    occ_b = tuple(p for p in range(m) if (leaf >> (m + p)) & 1)
    _docc, _open_orbs, spins = _pattern_of(m, occ_a, occ_b)
    s = 0
    for sp in spins:
        s += 1 if sp == 0 else -1
        if s < 0:
            return False
    return True


def ballot_leaders(leaves: Sequence[int], m: int) -> List[int]:
    """The ballot (Yamanouchi) leaders among ``leaves``, sorted.  A sector has
    exactly one leader per spin-``S`` CSF, ``S`` the sector's highest weight."""
    return sorted(l for l in leaves if is_ballot_leader(int(l), m))


def ballot_embedding(m: int, leaves: Sequence[int], U_target: np.ndarray) -> List[int]:
    """Assign each target-spin CSF column its ballot leader: the ballot
    determinant carrying that column's largest-magnitude coefficient, chosen
    greedily so the assignment is a bijection.  Returns one embedding leaf per
    column (column order); these are the CSF-chart tree leaves."""
    pos = {int(l): i for i, l in enumerate(leaves)}
    ballot = ballot_leaders(leaves, m)
    used, assign = set(), {}
    order = sorted(range(U_target.shape[1]),
                   key=lambda ci: -max(abs(U_target[pos[b], ci]) for b in ballot))
    for ci in order:
        for _, b in sorted(((abs(U_target[pos[b], ci]), b) for b in ballot),
                           reverse=True):
            if b not in used:
                assign[ci] = b
                used.add(b)
                break
    if len(used) != len(ballot) or len(assign) != U_target.shape[1]:
        raise ValueError("ballot-leader assignment is not a bijection")
    return [assign[ci] for ci in range(U_target.shape[1])]


# ---------------------------------------------------------------------------
# Spin operators applied sparsely in the leaf basis (verification helpers).
# ---------------------------------------------------------------------------

def apply_s_plus(psi: np.ndarray, m: int) -> np.ndarray:
    """Apply the total raising operator ``S+ = sum_p a^dag_{p,alpha} a_{p,beta}``
    (exact blocked-JW signs) to a statevector on ``2m`` qubits."""
    psi = np.asarray(psi)
    out = np.zeros_like(psi)
    for l in np.nonzero(psi)[0]:
        l = int(l)
        for p in range(m):
            if (l >> (m + p)) & 1 and not (l >> p) & 1:
                s1 = -1.0 if bin(l & ((1 << (m + p)) - 1)).count("1") % 2 else 1.0
                l1 = l - (1 << (m + p))
                s2 = -1.0 if bin(l1 & ((1 << p) - 1)).count("1") % 2 else 1.0
                out[l1 + (1 << p)] += s1 * s2 * psi[l]
    return out


def total_spin_expectation(psi: np.ndarray, m: int) -> float:
    """``<psi| S^2 |psi>`` via ``S^2 = S- S+ + Sz (Sz + 1)`` (no dense matrix)."""
    psi = np.asarray(psi)
    val = float(np.real(np.vdot(apply_s_plus(psi, m), apply_s_plus(psi, m))))
    for l in np.nonzero(psi)[0]:
        l = int(l)
        na = bin(l & ((1 << m) - 1)).count("1")
        nb = bin(l >> m).count("1")
        sz = 0.5 * (na - nb)
        val += sz * (sz + 1.0) * float(np.abs(psi[l]) ** 2)
    return val


# ---------------------------------------------------------------------------
# Determinant-chart (gate-angle) utilities: constraints, node split, ties,
# tie Jacobian and the Woodbury-corrected inverse metric.
#
# A state of projection Sz = S has total spin exactly S iff S+ |psi> = 0, one
# linear constraint per determinant of the Sz = S + 1 sector.  The functions
# below express those constraints in the tree angles: the ballot backbone
# stays free, the remaining ``r`` angles are tied, and the metric induced on
# the free angles is the diagonal metric plus a rank-``r`` correction.
# ---------------------------------------------------------------------------

def spin_constraint_matrix(m: int, support: Sequence[int]) -> np.ndarray:
    """The ``(r, k)`` matrix ``Phi`` with ``Phi @ c = 0`` iff the state
    ``sum_j c_j |support[j]>`` is annihilated by ``S+``; rows are the reachable
    ``Sz + 1`` determinants (sorted), columns follow ``support`` order."""
    support = [int(l) for l in support]
    rows: dict = {}
    for j, l in enumerate(support):
        e = np.zeros(1 << (2 * m))
        e[l] = 1.0
        out = apply_s_plus(e, m)
        for dst in np.nonzero(out)[0]:
            rows.setdefault(int(dst), {})[j] = float(out[dst])
    dsts = sorted(rows)
    Phi = np.zeros((len(dsts), len(support)))
    for i, dst in enumerate(dsts):
        for j, v in rows[dst].items():
            Phi[i, j] = v
    return Phi


def _node_depth(i: int) -> int:
    d = 0
    while not (2 ** d - 1 <= i < 2 ** (d + 1) - 1):
        d += 1
    return d


def _subtree_leaves(i: int, n_eff: int) -> set:
    d = _node_depth(i)
    p = i - (2 ** d - 1)
    span = 2 ** (n_eff - d)
    return set(range(p * span, (p + 1) * span))


def ballot_free_tied(num_qubits: int, support: Sequence[int], m: int,
                     reorder: bool = False):
    """Split the active tree angles of the pruned tree over ``support`` into
    free and tied, by the ballot rule: a node is FREE iff a ballot leader lies
    in each of its two subtrees (the backbone), TIED otherwise.

    Returns ``(free_idx, tied_idx, leaders)`` with indices into the active
    parameter vector (the chart/circuit parameter order)."""
    support = [int(l) for l in support]
    leaders = set(ballot_leaders(support, m))
    topo = chart_topology(num_qubits, support, reorder=reorder)
    red2orig = {r: support[k] for k, r in enumerate(topo["ral"])}
    red_set = set(topo["ral"])
    free_idx, tied_idx = [], []
    for j, v in enumerate(topo["active_params"]):
        L = {red2orig[r] for r in _subtree_leaves(2 * v + 1, topo["n_eff"]) & red_set}
        R = {red2orig[r] for r in _subtree_leaves(2 * v + 2, topo["n_eff"]) & red_set}
        (free_idx if (leaders & L) and (leaders & R) else tied_idx).append(j)
    return free_idx, tied_idx, sorted(leaders)


def sector_amplitude_map(num_qubits: int, support: Sequence[int],
                         reorder: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    """Callable ``theta -> c``: the tree amplitudes over ``support`` (column
    order of :func:`spin_constraint_matrix`) at active angles ``theta``."""
    support = sorted(int(l) for l in support)
    p2c = make_polyspherical_to_cartesian(num_qubits, support, reorder=reorder)
    idx = np.asarray(support, dtype=int)

    def amp_fn(theta):
        return np.asarray(p2c(active_vals=np.asarray(theta, dtype=float)))[idx]

    return amp_fn


def constraint_values(theta, phi: np.ndarray, amp_fn) -> np.ndarray:
    """Residuals ``Phi @ c(theta)`` (zero on the spin manifold)."""
    return phi @ amp_fn(np.asarray(theta, dtype=float))


def constraint_jacobian(theta, phi: np.ndarray, amp_fn) -> np.ndarray:
    """Exact ``(r, P)`` Jacobian of the residuals in the tree angles.  Each
    amplitude is degree-1 trigonometric in each angle, so the two-term shift
    ``dc/dtheta_j = (c(theta + pi/2 e_j) - c(theta - pi/2 e_j)) / 2`` is exact."""
    theta = np.asarray(theta, dtype=float)
    J = np.zeros((phi.shape[0], theta.size))
    for j in range(theta.size):
        tp = theta.copy(); tp[j] += 0.5 * np.pi
        tm = theta.copy(); tm[j] -= 0.5 * np.pi
        J[:, j] = phi @ (amp_fn(tp) - amp_fn(tm)) / 2.0
    return J


def solve_tied(theta, tied_idx: Sequence[int], phi: np.ndarray, amp_fn,
               tol: float = 1e-13, maxit: int = 100) -> np.ndarray:
    """Damped Newton on the tied angles (free angles held) until
    ``S+ |psi> = 0``; returns the full updated angle vector."""
    theta = np.array(theta, dtype=float)
    tied = list(tied_idx)
    if not tied:
        return theta
    for _ in range(maxit):
        f = constraint_values(theta, phi, amp_fn)
        nf = np.max(np.abs(f))
        if nf < tol:
            break
        J = constraint_jacobian(theta, phi, amp_fn)[:, tied]
        step = la.lstsq(J, f, rcond=None)[0]
        a = 1.0
        for _bt in range(40):
            trial = theta.copy()
            trial[tied] -= a * step
            if np.max(np.abs(constraint_values(trial, phi, amp_fn))) < nf:
                break
            a *= 0.5
        theta[tied] -= a * step
    return theta


def tie_jacobian(theta, free_idx: Sequence[int], tied_idx: Sequence[int],
                 phi: np.ndarray, amp_fn) -> np.ndarray:
    """``d theta_tied / d theta_free = -(Phi_tied)^{-1} Phi_free`` (implicit
    function theorem), shape ``(r, n_free)``."""
    J = constraint_jacobian(theta, phi, amp_fn)
    return -la.solve(J[:, list(tied_idx)], J[:, list(free_idx)])


def woodbury_inverse_metric(w_diag, free_idx: Sequence[int],
                            tied_idx: Sequence[int], J: np.ndarray,
                            eps: float = 1e-9) -> np.ndarray:
    """Inverse of the metric induced on the free angles,
    ``g_f = D_f + J^T D_t J``, via the Woodbury identity (one ``r x r`` solve).
    ``w_diag`` is the full diagonal of Fubini--Study weights (all active
    angles), ``J`` the tie Jacobian of :func:`tie_jacobian`."""
    w = np.asarray(w_diag, dtype=float)
    Df, Dt = w[list(free_idx)], w[list(tied_idx)]
    Dfi = np.where(Df > eps, 1.0 / np.where(Df > eps, Df, 1.0), 0.0)
    JDfi = J * Dfi[None, :]
    inner = np.diag(np.where(Dt > eps, 1.0 / np.where(Dt > eps, Dt, 1.0), 0.0)) \
        + JDfi @ J.T
    return np.diag(Dfi) - JDfi.T @ la.solve(inner, JDfi)


# ---------------------------------------------------------------------------
# The spin-adapted circuit object (returned by
# MinimalCircuit.from_particle_number(..., total_spin=S); not public API).
# ---------------------------------------------------------------------------

class _SpinAdaptedCircuit(MinimalCircuit):
    """A :class:`MinimalCircuit` whose variational chart is the CSF tree.

    The circuit is the plain pruned determinant tree over the spin-``S``
    support (``self.support``, ``self.circuit``); the variational parameters
    are the angles of a binary tree over the spin-``S`` CSFs, each placed at
    its ballot leader (``self.embedding``).  The chart maps
    (:meth:`statevector`, :meth:`parameters`), the exactly diagonal
    :meth:`inverse_metric` and the corner initialisation all act in the CSF
    chart, so :func:`minimize_energy` runs unchanged and every iterate is a
    pure spin-``S`` eigenstate.  :meth:`circuit_parameters` returns the
    determinant-tree gate angles realising a given CSF-chart point.
    """

    def __init__(self, num_qubits, det_support, *, num_spatial_orbitals,
                 num_particles, total_spin, sector, csf_unitary, embedding,
                 reorder=False, encoding=None):
        super().__init__(num_qubits, det_support, complex=False, reorder=reorder)
        self.total_spin = Fraction(total_spin)
        self.sector = [int(l) for l in sector]
        self.csf_unitary = np.asarray(csf_unitary, dtype=float)
        self.embedding = [int(l) for l in embedding]
        self._encoding = encoding          # set on the encoded path; else plain JW
        self._hf_meta = (int(num_spatial_orbitals),
                         (int(num_particles[0]), int(num_particles[1])))
        self._sector_arr = np.asarray(self.sector, dtype=int)
        self._emb_arr = np.asarray(self.embedding, dtype=int)
        self._spin_adapted = True
        # lazily built CSF-chart callables
        self._csf_p2c = None
        self._csf_c2p = None
        self._csf_inv_metric_fn = None
        self._csf_topology = None

    # -- CSF chart ----------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        if self._csf_topology is None:
            self._csf_topology = chart_topology(
                self.num_qubits, self.embedding, reorder=self.reorder)
        return len(self._csf_topology["active_params"])

    def csf_amplitudes(self, theta) -> np.ndarray:
        """CSF amplitude vector ``a(theta)`` (spin-``S`` column order)."""
        if self._csf_p2c is None:
            self._csf_p2c = make_polyspherical_to_cartesian(
                self.num_qubits, self.embedding, reorder=self.reorder)
        full = np.asarray(self._csf_p2c(active_vals=np.asarray(theta, dtype=float)))
        return full[self._emb_arr]

    def statevector(self, theta, omega=None) -> np.ndarray:
        """The prepared state ``|psi> = sum U[:, j] a_j(theta)`` -- a pure
        spin-``S`` eigenstate at every ``theta`` (``omega`` unused)."""
        psi = np.zeros(2 ** self.num_qubits)
        psi[self._sector_arr] = self.csf_unitary @ self.csf_amplitudes(theta)
        return psi

    def parameters(self, statevector) -> np.ndarray:
        """CSF-chart angles of a state (projected onto the spin manifold)."""
        psi = np.asarray(statevector)
        if np.iscomplexobj(psi):
            psi = np.real(psi)
        a = self.csf_unitary.T @ psi[self._sector_arr]
        full = np.zeros(2 ** self.num_qubits)
        full[self._emb_arr] = a
        if self._csf_c2p is None:
            self._csf_c2p = make_cartesian_to_polyspherical(
                self.num_qubits, self.embedding, reorder=self.reorder)
        return np.asarray(self._csf_c2p(full), dtype=float)

    def inverse_metric(self, theta) -> np.ndarray:
        """Exactly diagonal inverse Fubini--Study metric of the CSF chart (the
        change of basis is a fixed unitary, so the tree metric is unchanged)."""
        if self._csf_inv_metric_fn is None:
            self._csf_inv_metric_fn = make_inverse_metric(
                self.num_qubits, self.embedding, reorder=self.reorder)
        return np.asarray(
            self._csf_inv_metric_fn(active_vals=np.asarray(theta, dtype=float)),
            dtype=float)

    def singular_initialise(self, theta, energy_fn, *, lr=0.15, tau=1e-10):
        return np.asarray(
            _singular_initialise(np.asarray(theta, dtype=float), energy_fn,
                                 self.num_qubits, self.embedding, lr,
                                 tau=tau, reorder=self.reorder),
            dtype=float)

    # -- determinant-tree realisation ----------------------------------------

    def circuit_parameters(self, theta) -> np.ndarray:
        """Gate angles of the determinant-tree circuit realising ``theta``
        (the classical lowering; parameter order matches ``self.circuit``)."""
        return np.asarray(
            MinimalCircuit.parameters(self, self.statevector(theta)),
            dtype=float)

    def bound_circuit(self, theta):
        """The pruned determinant-tree Qiskit circuit bound at ``theta``."""
        return self._circuit.assign_parameters(self.circuit_parameters(theta))

    # -- diagnostics ---------------------------------------------------------

    def total_spin_expectation(self, theta) -> float:
        """``<S^2>`` of the prepared state (equals ``S (S + 1)`` identically).

        The ``total_spin_expectation`` kernel assumes the blocked-JW layout, so
        for an encoded circuit the state is first decoded back to blocked JW
        (undoing :func:`encode_blocked_leaves`) before measuring."""
        m = self._hf_meta[0]
        psi = self.statevector(theta)
        if self._encoding is None:
            return total_spin_expectation(psi, m)
        psi_jw = np.zeros(2 ** (2 * m), dtype=psi.dtype)
        for q in self.sector:
            occ_a, occ_b = _interleaved_leaf_occ(int(self._encoding.sae_to_jw(q)), m)
            psi_jw[leaf_int(m, occ_a, occ_b)] = (
                _blocked_to_interleaved_parity(occ_a, occ_b) * psi[q])
        return total_spin_expectation(psi_jw, m)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MinimalCircuit(num_qubits={self.num_qubits}, "
            f"support_size={len(self.support)}, num_parameters={self.num_parameters}, "
            f"total_spin={self.total_spin})"
        )


def build_spin_adapted(num_spatial_orbitals: int, num_particles: Tuple[int, int],
                       total_spin, *, support: Optional[Sequence[int]] = None,
                       reorder: bool = False, encoding=None) -> _SpinAdaptedCircuit:
    """Construct the spin-adapted ansatz behind
    ``MinimalCircuit.from_particle_number(..., total_spin=S)``.

    Targets the highest-weight block ``Sz = S`` and therefore requires
    ``n_alpha - n_beta = 2 * total_spin``; ``support`` optionally restricts
    the sector (e.g. point-group screening), and must keep whole
    occupation-pattern blocks.

    ``encoding`` optionally lands the ansatz in a symmetry-adapted
    :class:`~quantumsymmetry.Encoding` rather than plain Jordan--Wigner: the
    intermediate-JW CSF tree is built as usual, then every Slater determinant is
    mapped to its encoded correspondent (:func:`encode_blocked_leaves`), so the
    circuit acts on ``encoding.encoded_qubits`` and prepares the same spin-``S``
    eigenstate in the encoded basis.  When ``support`` is omitted it is taken
    from the encoding's symmetry-adapted sector.
    """
    m = int(num_spatial_orbitals)
    na, nb = int(num_particles[0]), int(num_particles[1])
    S = Fraction(total_spin)
    if 2 * S != na - nb:
        raise ValueError(
            f"total_spin={total_spin} requires the highest-weight sector "
            f"Sz = S, i.e. n_alpha - n_beta = {2 * S}; got "
            f"num_particles=({na}, {nb}).  Lower-weight members of the "
            "multiplet follow from the prepared state by the fixed lowering "
            "operator and need no separate ansatz.")
    if encoding is not None:
        # With an encoding, ``support`` is given in ENCODED leaves (or omitted,
        # in which case the encoding's whole symmetry-adapted sector is used);
        # translate to blocked-JW determinants for the CSF machinery.  A subset
        # (e.g. a CAS/model-space block) restricts the prepared sector.
        enc_support = (encoding.symmetry_adapted_support((na, nb))
                       if support is None else support)
        support = []
        for q in enc_support:
            occ_a, occ_b = _interleaved_leaf_occ(int(encoding.sae_to_jw(q)), m)
            support.append(leaf_int(m, occ_a, occ_b))
    leaves, U, labels = sector_csf_unitary(m, na, nb, allowed_leaves=support)
    cols = spin_block_columns(labels, S)
    if not cols:
        raise ValueError(f"the sector contains no total-spin-{S} states.")
    U_target = U[:, cols].copy()
    embedding = ballot_embedding(m, leaves, U_target)
    leaf_arr = np.asarray(leaves, dtype=int)
    det_support = sorted(
        int(l) for l in leaf_arr[np.any(np.abs(U_target) > 1e-12, axis=1)])
    if encoding is None:
        return _SpinAdaptedCircuit(
            2 * m, det_support, num_spatial_orbitals=m, num_particles=(na, nb),
            total_spin=S, sector=leaves, csf_unitary=U_target,
            embedding=embedding, reorder=reorder)
    # Remap the CSF machinery into the encoded space: sector leaves and the
    # circuit's determinant support are relabelled, and the reordering-parity
    # signs are folded into the (still column-orthonormal) CSF unitary.
    enc_leaves, signs = encode_blocked_leaves(encoding, leaves, m)
    enc_of = {int(l): q for l, q in zip(leaves, enc_leaves)}
    U_target_enc = U_target * signs[:, None]                # diag(signs) @ U_target
    embedding_enc = [enc_of[int(l)] for l in embedding]
    det_support_enc = sorted(enc_of[int(l)] for l in det_support)
    return _SpinAdaptedCircuit(
        int(encoding.encoded_qubits), det_support_enc, num_spatial_orbitals=m,
        num_particles=(na, nb), total_spin=S, sector=enc_leaves,
        csf_unitary=U_target_enc, embedding=embedding_enc, reorder=reorder,
        encoding=encoding)
