"""Variational time-evolution drivers for :class:`MinimalCircuit`.

Two Layer-2 drivers consume a (complex) :class:`MinimalCircuit` and a qubit
Hamiltonian to perform real-time dynamics on the polyspherical
``(theta, omega)`` chart:

* :func:`evolve_realtime` -- McLachlan/Dirac--Frenkel **real-time TDVP**.  The
  Fubini--Study metric is diagonal in chart coordinates and the velocities
  admit a closed-form binary-tree recursion driven by parameter-shift energy
  gradients.  Integrated with Euler or RK4.

* :func:`project_vqd` -- **p-VQD**: at each step the exact one-step propagated
  state ``e^{-i H dt}|psi(theta_t)>`` is projected back onto the variational
  manifold by maximising the fidelity ``|<psi(theta)|target>|^2``, warm-started
  from the previous parameters.

Both default to noiseless statevector evaluation (using the fast ``O(|S|)``
:meth:`MinimalCircuit.statevector`).  Shot-noise / hardware backends are an
example-level concern and are intentionally not part of this core API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .tree import node_amplitudes, node_angles
from .optimize import shift_rule_gradient
from .minimal_circuit import MinimalCircuit, _to_sparse_pauli_op

__all__ = [
    "evolve_realtime",
    "RealTimeResult",
    "project_vqd",
    "ProjectionResult",
]


# ---------------------------------------------------------------------------
# Real-time TDVP.
# ---------------------------------------------------------------------------

def _tdvp_rhs(
    mc: MinimalCircuit,
    theta: np.ndarray,
    omega: np.ndarray,
    energy_func: Callable[[np.ndarray, np.ndarray], float],
    *,
    tau_singular: float = 1e-12,
    fix_global_phase: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Real-time TDVP velocities ``(theta_dot, omega_dot)`` in chart coordinates."""
    theta = np.asarray(theta, float)
    omega = np.asarray(omega, float)
    topo = mc.tree

    n_theta = len(topo["active_params"])
    n_leaves = topo["n_leaves"]
    if theta.size != n_theta:
        raise ValueError(f"theta size {theta.size} != expected {n_theta}")
    if omega.size != n_leaves:
        raise ValueError(f"omega size {omega.size} != expected {n_leaves}")

    # 1. Joint shift-rule gradient on the combined (theta, omega) vector.
    def E_of_p(p):
        return energy_func(p[:n_theta], p[n_theta:])

    p0 = np.concatenate([theta, omega])
    grad = shift_rule_gradient(E_of_p, p0)
    E_theta = grad[:n_theta]
    E_omega_grad = grad[n_theta:]            # = 2 * Im(xi_k)
    Im_xi = 0.5 * E_omega_grad

    # 2. Re(xi_k) via pi-shift of omega_k.
    E0 = energy_func(theta, omega)
    Re_xi = np.empty(n_leaves)
    for k in range(n_leaves):
        om_pi = omega.copy()
        om_pi[k] += np.pi
        Re_xi[k] = 0.25 * (E0 - energy_func(theta, om_pi))

    # 3. Tree quantities.
    angle = node_angles(theta, topo=topo)
    r = node_amplitudes(angle=angle, topo=topo)
    Pnode = r * r

    n_nodes = topo["n_nodes"]
    n_internals = topo["n_internals"]

    Im_node = np.zeros(n_nodes)
    Re_node = np.zeros(n_nodes)
    for k in range(n_leaves):
        node = topo["leaf_tree_node"][k]
        Im_node[node] = Im_xi[k]
        Re_node[node] = Re_xi[k]
    for a in reversed(range(n_internals)):
        Im_node[a] = Im_node[2 * a + 1] + Im_node[2 * a + 2]
        Re_node[a] = Re_node[2 * a + 1] + Re_node[2 * a + 2]

    # 4. theta_dot via the L/R Im(xi) sums.
    theta_dot = np.zeros(n_theta)
    for a in topo["active_params"]:
        i = topo["active_idx"][a]
        ca = np.cos(angle[a]); sa = np.sin(angle[a])
        Pa = Pnode[a]
        if Pa < tau_singular or abs(ca) < tau_singular or abs(sa) < tau_singular:
            theta_dot[i] = 0.0
            continue
        L_sum = Im_node[2 * a + 1]
        R_sum = Im_node[2 * a + 2]
        Va = -(sa / ca) * L_sum + (ca / sa) * R_sum
        theta_dot[i] = Va / Pa

    # 5. omega_dot via the Xi recursion seeded by Xi_root = E.
    Xi = np.zeros(n_nodes)
    Xi[0] = Re_node[0]
    for a in range(n_internals):
        ca = np.cos(angle[a]); sa = np.sin(angle[a])
        Pa = Pnode[a]
        if a in topo["active_idx"]:
            Ea = E_theta[topo["active_idx"][a]]
            denom = 2.0 * Pa * sa * ca
            Da = 0.0 if abs(denom) < tau_singular else Ea / denom
        else:
            Da = 0.0
        Xi[2 * a + 1] = Xi[a] - (sa * sa) * Da
        Xi[2 * a + 2] = Xi[a] + (ca * ca) * Da

    omega_dot = np.empty(n_leaves)
    for k in range(n_leaves):
        omega_dot[k] = -Xi[topo["leaf_tree_node"][k]]
    if fix_global_phase:
        omega_dot -= omega_dot.mean()

    return theta_dot, omega_dot


def _euler_step(mc, theta, omega, dt, energy_func, **kw):
    dt_t, dt_o = _tdvp_rhs(mc, theta, omega, energy_func, **kw)
    return theta + dt * dt_t, omega + dt * dt_o


def _rk4_step(mc, theta, omega, dt, energy_func, **kw):
    f = _tdvp_rhs
    k1_t, k1_o = f(mc, theta, omega, energy_func, **kw)
    k2_t, k2_o = f(mc, theta + 0.5 * dt * k1_t, omega + 0.5 * dt * k1_o, energy_func, **kw)
    k3_t, k3_o = f(mc, theta + 0.5 * dt * k2_t, omega + 0.5 * dt * k2_o, energy_func, **kw)
    k4_t, k4_o = f(mc, theta + dt * k3_t, omega + dt * k3_o, energy_func, **kw)
    theta_next = theta + (dt / 6.0) * (k1_t + 2 * k2_t + 2 * k3_t + k4_t)
    omega_next = omega + (dt / 6.0) * (k1_o + 2 * k2_o + 2 * k3_o + k4_o)
    return theta_next, omega_next


@dataclass
class RealTimeResult:
    """Trajectory produced by :func:`evolve_realtime`."""

    times: np.ndarray
    thetas: np.ndarray            # shape (n_steps + 1, n_theta)
    omegas: np.ndarray            # shape (n_steps + 1, n_leaves)
    states: Optional[np.ndarray]  # shape (n_steps + 1, 2**n) if record_states
    num_energy_evaluations: int = 0  # actual cost-function (energy-oracle) calls


def evolve_realtime(
    mc: MinimalCircuit,
    hamiltonian: Any,
    initial_state: np.ndarray,
    total_time: float,
    num_steps: int,
    *,
    method: str = "rk4",
    energy_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    tau_singular: float = 1e-12,
    fix_global_phase: bool = True,
    record_states: bool = True,
) -> RealTimeResult:
    """Integrate the real-time TDVP flow on a :class:`MinimalCircuit`.

    Parameters
    ----------
    mc:
        A ``complex=True`` :class:`MinimalCircuit` (the chart needs phases).
    hamiltonian:
        Qiskit ``SparsePauliOp`` or OpenFermion ``QubitOperator`` (JW).
    initial_state:
        Complex state vector (length ``2**num_qubits``) supported on ``mc.support``.
    total_time, num_steps:
        Evolve from ``0`` to ``total_time`` in ``num_steps`` equal steps.
    method:
        ``"rk4"`` (default) or ``"euler"``.
    energy_func:
        Custom ``E(theta, omega) -> float`` overriding the default statevector
        oracle (e.g. for shot noise).
    """
    if not mc.complex:
        raise ValueError(
            "evolve_realtime requires a complex MinimalCircuit "
            "(build with complex=True)."
        )
    if method not in ("rk4", "euler"):
        raise ValueError(f"method must be 'rk4' or 'euler'; got {method!r}.")

    if energy_func is None:
        H = _to_sparse_pauli_op(hamiltonian, mc.num_qubits).to_matrix(sparse=True)

        def energy_func(theta, omega):  # type: ignore[misc]
            psi = mc.statevector(theta, omega)
            return float(np.real(np.vdot(psi, H @ psi)))

    # Count every actual cost-function (energy-oracle) evaluation, including the
    # shift-rule queries inside the TDVP right-hand side.
    _eval_count = {"n": 0}
    _energy_oracle = energy_func

    def energy_func(theta, omega):  # type: ignore[misc]
        _eval_count["n"] += 1
        return _energy_oracle(theta, omega)

    theta, omega = mc.parameters(np.asarray(initial_state, dtype=complex))
    theta = np.asarray(theta, float).copy()
    omega = np.asarray(omega, float).copy()

    stepper = _rk4_step if method == "rk4" else _euler_step
    dt = float(total_time) / int(num_steps)
    kw = dict(tau_singular=tau_singular, fix_global_phase=fix_global_phase)

    times = [0.0]
    thetas = [theta.copy()]
    omegas = [omega.copy()]
    states = [mc.statevector(theta, omega)] if record_states else None

    for s in range(int(num_steps)):
        theta, omega = stepper(mc, theta, omega, dt, energy_func, **kw)
        times.append((s + 1) * dt)
        thetas.append(theta.copy())
        omegas.append(omega.copy())
        if states is not None:
            states.append(mc.statevector(theta, omega))

    return RealTimeResult(
        times=np.asarray(times),
        thetas=np.asarray(thetas),
        omegas=np.asarray(omegas),
        states=np.asarray(states) if states is not None else None,
        num_energy_evaluations=int(_eval_count["n"]),
    )


# ---------------------------------------------------------------------------
# p-VQD.
# ---------------------------------------------------------------------------

# Four-term shift rule (RY tree angles) + two-term shift (per-leaf phases).
_SHIFT_ALPHA = (2.0 + np.sqrt(2.0)) / 4.0
_SHIFT_BETA = (2.0 - np.sqrt(2.0)) / 4.0
_SHIFT_S1 = np.pi / 4.0
_SHIFT_S2 = 3.0 * np.pi / 4.0
_SHIFT_S_OMEGA = np.pi / 2.0


def _fidelity_split(mc: MinimalCircuit) -> Tuple[int, int]:
    """Return ``(n_theta, n_omega)`` for the flat parameter vector."""
    n_omega = len(mc.support)
    n_theta = mc.num_parameters - n_omega
    return n_theta, n_omega


@dataclass
class ProjectionResult:
    """Trajectory produced by :func:`project_vqd`."""

    times: np.ndarray
    thetas: np.ndarray            # shape (n_steps, n_params)
    states: np.ndarray            # shape (n_steps, 2**n)
    step_infidelity: np.ndarray   # 1 - F at each step
    step_n_evals: np.ndarray
    cumulative_n_evals: int


def _shift_rule_fidelity_jac(fidelity, n_theta, n_omega, nfev_box):
    """Analytic gradient of ``1 - F`` for the pruned-tree ansatz."""
    def jac(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        grad = np.zeros_like(x)
        for i in range(n_theta):
            pp1 = x.copy(); pp1[i] += _SHIFT_S1
            pm1 = x.copy(); pm1[i] -= _SHIFT_S1
            pp2 = x.copy(); pp2[i] += _SHIFT_S2
            pm2 = x.copy(); pm2[i] -= _SHIFT_S2
            f_pp1 = fidelity(pp1); f_pm1 = fidelity(pm1)
            f_pp2 = fidelity(pp2); f_pm2 = fidelity(pm2)
            nfev_box[0] += 4
            grad[i] = -(_SHIFT_ALPHA * (f_pp1 - f_pm1) - _SHIFT_BETA * (f_pp2 - f_pm2))
        for k in range(n_omega):
            j = n_theta + k
            pp = x.copy(); pp[j] += _SHIFT_S_OMEGA
            pm = x.copy(); pm[j] -= _SHIFT_S_OMEGA
            f_pp = fidelity(pp); f_pm = fidelity(pm)
            nfev_box[0] += 2
            grad[j] = -0.5 * (f_pp - f_pm)
        return grad
    return jac


def project_vqd(
    mc: MinimalCircuit,
    hamiltonian: Any,
    initial_state: np.ndarray,
    times: np.ndarray,
    *,
    initial_parameters: Optional[np.ndarray] = None,
    inner_maxiter: int = 30,
    inner_ftol: float = 1e-9,
    gradient: str = "fd",
) -> ProjectionResult:
    """Statevector p-VQD: project exact one-step propagation onto the manifold.

    At step ``k`` the target is ``e^{-i H dt} |psi(theta_{k-1})>`` (computed by
    sparse ``expm_multiply``) and ``theta_k`` maximises the fidelity to it,
    warm-started from ``theta_{k-1}``.

    Parameters
    ----------
    mc:
        A ``complex=True`` :class:`MinimalCircuit`.
    hamiltonian:
        Qiskit ``SparsePauliOp`` or OpenFermion ``QubitOperator`` (JW).
    initial_state:
        Complex state vector at ``times[0]`` supported on ``mc.support``.
    times:
        Monotone array of times; the first entry is the initial time.
    gradient:
        ``"fd"`` (scipy finite-difference Jacobian) or ``"shift"`` (analytic
        parameter-shift gradient).
    """
    import scipy.sparse.linalg as sla
    from scipy.optimize import minimize

    if not mc.complex:
        raise ValueError("project_vqd requires a complex MinimalCircuit (complex=True).")
    if gradient not in ("fd", "shift"):
        raise ValueError(f"gradient must be 'fd' or 'shift'; got {gradient!r}.")

    H = _to_sparse_pauli_op(hamiltonian, mc.num_qubits).to_matrix(sparse=True)
    n_theta, n_omega = _fidelity_split(mc)

    def fast_state(x: np.ndarray) -> np.ndarray:
        return mc.statevector(x[:n_theta], x[n_theta:])

    # Initial parameters.
    if initial_parameters is not None:
        theta = np.asarray(initial_parameters, dtype=float).copy()
    else:
        th, om = mc.parameters(np.asarray(initial_state, dtype=complex))
        theta = np.concatenate([np.asarray(th, float), np.asarray(om, float)])

    times = np.asarray(times, dtype=float)
    n_steps = len(times)
    thetas = np.zeros((n_steps, theta.size), dtype=float)
    states = np.zeros((n_steps, 2 ** mc.num_qubits), dtype=complex)
    infids = np.zeros(n_steps, dtype=float)
    nfevs = np.zeros(n_steps, dtype=int)

    target_holder: List[np.ndarray] = [np.asarray(initial_state, dtype=complex)]

    def fidelity(x: np.ndarray) -> float:
        psi = fast_state(x)
        return float(np.abs(np.vdot(target_holder[0], psi)) ** 2)

    thetas[0] = theta
    states[0] = fast_state(theta)
    infids[0] = 1.0 - fidelity(theta)

    cumul = 0
    for k in range(1, n_steps):
        dt = float(times[k] - times[k - 1])
        target_holder[0] = sla.expm_multiply(-1j * dt * H, states[k - 1])

        nfev = [0]

        def cost(x: np.ndarray) -> float:
            nfev[0] += 1
            return 1.0 - fidelity(x)

        if gradient == "shift":
            jac = _shift_rule_fidelity_jac(fidelity, n_theta, n_omega, nfev)
        else:
            jac = None

        res = minimize(
            cost, x0=theta, method="L-BFGS-B", jac=jac,
            options={"maxiter": int(inner_maxiter), "ftol": float(inner_ftol)},
        )
        theta = np.asarray(res.x, dtype=float)
        thetas[k] = theta
        states[k] = fast_state(theta)
        infids[k] = float(res.fun)
        nfevs[k] = int(nfev[0])
        cumul += int(nfev[0])

    return ProjectionResult(
        times=times,
        thetas=thetas,
        states=states,
        step_infidelity=infids,
        step_n_evals=nfevs,
        cumulative_n_evals=int(cumul),
    )
