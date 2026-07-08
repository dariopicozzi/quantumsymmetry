"""Exact Fubini--Study (sector-Haar) sampling on a pruned-tree support set.

Sampling a pure state uniformly (Haar / Fubini--Study measure) on the
projective subspace spanned by ``K`` computational-basis states is
equivalent to drawing squared amplitudes ``(|c_1|^2, ..., |c_K|^2)`` from
the flat Dirichlet distribution ``Dir(1, ..., 1)`` and independent uniform
leaf phases.  The Dirichlet aggregation property factorises this exactly
over the binary tree of the polyspherical chart: at every ACTIVE internal
node the fraction of squared amplitude routed to the *right* subtree is an
independent draw

    t ~ Beta(n_R, n_L),        theta = arcsin(sqrt(t)) in [0, pi/2],

where ``n_L`` / ``n_R`` are the numbers of active leaves under the left /
right child.  FIXED nodes route all weight to one side (factor 1) and
INACTIVE nodes carry none, so neither is sampled.  Each draw therefore
costs ``O(#active parameters)`` -- no determinant evaluation, rejection,
or MCMC -- even when ``K`` is exponentially large.

This *is* sampling from the FS volume element ``sqrt(det g)``: the
diagonal polyspherical metric factorises the volume form into the same
per-node 1D densities that the stick-breaking identity produces.

Two samplers are provided:

* :func:`sample_sector_haar` -- the tree (stick-breaking) sampler above.
* :func:`sample_sector_haar_oracle` -- the Gaussian-vector oracle
  (``z ~ CN(0,1)^K``, normalise, invert the chart), exact by rotational
  invariance but requiring all ``K`` amplitudes explicitly.  Used to
  validate the tree sampler.

Both return chart coordinates ``(theta, omega)`` in the conventions of the
COMPLEX chart (`theta` ordered as ``chart_topology(...)['active_params']``;
``omega`` per-leaf in support order, canonical gauge ``omega[0] = 0``).
"""

import numpy as np

from .tree import chart_topology, _leaves_under

__all__ = [
    'tree_beta_parameters',
    'sample_sector_haar',
    'sample_sector_haar_oracle',
]


def tree_beta_parameters(num_qubits, support, reorder=False):
    """Return the per-active-node Beta parameters ``(n_R, n_L)``.

    For each active internal node (in ``chart_topology`` ``active_params``
    order) count the active leaves under its right and left child subtrees
    in the effective (constant-bit-reduced) tree.  These are the parameters
    of the Beta distribution of the squared-amplitude fraction routed to
    the right subtree under the sector-Haar measure.
    """
    topo = chart_topology(num_qubits, list(support), reorder=reorder)
    n_eff = topo['n_eff']
    ral_set = set(topo['ral'])
    params = []
    for a in topo['active_params']:
        n_left = len(_leaves_under(2 * a + 1, n_eff) & ral_set)
        n_right = len(_leaves_under(2 * a + 2, n_eff) & ral_set)
        params.append((n_right, n_left))
    return params, topo


def sample_sector_haar(num_qubits, support, n_samples=1, rng=None,
                       reorder=False):
    """Draw chart coordinates of sector-Haar random states (tree sampler).

    Parameters
    ----------
    num_qubits : int
        Total number of qubits.
    support : sequence of int
        Active computational-basis states spanning the sector.
    n_samples : int
        Number of independent samples.
    rng : numpy.random.Generator, optional
        Source of randomness (defaults to ``np.random.default_rng()``).
    reorder : bool
        Must match the ``reorder`` flag of the target circuit/chart.

    Returns
    -------
    theta : ndarray, shape (n_samples, n_active)
        Amplitude angles in ``active_params`` order, each in ``[0, pi/2]``.
    omega : ndarray, shape (n_samples, K)
        Per-leaf phases in support order, canonical gauge ``omega[:, 0] = 0``.
    """
    if rng is None:
        rng = np.random.default_rng()
    beta_params, topo = tree_beta_parameters(num_qubits, support,
                                             reorder=reorder)
    n_active = len(beta_params)
    K = topo['n_leaves']

    theta = np.empty((n_samples, n_active))
    for j, (n_r, n_l) in enumerate(beta_params):
        t = rng.beta(n_r, n_l, size=n_samples)
        theta[:, j] = np.arcsin(np.sqrt(t))

    omega = np.zeros((n_samples, K))
    if K > 1:
        omega[:, 1:] = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, K - 1))
    return theta, omega


def sample_sector_haar_oracle(num_qubits, support, n_samples=1, rng=None,
                              reorder=False):
    """Draw sector-Haar chart coordinates via the Gaussian-vector oracle.

    Exact by rotational invariance of the complex Gaussian: draw
    ``z ~ CN(0, 1)^K``, normalise, scatter onto the support, and invert the
    complex chart.  Cost is ``O(K)`` per sample (explicit amplitudes), so
    this is a validation oracle, not the scalable method.

    Returns ``(theta, omega)`` with the same shapes/conventions as
    :func:`sample_sector_haar`.
    """
    # Imported here to avoid a circular import at module load time.
    from .metric import make_cartesian_to_polyspherical

    if rng is None:
        rng = np.random.default_rng()
    support = [int(s) for s in support]
    K = len(support)
    c2p = make_cartesian_to_polyspherical(num_qubits, support, complex=True,
                                          reorder=reorder)

    topo = chart_topology(num_qubits, support, reorder=reorder)
    n_active = len(topo['active_params'])
    theta = np.empty((n_samples, n_active))
    omega = np.empty((n_samples, K))
    dim = 1 << num_qubits
    for s in range(n_samples):
        z = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        z /= np.linalg.norm(z)
        psi = np.zeros(dim, dtype=complex)
        psi[support] = z
        th, om = c2p(psi)
        theta[s] = th
        omega[s] = np.mod(om, 2.0 * np.pi)
        omega[s, 0] = 0.0
    return theta, omega
