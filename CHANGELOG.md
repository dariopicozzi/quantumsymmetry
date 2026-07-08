# Changelog

## 0.3.0

Adds the binary-tree variational ansatz (`MinimalCircuit`) and its companion
tooling, alongside the existing symmetry-adapted encodings.

- **Universal binary-tree ansatz** (`MinimalCircuit`) with a closed-form
  diagonal Fubini–Study metric and a pruning compiler whose two-qubit gate
  count grows linearly in the number of active basis states.
- **Natural-gradient VQE** (`minimize_energy`) with an exact four-term
  parameter-shift rule.
- **Real- and imaginary-time evolution** (`evolve_realtime`, `project_vqd`).
- **Sector-Haar (Fubini–Study) sampling** (`sample_sector_haar`).
- **Schrieffer–Wolff dressing layer** (`make_dressing_pool`, `apply_dressing`,
  `decoupling_surrogate`).
- **Exact total-spin adaptation** via
  `MinimalCircuit.from_particle_number(..., total_spin=S)`.
- A new tutorial series, `docs/tutorials/tree_01_welcome.ipynb` through
  `tree_07_spin.ipynb`, covering each of the above.

## 0.2.x and earlier

Symmetry-adapted encodings (`Encoding`) for qubit-number reduction by
point-group and other Boolean symmetries, compatible with OpenFermion and
Qiskit. See Picozzi & Tennyson (2023), *Quantum Science and Technology* 8(3),
[doi:10.1088/2058-9565/acd86c](https://doi.org/10.1088/2058-9565/acd86c).
