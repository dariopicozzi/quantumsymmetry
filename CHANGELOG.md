# Changelog

## 0.3.5

- Add optional fixed spin-orbital occupations to `PeriodicEncoding` through
  `fixed_spin_orbitals={mode: occupation}`. The corresponding Boolean
  projection is combined with the selected periodic sector before qubit
  reduction, with incompatible constraints rejected explicitly.
- Preserve Qiskit mapper alignment when projection removes an operator, and
  represent fully reduced scalar operators as zero-qubit `SparsePauliOp`
  objects.

## 0.3.4

- Return bare periodic active-space Hamiltonians without adding an automatic
  particle-number penalty. Fixed-particle-number calculations should use an
  eigensolver restricted to the physical sector or initialize a
  number-conserving ansatz in that sector.

## 0.3.3

- Add optional static constrained-RPA downfolding for periodic active-space
  Hamiltonians through `PeriodicEncoding(..., downfolding="crpa")`.

## 0.3.2

- Find periodic Boolean symmetries from the finite Born–von Kármán
  space-group action restricted to the selected active orbitals, including
  operations that become commuting involutions only after restriction.

## 0.3.1

- Fix periodic Gamma folding on larger even k-point meshes by using the
  requested mesh consistently in PySCF's forward and inverse transformations.

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
